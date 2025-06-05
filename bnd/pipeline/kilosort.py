import os
from configparser import ConfigParser
from pathlib import Path

import torch
from kilosort import run_kilosort
from kilosort.utils import PROBE_DIR, download_probes

from ..logger import set_logging
from ..config import Config, _load_config
from ..config import find_file

logger = set_logging(__name__)


def read_metadata(filepath: Path) -> dict:
    """Parse a section-less INI file (eg NPx metadata file) and return a dictionary of key-value pairs."""
    with open(filepath, "r") as f:
        content = f.read()
        # Inject a dummy section header
        content_with_section = "[dummy_section]\n" + content

    config = ConfigParser()
    config.optionxform = str  # disables lowercasing
    config.read_string(content_with_section)

    return dict(config.items("dummy_section"))


def add_entry_to_metadata(filepath: Path, tag: str, value: str) -> None:
    """
    Add or update a tag=value entry in the NPx metadata.
    """
    with open(filepath, "a") as f:  # append mode
        f.write(f"{tag}={value}\n")


def _read_probe_type(meta_file_path: str) -> str:
    meta = read_metadata(meta_file_path)
    probe_type_val = meta["imDatPrb_type"]
    if int(probe_type_val) == 0:
        probe_type = (
            "neuropixPhase3B1_kilosortChanMap.mat"  # Neuropixels Phase3B1 (staggered)
        )
    elif int(probe_type_val) == 2013:
        probe_type = "NP2_kilosortChanMap.mat"
    else:
        raise ValueError(
            "Probe type not recogised. It appears to be different from Npx 1.0 or 2.0"
        )
    return probe_type


def _fix_session_ap_metadata(meta_file_path: Path) -> None:
    """to inject `fileSizeBytes` and `fileTimeSecs` if they are missing"""
    meta = read_metadata(meta_file_path)
    if "fileSizeBytes" not in meta:
        datafile_path = find_file(meta_file_path.parent, "ap.bin")[0]
        data_size = os.path.getsize(datafile_path)
        add_entry_to_metadata(meta_file_path, "fileSizeBytes", str(data_size))
        data_duration = data_size / int(meta["nSavedChans"]) / 2 / int(meta["imSampRate"])
        add_entry_to_metadata(meta_file_path, "fileTimeSecs", str(data_duration))
        logger.warning(
            f"AP Metadata missing values: Injected fileSizeBytes: {data_size} and fileTimeSecs: {data_duration}"
        )
        _fix_session_lf_metadata(meta_file_path)


def _fix_session_lf_metadata(meta_ap_path: Path) -> None:
    """to inject `fileSizeBytes` and `fileTimeSecs` to the LFP metadata, if they are missing"""
    meta_file_path = meta_ap_path.parent / (meta_ap_path.stem.replace("ap", "lf") + ".meta")
    meta = read_metadata(meta_file_path)
    if "fileSizeBytes" not in meta:
        datafile_path = find_file(meta_file_path.parent, "lf.bin")[0]
        data_size = os.path.getsize(datafile_path)
        add_entry_to_metadata(meta_file_path, "fileSizeBytes", str(data_size))
        data_duration = data_size / int(meta["nSavedChans"]) / 2 / int(meta["imSampRate"])
        add_entry_to_metadata(meta_file_path, "fileTimeSecs", str(data_duration))
        logger.warning(
            f"LFP Metadata missing values: Injected fileSizeBytes: {data_size} and fileTimeSecs: {data_duration}"
        )


def run_kilosort_on_stream(
    config: Config,
    probe_folder_path: Path,
    recording_path: Path,
    session_path: Path,
    probe_name: str = "neuropixPhase3B1_kilosortChanMap.mat",
) -> None:
    """
    Runs kilosort4 on a raw SpikeGLX data and saves to output folder within the directory

    Parameters
    ----------
    probe_folder_path : Path
        Path to probe folder with raw SpikeGLX data (i.e., _imec0 or _imec1)
    recording_path : Path
        Path to recording directory with probe folders (i.e., _g0 or _g1)
    session_path : Path
        Path to the session directory
    probe_name : str
        Type of neuropixels probe

    Returns
    -------

    """
    meta_file_path = config.get_subdirectories_from_pattern(probe_folder_path, "*ap.meta")[0]

    sorter_params = {
        "n_chan_bin": int(read_metadata(meta_file_path)["nSavedChans"]),
    }

    ksort_output_path = (
        session_path
        / f"{session_path.name}_ksort"
        / recording_path.name
        / probe_folder_path.name
    )
    ksort_output_path.mkdir(parents=True, exist_ok=True)

    if not PROBE_DIR.exists():
        logger.info("Probe directory not found, downloading probes")
        download_probes()

    if any(PROBE_DIR.glob(f"{probe_name}")):
        # Sometimes the gateway can throw an error so just double check.
        download_probes()

    # Check if the metadata file is complete
    # when SpikeGLX crashes, metadata misses some values.
    _fix_session_ap_metadata(meta_file_path)
    # Find out which probe type we have
    probe_name = _read_probe_type(meta_file_path)

    _ = run_kilosort(
        settings=sorter_params,
        probe_name=probe_name,
        data_dir=probe_folder_path,
        results_dir=ksort_output_path,
        save_preprocessed_copy=False,
        verbose_console=False,
    )
    return


def run_kilosort_on_recording(
    config: Config, recording_path: Path, session_path: Path
) -> None:
    """
    Run kilosort on a single recording within a session

    Parameters
    ----------
    config : Config
        Configuration class
    recording_path : Path
        Path to recording directory with probe folders (i.e., _g0 or _g1)
    session_path : Path
        Path to the session directory

    Returns
    -------

    """

    if isinstance(recording_path, str):
        raw_recording_path = Path(recording_path)

    if not recording_path.is_relative_to(config.LOCAL_PATH / "raw"):
        raise ValueError(f"{recording_path} is not in {config.LOCAL_PATH / 'raw'}")

    probe_paths = config.get_subdirectories_from_pattern(recording_path, "*_imec?")
    for probe_path in probe_paths:
        logger.info(f"Processing probe: {probe_path.name[-5:]}")

        run_kilosort_on_stream(config, probe_path, recording_path, session_path)

    return


def run_kilosort_on_session(session_path: Path) -> None:
    """
    Entry function to run kilosort4 on a single session recording

    Parameters
    ----------
    session_path : Path:
        Path to the session directory

    Returns
    -------

    """
    config = _load_config()

    if isinstance(session_path, str):
        session_path = Path(session_path)

    kilosort_output_folders = config.get_subdirectories_from_pattern(session_path, "*_ksort")

    if not any(session_path.rglob("*.bin")):
        logger.warning(
            f"No ephys files found. Consider running `bnd dl {session_path.name} -e"
        )

    elif kilosort_output_folders:
        logger.warning(f"Kilosort output already exists. Skipping kilosort call")

    else:
        ephys_recording_folders = config.get_subdirectories_from_pattern(session_path, "*_g?")
        # Check kilosort is installed in environment
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. GPU device: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA is not available. GPU computations will not be enabled.")
            if len(ephys_recording_folders) > 1:
                raise ValueError(
                    "It seems you are trying to run kilosort without GPU. Look at the README on instrucstions of how to do this. "
                )

        for recording_path in ephys_recording_folders:
            logger.info(f"Processing recording: {recording_path.name}")
            run_kilosort_on_recording(
                config,
                recording_path,
                session_path,
            )

    return
