from pathlib import Path

import torch
from kilosort import run_kilosort
from kilosort.utils import PROBE_DIR, download_probes

from bnd import set_logging
from bnd.config import Config, _load_config

logger = set_logging(__name__)


def _read_probe_type(meta_file_path: str) -> str:
    with open(meta_file_path, "r") as meta_file:
        for line in meta_file:
            if "imDatPrb_type" in line:
                _, value = line.strip().split("=")
                break

        if int(value) == 0:
            probe_type = (
                "neuropixPhase3B1_kilosortChanMap.mat"  # Neuropixels Phase3B1 (staggered)
            )
        elif int(value) == 21:
            probe_type = "NP2_kilosortChanMap.mat"
        else:
            raise ValueError(
                "Probe type not recogised. It appears to be different from Npx 1.0 or 2.0"
            )
    return probe_type


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

    sorter_params = {
        "n_chan_bin": 385,
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

    # Find out which probe type we have
    meta_file_path = config.get_subdirectories_from_pattern(probe_folder_path, "*ap.meta")
    probe_name = _read_probe_type(str(meta_file_path[0]))

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

    kilosort_output_folders = config.get_subdirectories_from_pattern(
        session_path, "*_ksort"
    )

    if not any(session_path.rglob("*.bin")):
        logger.warning(
            f"No ephys files found. Consider running `bnd dl {session_path.name} -e"
        )

    elif kilosort_output_folders:
        logger.warning(f"Kilosort output already exists. Skipping kilosort call")

    else:
        # Check kilosort is installed in environment
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. GPU device: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA is not available. GPU computations will not be enabled.")
            return
        for recording_path in ephys_recording_folders:
            logger.info(f"Processing recording: {recording_path.name}")
            run_kilosort_on_recording(
                config,
                recording_path,
                session_path,
            )

    return
