import os
import json
import shutil
import subprocess
import tempfile
import textwrap
from configparser import ConfigParser
from pathlib import Path

from ..logger import set_logging
from ..config import Config, _load_config
from ..config import find_file

logger = set_logging(__name__)


_KILOSORT_RUNNER_CODE = textwrap.dedent(
    """
    import json
    import sys
    from pathlib import Path

    params_path = Path(sys.argv[1])
    params = json.loads(params_path.read_text())

    from kilosort import run_kilosort
    from kilosort.utils import PROBE_DIR, download_probes

    probe_name = params["probe_name"]

    if not PROBE_DIR.exists():
        download_probes()

    if not any(PROBE_DIR.glob(probe_name)):
        download_probes()

    run_kilosort(
        settings=params["settings"],
        probe_name=probe_name,
        data_dir=params["data_dir"],
        results_dir=params["results_dir"],
        save_preprocessed_copy=params.get("save_preprocessed_copy", False),
    )
"""
).strip()


def _get_kilosort_env_name() -> str:
    return (
        os.environ.get("BND_KILOSORT_ENV")
        or os.environ.get("KILOSORT_CONDA_ENV")
        or "kilosort"
    )


def _find_conda_runner() -> str:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe

    for candidate in ("conda", "mamba", "micromamba"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise FileNotFoundError(
        "Could not find a conda runner executable (tried CONDA_EXE, conda, mamba, micromamba)."
    )


def _run_in_conda_env(
    env_name: str, args: list[str], *, capture_output: bool = False
) -> subprocess.CompletedProcess:
    runner = _find_conda_runner()
    cmd = [runner, "run", "-n", env_name, *args]

    # Workaround for WSL/DrvFs temp-file oddities (e.g., ftruncate -> ENOENT) when
    # TEMP/TMP point to `/mnt/c/...`. Force a sane temp dir for subprocesses.
    env = os.environ.copy()
    if Path("/tmp").exists():
        env["TMPDIR"] = "/tmp"
        env["TEMP"] = "/tmp"
        env["TMP"] = "/tmp"
    try:
        return subprocess.run(
            cmd,
            check=True,
            capture_output=capture_output,
            text=capture_output,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command failed in conda env '{env_name}': {cmd} (exit code {e.returncode})."
        ) from e


def _check_kilosort_cuda(env_name: str) -> tuple[bool, str | None]:
    code = textwrap.dedent(
        """
        import torch

        if torch.cuda.is_available():
            print("CUDA_AVAILABLE")
            print(torch.cuda.get_device_name(0))
        else:
            print("CUDA_NOT_AVAILABLE")
        """
    ).strip()

    script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            script_path = Path(f.name)

        proc = _run_in_conda_env(
            env_name, ["python", str(script_path)], capture_output=True
        )
    except Exception:
        logger.debug("Could not check CUDA in env '%s'", env_name, exc_info=True)
        return False, None
    finally:
        if script_path:
            try:
                script_path.unlink(missing_ok=True)
            except Exception:
                pass

    lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    if not lines:
        return False, None

    if lines[0] == "CUDA_AVAILABLE":
        return True, lines[1] if len(lines) > 1 else None

    return False, None


def _run_kilosort_in_env(
    *,
    env_name: str,
    settings: dict,
    probe_name: str,
    data_dir: Path,
    results_dir: Path,
) -> None:
    payload = dict(
        settings=settings,
        probe_name=probe_name,
        data_dir=str(data_dir),
        results_dir=str(results_dir),
        save_preprocessed_copy=False,
    )

    tmp_path: Path | None = None
    runner_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(payload, tmp)
            tmp_path = Path(tmp.name)

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as runner:
            runner.write(_KILOSORT_RUNNER_CODE)
            runner_path = Path(runner.name)

        _run_in_conda_env(
            env_name, ["python", str(runner_path), str(tmp_path)]
        )

    except Exception as e:
        raise RuntimeError(
            f"Failed to run Kilosort in the separate conda env '{env_name}'. "
            "Make sure it exists and has the `kilosort` package installed. "
            "You can override the env name via BND_KILOSORT_ENV."
        ) from e

    finally:
        for p in (tmp_path, runner_path):
            if p:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass


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

    # Check if the metadata file is complete
    # when SpikeGLX crashes, metadata misses some values.
    _fix_session_ap_metadata(meta_file_path)
    # Find out which probe type we have
    probe_name = _read_probe_type(meta_file_path)

    env_name = _get_kilosort_env_name()
    _run_kilosort_in_env(
        env_name=env_name,
        settings=sorter_params,
        probe_name=probe_name,
        data_dir=probe_folder_path,
        results_dir=ksort_output_path,
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
        env_name = _get_kilosort_env_name()
        cuda_available, device_name = _check_kilosort_cuda(env_name)
        if cuda_available:
            if device_name:
                logger.info(f"CUDA is available in '{env_name}'. GPU device: {device_name}")
            else:
                logger.info(f"CUDA is available in '{env_name}'.")
        else:
            logger.warning(
                f"CUDA is not available in '{env_name}'. GPU computations will not be enabled."
            )
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
