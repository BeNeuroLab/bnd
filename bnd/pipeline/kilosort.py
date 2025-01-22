from pathlib import Path

from bnd import set_logging
from bnd.config import Config, _load_config

logger = set_logging(__name__)


# try:
#     from kilosort import run_kilosort
# except ImportError as e:
#     # Log the error and raise it
#     logger.error("Failed to import the required package: %s", e)
#     raise ImportError(
#         "Could not import spike sorting functionality. Make sure kilosort is installed"
#         " in the environment"
#     ) from e


def run_kilosort_on_stream(probe_folder_path, recording_path, session_path) -> None:
    """
    Runs kilosort4 on a raw SpikeGLX data and saves to output folder within the directory

    Parameters
    ----------
    probe_folder_path : Path
        Path to probe folder with raw SpikeGLX data
    recording_path : Path
        Path to recording directory with probe folders
    session_path : Path
        Path to the session directory

    Returns
    -------

    """

    # TODO: Improve this
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

    # _ = run_kilosort(
    #     settings=sorter_params,
    #     probe_name='neuropixPhase3B1_kilosortChanMap.mat',
    #     data_dir=probe_folder_path,
    #     results_dir=ksort_output_path,
    #     save_preprocessed_copy=True,
    # )
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
        Path to recording directory with probe folders
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

        run_kilosort_on_stream(probe_path, recording_path, session_path)

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
    # Check kilosort is installed in environment
    # try:
    #     import torch
    #
    #     if torch.cuda.is_available():
    #         logger.info(
    #             f"CUDA is available. GPU device: {torch.cuda.get_device_name(0)}"
    #         )
    #     else:
    #         logger.warning("CUDA is not available. GPU computations will not be enabled.")
    # except ImportError as e:
    #     # Log the error and raise it
    #     logger.error("Failed to import the required package: %s", e)
    #     raise ImportError(
    #         "Could not import spike sorting functionality. Make sure torch is installed"
    #         " in the environment"
    #     ) from e

    config = _load_config()

    if isinstance(session_path, str):
        session_path = Path(session_path)

    ephys_recording_folders = config.get_subdirectories_from_pattern(
        session_path, "*g?"
    )
    for recording_path in ephys_recording_folders:
        logger.info(f"Processing recording: {recording_path.name}")
        run_kilosort_on_recording(
            config,
            recording_path,
            session_path,
        )

    return
