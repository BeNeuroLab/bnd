import os
from pathlib import Path

import numpy as np
from rich import print

from bnd import set_logging
from bnd.config import _load_config
from bnd.pipeline.kilosort import run_kilosort_on_session
from bnd.pipeline.nwbtools.anipose_interface import AniposeInterface
from bnd.pipeline.nwbtools.beneuro_converter import BeNeuroConverter
from bnd.pipeline.nwbtools.multiprobe_kilosort_interface import (
    MultiProbeKiloSortInterface,
)

logger = set_logging(__name__)
config = _load_config()


def _try_adding_kilosort_to_source_data(source_data: dict, session_path: Path) -> None:
    if any(session_path.glob("**/spike_times.npy")) and any(
        config.get_subdirectories_from_pattern(session_path, "*_ksort")
    ):
        # Check if there is more than one recording
        ksorted_folders = config.get_subdirectories_from_pattern(
            session_path / f"{session_path.name}_ksort", "*_g?"
        )
        if len(ksorted_folders) > 1:
            while True:
                user_input = input(
                    f"Found {len(ksorted_folders)} ksorted recordings. Please select one {np.arange(len(ksorted_folders))}: "
                )
                try:
                    ksorted_folder_path = ksorted_folders[int(user_input)]
                    break
                except Exception as e:
                    print(f"Invalid input: {e}. Enter a valid integer")
        elif len(ksorted_folders) == 1:
            ksorted_folder_path = ksorted_folders[0]

        # Attempt to wrap interface
        try:
            # TODO: Add custom map options
            MultiProbeKiloSortInterface(ksorted_folder_path)
            source_data.update(
                Kilosort={
                    "ksorted_folder_path": ksorted_folder_path,  # For neuroconv consistency
                }
            )
            return int(user_input) if len(ksorted_folders) > 1 else None

        # warn if we can't read it
        except Exception as e:
            logger.warning(f"Problem loading Kilosort data: {str(e)}")

    elif len(config.get_subdirectories_from_pattern(session_path, "*_g?")) > 0:
        # if there's no kilosort output found,
        # check if there could be one because the raw data exists
        logger.warning(
            "You might want to run Kilosort. Found ephys data but no Kilosort output."
        )
    else:
        logger.warning("No ephys or kilosort data found")

    return


def _try_adding_anipose_to_source_data(source_data: dict, session_path: Path):
    csv_paths = list(session_path.glob("**/*3dpts_angles.csv"))

    if len(csv_paths) == 0:
        logger.warning("No pose estimation data found.")
        return

    if len(csv_paths) > 1:
        raise FileExistsError(
            f"More than one pose estimation csv file " f"found: {csv_paths}"
        )

    csv_path = csv_paths[0]
    try:
        AniposeInterface(csv_path, session_path)
    except Exception as e:
        logger.warning(f"Problem loading anipose data: {str(e)}")
    else:
        source_data.update(
            Anipose={
                "csv_path": str(csv_path),
                "raw_session_path": str(session_path),
            }
        )


def run_nwb_conversion(session_path: Path, kilosort_flag: bool, custom_map: bool):
    logger.info(f"Running nwb conversion on session: {session_path.name}")

    # TODO: Throw question on which recording to use if it finds many
    config = _load_config()

    # Check session_path is Path object
    if isinstance(session_path, str):
        session_path = Path(session_path)

    # Set output path
    nwb_file_output_path = session_path / f"{session_path.name}.nwb"
    if nwb_file_output_path.exists():
        response = (
            input(
                f"File '{nwb_file_output_path.name}' already exists. Do you want to overwrite it? (y/n): "
            )
            .strip()
            .lower()
        )
        if response != "y":
            logger.warning(f"Aborting nwb conversion")
            return
        else:
            os.remove(nwb_file_output_path)

    # Run kilosort if needed
    if kilosort_flag:
        run_kilosort_on_session(session_path)

    # specify where the data should be read from by the converter
    source_data = dict(
        PyControl={
            "file_path": str(session_path),
        },
    )

    recording_to_process = _try_adding_kilosort_to_source_data(source_data, session_path)
    _try_adding_anipose_to_source_data(source_data, session_path)

    # finally, run the conversion
    converter = BeNeuroConverter(source_data, recording_to_process, verbose=False)

    metadata = converter.get_metadata()

    metadata["NWBFile"].deep_update(
        lab="Be.Neuro Lab",
        institution="Imperial College London",
    )

    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwb_file_output_path,
    )
    logger.info(f"Successfully saved file {nwb_file_output_path.name}")

    return
