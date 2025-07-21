import os
from pathlib import Path

import numpy as np
from rich import print

from ..config import _load_config, find_file
from ..logger import set_logging
from .kilosort import run_kilosort_on_session
from .nwbtools.anipose_interface import AniposeInterface
from .nwbtools.beneuro_converter import BeNeuroConverter
from .nwbtools.multiprobe_kilosort_interface import MultiProbeKiloSortInterface
from .nwbtools.multiprobe_lfp_interface import MultiProbeNpxLFPInterface

logger = set_logging(__name__)
config = _load_config()


def _select_folder_from_multiple(folders):
    if len(folders) > 1:
        while True:
            user_input = input(
                f"Found {len(folders)} recordings. Please select one {np.arange(len(folders))}. Default [1]: "
            )
            if "y" in user_input.lower():
                user_input = 1
            try:
                folder = folders[int(user_input)]
                break
            except Exception as e:
                print(f"Invalid input: {e}. Enter a valid integer")

    elif len(folders) == 1:
        folder = folders[0]

    return folder


def _try_adding_npx_lfp_to_source_data(source_data: dict, session_path: Path) -> None:

    lfp_files = find_file(session_path, ".lf.bin")
    if not lfp_files:
        logger.warning("No LFP files found in session")
        return

    ephys_folders = config.get_subdirectories_from_pattern(session_path, "*_g?")
    ephys_folder_path = _select_folder_from_multiple(ephys_folders)

    try:
        MultiProbeNpxLFPInterface(ephys_folder_path)
    except Exception as e:
        logger.warning(f"Problem loading lfp data data: {str(e)}")
    else:
        source_data.update(
            LFP={
                "ephys_folder_path": ephys_folder_path,
            }
        )

    return


def _try_adding_kilosort_to_source_data(
    source_data: dict, session_path: Path, custom_map: bool
) -> None:
    if any(session_path.glob("**/spike_times.npy")) and any(
        config.get_subdirectories_from_pattern(session_path, "*_ksort")
    ):
        # Check if there is more than one recording
        ksorted_folders = sorted(
            config.get_subdirectories_from_pattern(
                session_path / f"{session_path.name}_ksort", "*_g?"
            )
        )
        if len(ksorted_folders) > 1:
            while True:
                user_input = input(
                    f"Found {len(ksorted_folders)} ksorted recordings. Please select one {np.arange(len(ksorted_folders))}. Default [1]: "
                )
                if "y" in user_input.lower():
                    user_input = 1
                try:
                    ksorted_folder_path = ksorted_folders[int(user_input)]
                    break
                except Exception as e:
                    print(f"Invalid input: {e}. Enter a valid integer")
        elif len(ksorted_folders) == 1:
            ksorted_folder_path = ksorted_folders[0]

        # Attempt to wrap interface
        # try:
        MultiProbeKiloSortInterface(ksorted_folder_path, custom_map)
        source_data.update(
            Kilosort={
                "ksorted_folder_path": ksorted_folder_path,  # For neuroconv consistency
                "custom_map": custom_map,
            }
        )
        return int(user_input) if len(ksorted_folders) > 1 else None

        # warn if we can't read it
        # except Exception as e:
        # logger.warning(f"Problem loading Kilosort data: {str(e)}")

    elif len(config.get_subdirectories_from_pattern(session_path, "*_g?")) > 0:
        # if there's no kilosort output found,
        # check if there could be one because the raw data exists
        logger.warning(
            "You might want to run Kilosort. Found ephys data but no Kilosort output."
        )
    else:
        logger.warning("No ephys or kilosort data found")

    return


def _try_adding_anipose_to_source_data(source_data: dict, session_path: Path) -> None:
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
        AniposeInterface(csv_path)
    except Exception as e:
        logger.warning(f"Problem loading anipose data: {str(e)}")
    else:
        source_data.update(
            Anipose={
                "csv_path": str(csv_path),
            }
        )
    return


def run_nwb_conversion(session_path: Path, kilosort_flag: bool, custom_map: bool, lfp: bool):
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
        if "y" not in response:
            logger.warning("Aborting nwb conversion")
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

    recording_to_process = _try_adding_kilosort_to_source_data(
        source_data, session_path, custom_map
    )
    _try_adding_anipose_to_source_data(source_data, session_path)

    if lfp:
        _try_adding_npx_lfp_to_source_data(source_data, session_path)

    converter = BeNeuroConverter(source_data, recording_to_process, verbose=False)

    metadata = converter.get_metadata()

    metadata["Subject"].deep_update(
        subject_id=config.get_animal_name(session_name=session_path.name),
        sex="F",
        species="Mus musculus",
    )

    metadata["NWBFile"].deep_update(
        lab="Be.Neuro Lab",
        institution="Imperial College London",
    )

    # finally, run the conversion
    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwb_file_output_path,
    )
    logger.info(f"Successfully saved file {nwb_file_output_path.name}")

    return
