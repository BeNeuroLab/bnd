from pathlib import Path

from bnd import set_logging
from bnd.config import _load_config
from bnd.pipeline.nwbtools.base_converter import BeNeuroConverter

# from bnd.pipeline.kilosort import run_kilosort_on_session

logger = set_logging(__name__)
config = _load_config()


def _try_adding_kilosort_to_source_data(source_data: dict, session_path: Path) -> None:
    if any(session_path.glob("**/spike_times.npy")):
        # TODO: Check you only run it on one recording
        # if it looks like kilosort has been run
        # then try loading it
        try:
            # TODO: Add custom map options
            MultiProbeKiloSortInterface(str(session_path))
        # warn if we can't read it
        except Exception as e:
            logger.warning(f"Problem loading Kilosort data: {str(e)}")
        # if we can, then add it to the conversion
        else:
            source_data.update(
                Kilosort={
                    "folder_path": str(session_path),  # For neuroconv consistency
                }
            )

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
            f"More than one pose estimation HDF file " f"found: {csv_paths}"
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

    config = _load_config()

    # Check session_path is Path object
    if isinstance(session_path, str):
        session_path = Path(session_path)

    # Run kilosort if needed
    # if kilosort_flag:
        # run_kilosort_on_session(session_path)

    # specify where the data should be read from by the converter
    source_data = dict(PyControl={"file_path": str(session_path), }, )
    breakpoint()
    _try_adding_kilosort_to_source_data(source_data, session_path)
    _try_adding_anipose_to_source_data(source_data, session_path)

    # finally, run the conversion
    converter = BeNeuroConverter(source_data, verbose=False)

    metadata = converter.get_metadata()

    metadata["NWBFile"].deep_update(lab="Be.Neuro Lab",
        institution="Imperial College London", )

    converter.run_conversion(metadata=metadata, nwbfile_path=nwb_file_output_path, )
    logger.info(f"Successfully saved file {nwb_file_output_path.name}")



    return
