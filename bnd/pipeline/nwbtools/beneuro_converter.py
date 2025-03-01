from pathlib import Path

import numpy as np
import spikeinterface.extractors as se
from neuroconv import NWBConverter
from neuroconv.datainterfaces import SpikeGLXRecordingInterface  # PhySortingInterface
from neuroconv.tools.signal_processing import get_rising_frames_from_ttl

from bnd import set_logging
from bnd.config import _load_config
from bnd.pipeline.nwbtools.anipose_interface import AniposeInterface
from bnd.pipeline.nwbtools.multiprobe_kilosort_interface import (
    MultiProbeKiloSortInterface,
)
from bnd.pipeline.nwbtools.pycontrol_interface import PyControlInterface

logger = set_logging(__name__)
config = _load_config()


def chunked_first_rise(memmap_array: np.memmap, chunk_size: int = 1_000):
    """
    Get the first rising edge in a 1D memmap array containing the sync signal generated by PyControl.

    Loading the whole last channel into memory could be costly, so go through the memmap in chunks.

    Parameters
    ----------
    memmmmap_array : np.memmap
        memmap array containing the sync signal generated by PyControl
    chunk_size : int
        size of the chunks to load into memory at one time

    Returns
    -------
    index of the first rising edge in the sync signal
    -1 if no rising edge is found
    """
    for start_idx in range(0, len(memmap_array), chunk_size):
        chunk = memmap_array[start_idx : (start_idx + chunk_size)]
        rising_frames = get_rising_frames_from_ttl(chunk)
        if len(rising_frames) > 0:
            return start_idx + rising_frames[0]

    return -1


class BeNeuroConverter(NWBConverter):
    """
    Converter class for data from the BeNeuro.Lab

    Example use:

    ```python
    source_data = dict(
        PyControl = {
            "file_path" : str(session_folder_path)
        },
        Kilosort = {
            "folder_path" : str(session_folder_path),
        },
        AnimalProfile = {
            "session_path" : str(session_folder_path),
        },
        Anipose = {
            "csv_path" : str(path_to_pose_estimation_csv_file),
            "raw_session_path" : str(session_folder_path),
        },
    )

    converter = BeNeuroConverter(source_data)

    converter.run_conversion(
        metadata=converter.get_metadata(),
        nwbfile_path=nwb_file_output_path,
    )
    ```
    """

    # this contains all possible interfaces
    # if source_data doesn't have one of them, it won't be used
    data_interface_classes = {
        "Kilosort": MultiProbeKiloSortInterface,
        "PyControl": PyControlInterface,
        "Anipose": AniposeInterface,
    }

    def __init__(self, source_data, recording_to_process=None, verbose=True):
        super().__init__(source_data, verbose)
        self.recording_to_process = recording_to_process

    # logger.info("Extracted available interfaces")

    def temporally_align_data_interfaces(self):
        adjusting_times = {}

        if "PyControl" in self.data_interface_objects:
            # start with PyControlInterface
            pycont_start_time = self.data_interface_objects[
                "PyControl"
            ].get_first_rising_edge_time()

            self.data_interface_objects["PyControl"].adjust_timestamps(pycont_start_time)
            adjusting_times["PyControl"] = pycont_start_time

        if "Kilosort" in self.data_interface_objects:
            # then do kilosort
            # just to make the names shorter
            multikilo = self.data_interface_objects["Kilosort"]

            raw_session_path = Path(
                self.data_interface_objects["PyControl"].source_data["file_path"]
            )
            spikeglx_output_folder_path = config.get_subdirectories_from_pattern(
                raw_session_path, "*_g?"
            )

            if len(spikeglx_output_folder_path) > 1:
                assert self.recording_to_process is not None
                spikeglx_output_folder_path = spikeglx_output_folder_path[
                    self.recording_to_process
                ]
            elif len(spikeglx_output_folder_path) == 1:
                spikeglx_output_folder_path = spikeglx_output_folder_path[0]

            for probe_name, kilosort_interface in zip(
                multikilo.probe_names, multikilo.kilosort_interfaces
            ):
                # Old enforcements from one recording per session
                # ap_paths = list(raw_session_path.glob(f"**/*{probe_name}.ap.bin"))
                # assert len(ap_paths) == 1
                # this is needed for the alignment to work
                # it doesn't need the sync channel inside here, I'll extract that later

                # ===================================== NOTE =====================================
                # In a previous version of neurconv we found that this function wasn't actually
                # shifting the timestamps of the recording, but this seems to be the case. Also, before,
                # This was needed for the allignment to work but now it doesn seem to be the case so
                # I will remove it all together
                # kilosort_interface.register_recording(
                #     # SpikeGLXRecordingInterface(ap_paths[0]),
                #     SpikeGLXRecordingInterface(
                #         folder_path=str(spikeglx_output_folder_path),
                #         stream_id=f"{probe_name}.ap",
                #     )
                # )
                # ===============================================================================

                # this is used to get the sync channel's values
                # and figure out when the first rising edge is
                rec_with_sync_channel = se.read_spikeglx(
                    spikeglx_output_folder_path,  # Used to be raw_session_path
                    stream_name=f"{probe_name}.ap",
                    load_sync_channel=True,
                )

                # Find the first rising edge without having to read the
                # whole array first, which can be time consuming in my experience
                first_rising_frame = chunked_first_rise(
                    rec_with_sync_channel.get_traces()[1:, -1], chunk_size=1_000
                )
                first_rise_seconds = (
                    first_rising_frame / rec_with_sync_channel.sampling_frequency
                )

                # first_rise_seconds is when the clock starts, so this times -1
                # is when the PyControl recording started relative to the clock
                kilosort_interface.set_aligned_starting_time(-first_rise_seconds)

                adjusting_times[f"Kilosort {probe_name}"] = first_rise_seconds * 1000

        logger.info(f"Interface adjusted with time values: {adjusting_times}")
