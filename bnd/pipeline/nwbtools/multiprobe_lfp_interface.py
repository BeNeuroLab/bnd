"""
Module for lfp extraction
"""

from pathlib import Path
from typing import Literal, Optional

from neuroconv.datainterfaces import SpikeGLXRecordingInterface
from neuroconv.tools.spikeinterface import add_recording_to_nwbfile
from neuroconv.utils import DeepDict
from pynwb import NWBFile

from ...config import _load_config

config = _load_config()


class MultiProbeNpxLFPInterface(SpikeGLXRecordingInterface):
    """
    Class for hadling lfps from multi npx spikeglx recording
    """

    def __init__(self, ephys_folder_path: Path, es_key: str = "ElectricalSeries"):
        """Multiprobe lfp interface

        Parameters
        ----------
        ephys_folder_path : Path
            path to ephys folders
        es_key : str, optional
            name for the series in the nwb object, by default "ElectricalSeries"
        """
        self.ephys_folder_path = ephys_folder_path
        self.es_key = es_key
        self.probe_folder_paths = sorted(
            config.get_subdirectories_from_pattern(self.ephys_folder_path, "*imec*")
        )
        self.spikeglx_lfp_interfaces = [
            SpikeGLXRecordingInterface(
                folder_path=str(folder_path),
                stream_id=f"{folder_path.name[-5:]}.lf",
                verbose=False,
            )
            for folder_path in self.probe_folder_paths
        ]

    def get_metadata(self) -> DeepDict:
        return DeepDict()

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict | None = None,
        stub_test: bool = False,
        starting_time: float | None = None,
        write_as: Literal["raw", "lfp", "processed"] = "lfp",
        write_electrical_series: bool = True,
        iterator_type: str | None = "v2",
        iterator_opts: dict | None = None,
        always_write_timestamps: bool = False,
    ):
        """Method to overwrite neuroconv and add each interface separately

        Parameters
        ----------
        nwbfile : NWBFile
            _description_
        metadata : dict | None, optional
            _description_, by default None
        stub_test : bool, optional
            _description_, by default False
        starting_time : float | None, optional
            _description_, by default None
        write_as : Literal[&quot;raw&quot;, &quot;lfp&quot;, &quot;processed&quot;], optional
            _description_, by default "lfp"
        write_electrical_series : bool, optional
            _description_, by default True
        iterator_type : str | None, optional
            _description_, by default "v2"
        iterator_opts : dict | None, optional
            _description_, by default None
        always_write_timestamps : bool, optional
            _description_, by default False
        """
        for spike_glx_interface, probe_folder_path in zip(
            self.spikeglx_lfp_interfaces, self.probe_folder_paths
        ):
            recording = spike_glx_interface.recording_extractor

            metadata = metadata or spike_glx_interface.get_metadata()
            metadata["Ecephys"][spike_glx_interface.es_key] = dict(
                name=f"{probe_folder_path.name[-5:]}",
                description=f"Npx LFP of probe {probe_folder_path.name[-5:]}",
            )

            add_recording_to_nwbfile(
                recording=recording,
                nwbfile=nwbfile,
                metadata=metadata,
                starting_time=starting_time,
                write_as=write_as,
                write_electrical_series=write_electrical_series,
                es_key=spike_glx_interface.es_key,
                iterator_type=iterator_type,
                iterator_opts=iterator_opts,
                always_write_timestamps=always_write_timestamps,
            )
