import warnings
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
from ndx_pose import PoseEstimation, PoseEstimationSeries
from neuroconv.basetemporalalignmentinterface import BaseTemporalAlignmentInterface
from neuroconv.tools.signal_processing import get_rising_frames_from_ttl
from neuroconv.utils import DeepDict
from pynwb import NWBFile

from bnd import set_logging

logger = set_logging(__name__)


class AniposeInterface(BaseTemporalAlignmentInterface):
    DEFAULT_FPS = 100

    keypoint_names = (
        "shoulder_center",
        "left_shoulder",
        "left_paw",
        "right_shoulder",
        "right_elbow",
        "right_paw",
        "hip_center",
        "left_knee",
        "left_ankle",
        "left_foot",
        "right_knee",
        "right_ankle",
        "right_foot",
        "tail_base",
        "tail_middle",
        "tail_tip",
        "left_elbow",
        "left_wrist",
        "right_wrist",
    )

    angle_names_and_references = (
        ("left_elbow_angle", ["left_shoulder", "left_elbow", "left_wrist"]),
        ("right_elbow_angle", ["right_shoulder", "right_elbow", "right_wrist"]),
        ("left_knee_angle", ["hip_center", "left_knee", "left_ankle"]),
        ("right_knee_angle", ["hip_center", "right_knee", "right_ankle"]),
        ("left_ankle_angle", ["left_knee", "left_ankle", "left_foot"]),
        ("right_ankle_angle", ["right_knee", "right_ankle", "right_foot"]),
    )

    def __init__(self, csv_path: Path):
        super().__init__()

        self.csv_path = Path(csv_path)
        self.pose_data = self.load_anipose_from_csv()

    def _add_to_behavior_module(self, beh_obj, nwbfile: NWBFile) -> None:
        behavior_module = nwbfile.processing.get("behavior")

        if behavior_module is None:
            behavior_module = nwbfile.create_processing_module(
                "behavior", "processed behavioral data"
            )

        behavior_module.add(beh_obj)

    def get_original_timestamps(self) -> np.ndarray:
        raise ValueError(f"This functionality is deprecated. We always assume default FPS")
        # return self.load_timestamps_from_spikeglx()

    def get_timestamps(self) -> np.ndarray:
        return self.get_original_timestamps()

    def set_aligned_timestamps(self):
        raise NotImplementedError

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: Optional[DeepDict] = None,
        stub_test: bool = False,
        use_default_fps: bool = True,
    ):
        # Alignment: As cameras start recording when PyControl sends them the signal at t=0,
        # and in theory sends a signal with DEFAULT_FPS frequency, set the default option for the
        # timing of the frames to use `starting_time` and `rate` instead of explicit timestamps.
        if use_default_fps:
            timestamps = None
            starting_time = 0.0
            rate = float(self.DEFAULT_FPS)

        elif not use_default_fps:
            timestamps = self.get_original_timestamps()
            starting_time = None
            rate = None

        keypoint_series_objects = []
        for keypoint_name in self.keypoint_names:
            keypoint_series = PoseEstimationSeries(
                name=keypoint_name,
                description=f"Marker placed at {keypoint_name.replace('_', ' ')}",
                data=self.pose_data[
                    [f"{keypoint_name}_x", f"{keypoint_name}_y", f"{keypoint_name}_z"]
                ].to_numpy(),
                unit="mm",
                reference_frame="(0, 0, 0) is hip_center's median across all frames",
                timestamps=timestamps,
                starting_time=starting_time,
                rate=rate,
                confidence=np.full(self.n_frames, np.nan),
                confidence_definition="Filled with nan because we don't have an estimate.",
            )
            keypoint_series_objects.append(keypoint_series)

        for angle_name, angle_reference in self.angle_names_and_references:
            angle_array = self.pose_data[[f"{angle_name}"]].to_numpy()
            angle_series = PoseEstimationSeries(
                name=angle_name,
                data=np.concatenate(
                    (angle_array, np.zeros((angle_array.shape[0], 1))), axis=1
                ),
                description="Angle information. Second dimension is zeros since since minimum"
                " 2D array is needed for PoseEstimationSeries",
                unit="degrees",
                reference_frame=f"Triangulation of keypoints: {angle_reference}",
                timestamps=timestamps,
                starting_time=starting_time,
                rate=rate,
                confidence=np.full(self.n_frames, np.nan),
                confidence_definition="Filled with nan because we don't have an estimate.",
            )
            keypoint_series_objects.append(angle_series)

        pose_estimation = PoseEstimation(
            name="Pose estimation",
            pose_estimation_series=keypoint_series_objects,
            description="Estimated positions selected parts of the animal's body.",
        )

        self._add_to_behavior_module(pose_estimation, nwbfile)

    def load_anipose_from_h5(self) -> np.ndarray:
        """
        Load the array containing the pose estimation from the HDF5 output of sleap-anipose
        """
        warnings.warn(
            "load_anipose_from_h5() is deprecated and will be removed in a "
            "future version. Please use load_anipose_from_csv() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        with h5py.File(self.h5_path, "r") as file:
            assert file["tracks"].shape[1] == 1
            pose_data = file["tracks"][:, 0, :, :]

        return pose_data

    def load_anipose_from_csv(self) -> pd.DataFrame:
        """
        Load pose estimation results from a CSV file where each keypoint and angle
        has its own column.
        """
        pose_data = pd.read_csv(self.csv_path)
        return pose_data

    @property
    def n_frames(self) -> int:
        return self.pose_data.shape[0]

    def load_timestamps_from_spikeglx(self) -> np.ndarray:
        """
        WARNING. DEPRECATED FUNCTION
        """
        if self.raw_session_path is None:
            raise ValueError(
                "Cannot load timestamps from SpikeGLX recording because "
                "the path to the session's raw data was not provided, "
                "most likely when creating BeNeuroConverter."
            )

        # stream_names = get_ap_stream_names(
        #     _find_spikeglx_recording_folders_in_session(self.raw_session_path)[0]
        # )

        # if len(stream_names) == 0:
        #     raise FileNotFoundError(
        #         f"Could not find SpikeGLX .ap streams in {self.raw_session_path}"
        #     )

        # logger.info("Setting pose estimation timestamps using pulse signal from SpikeGLX...")

        # rising_edges_dict = {}
        # for stream_name in stream_names:
        #     rec_with_sync_channel = se.read_spikeglx(
        #         self.raw_session_path, stream_name=stream_name, load_sync_channel=True
        #     )

        #     last_channel = np.array(rec_with_sync_channel.get_traces()[1:, -1])
        #     rising_frames = get_rising_frames_from_ttl(last_channel)
        #     rising_edges_sec = rising_frames / rec_with_sync_channel.sampling_frequency
        #     rising_edges_dict[stream_name] = rising_edges_sec

        # for rising_edges_sec in rising_edges_dict.values():
        #     assert rising_edges_sec.size == self.n_frames

        # mid_timestamps_sec = sum(rising_edges_dict.values()) / len(rising_edges_dict)

        # # our synchronization time is the first rising edge, so that has to be at t=0
        # mid_timestamps_sec -= mid_timestamps_sec[0]

        return  # mid_timestamps_sec
