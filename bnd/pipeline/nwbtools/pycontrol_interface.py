# NOTE currently I need the Session class to read pycontrol files
# what if pycontrol-interface worked by getting a Session object
# or some data that can be parsed and validated with pydantic?
# then I can specify what it expects to get from each experiment

import warnings
from pathlib import Path

import dateutil.tz
import numpy as np
from neuroconv.basetemporalalignmentinterface import BaseTemporalAlignmentInterface
from neuroconv.utils import DeepDict
from pydantic import FilePath
from pynwb import NWBFile
from pynwb.behavior import BehavioralEvents, Position, SpatialSeries
from pynwb.epoch import TimeIntervals

from ...logger import set_logging

from .pycontrol_data_import import Event, Print, Session, State

logger = set_logging(__name__)


class PyControlInterface(BaseTemporalAlignmentInterface):
    def __init__(self, file_path: FilePath):
        super().__init__(file_path=file_path)
        self.reload_session()

    def reload_session(self) -> None:
        sessions = []
        for txt_file in Path(self.source_data["file_path"]).glob("*.txt"):
            try:
                sessions.append(Session(txt_file, int_subject_IDs=False))
            except StopIteration:
                pass
        if len(sessions) != 1:
            raise ValueError(f"Expected 1 PyControl .txt file, found {len(sessions)}")

        self.session = sessions[0]

    def get_original_timestamps(self) -> np.ndarray:
        raise NotImplementedError

    def get_timestamps(self) -> np.ndarray:
        raise NotImplementedError

    def set_aligned_timestamps(self):
        raise NotImplementedError

    def get_first_rising_edge_time(self) -> int:
        """
        In newer sessions the time of the first trigger is saved in a print event
        and can be used to adjust times such that this event is at 0.
        In older sessions this is not saved, so we just assume it's 0.
        """
        default_trigger_time = 0
        t = self.session.times.get("before_camera_trigger", default_trigger_time)
        try:
            # see if it's an array
            len(t)
        except TypeError:
            # it's a number, so return it
            return t
        else:
            assert len(t) == 1, "There should be a single 'before_camera_trigger' event"
            return t[0]

    def adjust_timestamps(self, start_time: int) -> None:
        self.session.events = [
            Event(ev.time - start_time, ev.name) for ev in self.session.events
        ]
        self.session.states = [
            State(s.time - start_time, s.name, s.duration) for s in self.session.states
        ]
        self.session.print_data = [
            Print(p.time - start_time, p.name, p.value) for p in self.session.print_data
        ]

        for k in self.session.times.keys():
            self.session.times[k] -= start_time

        for k in self.session.analog_data.keys():
            self.session.analog_data[k][:, 0] -= start_time

    def _get_pos_timestamps_data(self, motion_sensor: str) -> np.ndarray:
        """Get motion sensore data and timestamptes

        Parameters
        ----------
        motion_sensor : str
            Name of motion sensore x or y (e.g., "MotSen1-X" or ""MotSen1-y")

        Returns
        -------
        time : np.ndarray
        data : np.ndarray
        """
        if motion_sensor not in ["MotSen1-X", "MotSen1-Y"]:
            raise ValueError(
                f"motion sensor: {motion_sensor} not a valid option (['MotSen1-X', 'MotSen1-Y'])"
            )

        time, data = self.session.analog_data[f"{motion_sensor}"][:, [0, 1]].T
        return time, data

    def _get_spatial_series(self, motion_sensor: str):
        if motion_sensor not in ["MotSen1-X", "MotSen1-Y"]:
            raise ValueError(
                f"motion sensor: {motion_sensor} not a valid option (['MotSen1-X', 'MotSen1-Y'])"
            )
        time, data = self._get_pos_timestamps_data(motion_sensor)
        spatial_series_obj = SpatialSeries(
            name=f"{motion_sensor}",
            description=f"Ball position as measured by PyControl ({motion_sensor})",
            data=data,
            timestamps=time.astype(float),
            reference_frame="(0,0) is what?",  # TODO
        )
        return spatial_series_obj

    def _add_to_behavior_module(self, beh_obj, nwbfile: NWBFile) -> None:
        # behavior_module = nwbfile.processing.get(
        #    "behavior",
        #    nwbfile.create_processing_module("behavior", "processed behavioral data")
        # )
        behavior_module = nwbfile.processing.get("behavior")

        if behavior_module is None:
            behavior_module = nwbfile.create_processing_module(
                "behavior", "processed behavioral data"
            )

        behavior_module.add(beh_obj)

    def add_position(self, nwbfile: NWBFile) -> None:

        spatial_series_obj_motion_sensor_x = self._get_spatial_series("MotSen1-X")
        spatial_series_obj_motion_sensor_y = self._get_spatial_series("MotSen1-Y")

        self._add_to_behavior_module(
            Position(
                spatial_series=[
                    spatial_series_obj_motion_sensor_x,
                    spatial_series_obj_motion_sensor_y,
                ]
            ),
            nwbfile,
        )

    def add_print_events(self, nwbfile: NWBFile):
        print_events = BehavioralEvents(name="print_events")

        for print_item in self.session.print_items:
            print_events.create_timeseries(
                name=print_item,
                data=[it.value for it in self.session.print_data if it.name == print_item],
                timestamps=[
                    float(it.time)
                    for it in self.session.print_data
                    if it.name == print_item
                ],
                unit="",
            )

        self._add_to_behavior_module(print_events, nwbfile)

    def add_behavioral_events(self, nwbfile: NWBFile) -> None:
        behavioral_events = BehavioralEvents(name="behavioral_events")

        behavioral_events.create_timeseries(
            name="behavioral events",
            data=[e.name for e in self.session.events],
            timestamps=[float(e.time) for e in self.session.events],
            unit="",
        )

        self._add_to_behavior_module(behavioral_events, nwbfile)

    def add_behavioral_states(self, nwbfile: NWBFile) -> None:
        behavioral_states = TimeIntervals(
            name="behavioral_states",
            description="intervals when each PyControl state was active",
        )

        behavioral_states.add_column(
            name="state_name", description="Name of the state as defined in PyControl"
        )

        try:
            session_length = self.session.times["session_timer"][0]
        except:
            warnings.warn("Could not find explicit session length, using latest timestamp.")

            session_length = max(
                [
                    e.time
                    for coll in [
                        self.session.events,
                        self.session.states,
                        self.session.print_data,
                    ]
                    for e in coll
                ]
            )

        for state in self.session.states:
            duration = (
                state.duration
                if state.duration is not None
                else session_length - state.time
            )

            behavioral_states.add_row(
                start_time=float(state.time),
                stop_time=float(state.time + duration),
                state_name=state.name,
            )

        self._add_to_behavior_module(behavioral_states, nwbfile)

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: DeepDict) -> None:
        self.add_behavioral_states(nwbfile)
        self.add_behavioral_events(nwbfile)
        self.add_print_events(nwbfile)
        try:
            self.add_position(nwbfile)
        except Exception as e:
            logger.warning(f"Error parsing motion sensors: {e}")

    def get_metadata(self) -> DeepDict:
        metadata = DeepDict()

        metadata["NWBFile"]["session_description"] = self.session.task_name
        metadata["NWBFile"]["session_start_time"] = self.session.datetime.astimezone(
            dateutil.tz.gettz("Europe/London")
        )

        return metadata
