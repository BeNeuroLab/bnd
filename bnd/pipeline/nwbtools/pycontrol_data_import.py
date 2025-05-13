# Modified PyControl data loading class
# Copied from @JoannaChang's behavior_analysis repo

import os
import re
from collections import namedtuple
from datetime import datetime

import numpy as np

from bnd import set_logging

logger = set_logging(__name__)


Event = namedtuple("Event", ["time", "name"])
State = namedtuple("State", ["time", "name", "duration"])
Print = namedtuple("Print", ["time", "name", "value"])

# ----------------------------------------------------------------------------------
# Session class
# ----------------------------------------------------------------------------------


class Session:
    """Import data from a pyControl file and represent it as an object with attributes:
    - file_name
    - experiment_name
    - task_name
    - setup_ID
    - CPI
        Counts per interval
    - subject_ID
        If argument int_subject_IDs is True, suject_ID is stored as an integer,
        otherwise subject_ID is stored as a string.
    - datetime
        The date and time that the session started stored as a datetime object.
    - datetime_string
        The date and time that the session started stored as a string of format 'YYYY-MM-DD HH:MM:SS'
    - state_IDs
        A dictionary with keys that are the names of the framework states and corresponding values
    - event_IDs
        A dictionary with keys that are the names of the framework events and corresponding values

    #TODO: clean up
    - events
        A list of all framework events in the order they occured.
        Each entry is a namedtuple with fields 'time' & 'name', such that you can get the
        name and time of event/state entry x with x.name and x.time respectively.
    - states
        A list of all framework states in the order they occured.
        Each entry is a namedtuple with fields 'time', 'name' & 'duration', such that you can get the name, time and duration of state x with x.name, x.time and x.duration respectively.
    - times
        A dictionary with keys that are the names of the framework events and states and
        corresponding values which are Numpy arrays of all the times (in milliseconds since the
        start of the framework run) at which each event/state entry occured.
    - print_data
        A list of all the lines output by print statements during the framework run.
        Each entry is a namedtuple with fields 'time', 'name' & 'value', such that you can get the
        name, time and value of print statement x with x.name, x.time and x.value respectively.
    - analog_data
        A dictionary of sensor data.
        Each entry is a numpy array (tsteps x 2) with the first column being the time (ms) and the second column being the sensor value.
    """

    def __init__(self, file_path, int_subject_IDs: bool = True, verbose: bool = False):
        self.verbose = verbose

        # Load lines from file.
        self.file_path = file_path

        with open(file_path, "r") as f:
            if self.verbose:
                logger.info("Importing data file: " + os.path.split(file_path)[1])
            all_lines = [line.strip() for line in f.readlines() if line.strip()]

        # Extract and store session information.
        self.file_name = os.path.split(file_path)[1]

        info_lines = [line[2:] for line in all_lines if line[0] == "I"]

        self.experiment_name = next(
            line for line in info_lines if "Experiment name" in line
        ).split(" : ")[1]
        self.task_name = next(line for line in info_lines if "Task name" in line).split(
            " : "
        )[1]
        self.setup_ID = next(line for line in info_lines if "Setup ID" in line).split(
            " : "
        )[1]
        subject_ID_string = next(line for line in info_lines if "Subject ID" in line).split(
            " : "
        )[1]
        datetime_string = next(line for line in info_lines if "Start date" in line).split(
            " : "
        )[1]

        if int_subject_IDs:  # Convert subject ID string to integer.
            self.subject_ID = int("".join([i for i in subject_ID_string if i.isdigit()]))
        else:
            self.subject_ID = subject_ID_string

        self.datetime = datetime.strptime(datetime_string, "%Y/%m/%d %H:%M:%S")
        self.datetime_string = self.datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Extract and store session data.

        self.state_IDs = eval(next(line for line in all_lines if line[0] == "S")[2:])
        self.event_IDs = eval(next(line for line in all_lines if line[0] == "E")[2:])

        data_lines = [line[2:].split(" ") for line in all_lines if line[0] == "D"]

        self.times = {}
        self._set_events(data_lines)
        self._set_states(data_lines)
        self._set_print_data(all_lines)
        self._set_analog_data()

    def _set_events(self, data_lines):
        """Set events and times dictionary."""

        event_ID2name = {v: k for k, v in {**self.event_IDs}.items()}
        self.events = [
            Event(int(dl[0]), event_ID2name[int(dl[1])])
            for dl in data_lines
            if int(dl[1]) in event_ID2name.keys()
        ]

        for event_name in event_ID2name.values():
            self.times[event_name] = np.array(
                [ev.time for ev in self.events if ev.name == event_name]
            )

    def _set_states(self, data_lines):
        """Set states and times dictionary."""
        state_ID2name = {v: k for k, v in {**self.state_IDs}.items()}

        # Get state durations
        state_info = [
            (int(dl[0]), state_ID2name[int(dl[1])])
            for dl in data_lines
            if int(dl[1]) in state_ID2name.keys()
        ]

        state_durations = np.diff([time for time, _ in state_info])
        state_durations = np.append(state_durations, None)

        self.states = [
            State(time, name, duration)
            for (time, name), duration in zip(state_info, state_durations)
        ]

        for state_name in state_ID2name.values():
            self.times[state_name] = np.array(
                [st.time for st in self.states if st.name == state_name]
            )

    def _set_print_data(self, all_lines):
        """Set print data and times dictionary."""
        # Extract and store print data.
        print_lines = [line[2:] for line in all_lines if line[0] == "P"]
        self.print_data = []
        self.print_items = []
        for line in print_lines:
            # ignore if line does not follow regex: time, value, name
            matched = re.match(r"(\d+) (.*), ([\w'-]+)", line)
            if matched:
                try:
                    value = int(matched.groups()[1])
                except ValueError:
                    value = matched.groups()[1]
                time = int(matched.groups()[0])
                name = matched.groups()[2]
                self.print_data.append(Print(time, name, value))
                self.print_items.append(name)
        self.print_items = np.unique(self.print_items)

        for item in self.print_items:
            self.times[item] = np.array([p.time for p in self.print_data if p.name == item])
        try:
            # Get CPI
            self.CPI = [item.value for item in self.print_data if item.name == "CPI"][0]
        except:
            if self.verbose:
                logger.info("CPI not defined!")

    def get_led_directions(self):
        """Return list of led positions in the order they were presented."""
        return [item.value for item in self.print_data if item.name == "LED_direction"]

    def get_event(self, event_name):
        """Return list of events with specified name."""
        return [ev for ev in self.events if ev.name == event_name]

    def get_state(self, state_name):
        """Return list of states with specified name."""
        return [st for st in self.states if st.name == state_name]

    def get_next_state(self, state_name):
        """Return list of states after each state with specified name."""
        state_idx = np.where([st.name == state_name for st in self.states])[0]
        if self.states[-1].name == state_name:  # Last state is specified state
            return [self.states[i + 1] for i in state_idx[:-1]] + [None]
        else:
            return [self.states[i + 1] for i in state_idx]

    def _set_analog_data(self):
        """Return dictionary of analog data from analog files in the same folder as the session file."""

        # Import any analog files.
        file_dir = os.path.dirname(self.file_path)
        file_name = os.path.split(self.file_path)[1]
        analog_files = [
            f
            for f in os.listdir(file_dir)
            if file_name.split(".")[0] in f and f != file_name
        ]

        self.analog_data = {}

        for analog_file in analog_files:
            analog_name = analog_file[len(file_name.split(".")[0]) + 1 : -4]
            with open(os.path.join(file_dir, analog_file), "rb") as f:
                self.analog_data[analog_name] = np.fromfile(f, dtype="<i").reshape(-1, 2)

        return self.analog_data
