"""
Module for conversion from nwb to pyaldata format
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from ndx_pose import PoseEstimationSeries
from pynwb import NWBHDF5IO
from pynwb.behavior import SpatialSeries
from pynwb.misc import Units

from ..logger import set_logging
from ..config import _load_config
from ..pipeline.nwb import run_nwb_conversion

logger = set_logging(__name__)


def _count_bytes(obj):
    """
    Recurrent function to call bytes of any nested variable

    Parameters
    ----------
    obj : _type_
        Variable on which to count bytes

    Returns
    -------
    int
        Total number of bytes
    """
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, np.recarray):
        return sum(_count_bytes(obj[field]) for field in obj.dtype.names)
    elif isinstance(obj, np.void):
        return sum(_count_bytes(obj[field]) for field in obj.dtype.names)
    elif isinstance(obj, pd.DataFrame):  # Pandas DataFrame
        return obj.memory_usage(deep=True).sum()
    elif isinstance(obj, pd.Series):
        return obj.memory_usage(deep=True)
    elif isinstance(obj, dict):
        return sum(_count_bytes(v) for v in obj.values())
    elif isinstance(obj, (list, tuple)):
        return sum(_count_bytes(v) for v in obj)
    return 0


def _get_nbytes_from_recarray(recarray: np.recarray):
    """
    Cumulative sum in a specific type of numpy object - np.recarray

    Parameters
    ----------
    recarray : np.recarray
        Variable on which to count bytes

    Returns
    -------
    int
        Total number of bytes
    """
    nbytes = 0
    for record in recarray:
        nbytes = nbytes + _count_bytes(record)

    return nbytes


def _bin_spikes(probe_units: Units, bin_size: float) -> np.array:
    """
    Bin spikes from pynwb from one probe

    Parameters
    ----------
    probe_units :
        PyNWB Units object from which to read spike times
    bin_size :
        Bin size in seconds to use

    Returns
    -------
    Array :
        Matrix of neuron x time with binned spikes

    """

    start_time = 0  # This is hardcoded since its aligned in nwb conversion
    end_time = np.max(probe_units.spike_times[:])
    number_of_bins = int(np.ceil((end_time - start_time) / bin_size))

    # Initialize the binned spikes array
    binned_spikes = np.zeros((len(probe_units.id[:]), number_of_bins), dtype=int)

    # Populate the binned spikes array
    for neuron_id in probe_units.id:
        spike_times = probe_units.get_unit_spike_times(neuron_id)

        # Taks spikes from after start time
        spike_times = spike_times[spike_times >= start_time]

        # Bin the spikes with flooring round
        bin_indices = (spike_times / bin_size).astype(int)

        # Add them to the corresponding bin.
        np.add.at(binned_spikes, (neuron_id, bin_indices), 1)

    return binned_spikes


def _transform_chan_best_to_unit_guide(arr: np.array) -> np.array:
    # Find unique values and their counts
    unique_values, counts = np.unique(arr, return_counts=True)

    # Create the result list
    result = []

    # Iterate through unique values and counts to form the Nx2 matrix
    for val, count in zip(unique_values, counts):
        for i in range(1, count + 1):
            result.append([val, i])

    # Convert to numpy array
    result_array = np.array(result)
    return result_array


def _parse_pynwb_probe(
    probe_units: Units, electrode_info: pd.DataFrame, bin_size: float
) -> dict:
    """
    Parse nwb Units object to bin spikes, reorder units and extract kilosort labels

    Parameters
    ----------
    probe_units :
        Nwb units object containing units table.
    electrode_info :
        Electrode dataframe containing brain location of each electrode
    bin_size :
        Bin size in seconds to use when binning the data

    Returns
    -------

    """
    # This returns a neurons x bin array of 0s and 1s
    binned_spikes = _bin_spikes(probe_units, bin_size)

    # This returns a [nTemplates, nTimePoints, nTempChannels] matrix
    templates = probe_units.waveform_mean[:]

    # NOTE: We do not need the templates_ind.npy since in the case of Kilosort templates_ind
    # is just the integers from 0 to nChannels-1, as templates are defined on all channels.

    # TODO: Load channel_map.npy to .nwb to have more thorough mapping between templates and
    #   channels. I am not currently loading the channel_map.npy file since for Neuropixels 1.0
    #   it is simply an array of 0 to 383

    # Get max amplitude channel based on templates
    chan_best = (templates**2).sum(axis=1).argmax(axis=-1)

    # Get brain area channel map for this specific probe
    electrode_info_df = electrode_info.to_dataframe()
    probe_electrode_locations_df = electrode_info_df[
        electrode_info_df["group_name"] == probe_units.name.split("_")[-1]
    ]
    probe_channel_map = probe_electrode_locations_df["location"].to_dict()

    no_pinpoint_channel_map = all(value == "nan" for value in probe_channel_map.values())

    brain_area_spikes_and_chan_best = {}

    if no_pinpoint_channel_map:
        brain_areas = {f"all_{probe_units.name.split('_')[-1]}"}
    else:
        brain_areas = {
            value for value in probe_channel_map.values() if value not in ["out", "void"]
        }

    for brain_area in brain_areas:
        if no_pinpoint_channel_map:  # Take all channels if there is no channel map
            brain_area_channels = [key for key, value in probe_channel_map.items()]
        else:
            brain_area_channels = [
                key for key, value in probe_channel_map.items() if value == brain_area
            ]

        brain_area_neurons = np.where(np.isin(chan_best, brain_area_channels))[0]

        # Define chan best
        unsorted_chan_best = chan_best[brain_area_neurons]
        sorted_chan_best_indices = np.argsort(
            unsorted_chan_best
        )  # Variable with sorted indices
        sorted_chan_best = unsorted_chan_best[sorted_chan_best_indices]

        # Define unit guide
        unit_guide = _transform_chan_best_to_unit_guide(sorted_chan_best)

        # Take neurons that are brain area specific and them sort them according to unit guide
        brain_area_spikes_and_chan_best[brain_area.replace("-", "_")] = {
            "spikes": binned_spikes[brain_area_neurons, :][sorted_chan_best_indices, :]
        }
        brain_area_spikes_and_chan_best[brain_area.replace("-", "_")][
            "chan_best"
        ] = sorted_chan_best
        brain_area_spikes_and_chan_best[brain_area.replace("-", "_")][
            "unit_guide"
        ] = unit_guide
        brain_area_spikes_and_chan_best[brain_area.replace("-", "_")]["KSLabel"] = (
            probe_units.KSLabel[brain_area_neurons][sorted_chan_best_indices]
        )

    return brain_area_spikes_and_chan_best


def _parse_pose_estimation_series(
    pose_est_series: PoseEstimationSeries,
) -> pd.DataFrame:
    """
    Parse pose estimation series data from anipose output

    Parameters
    ----------
    pose_est_series :
        ndx_pose object to parse from nwb file

    Returns
    -------
    pd.DataFrame :
        Contains x, y, z or angle and timestamps

    """

    if pose_est_series.data[:].shape[1] == 3:
        colnames = ["x", "y", "z"]
    elif pose_est_series.data[:].shape[1] == 2 and all(pose_est_series.data[:, 1] == 0):
        # If this is true we assume we are dealing with angle data
        colnames = ["angle"]
    else:
        raise ValueError(
            f"Shape {pose_est_series.data[:].shape} is not supported by pynwb."
            f" Please provide a valid PoseEstimationSeries object"
        )

    df = pd.DataFrame()
    for i, col in enumerate(colnames):
        df[col] = pose_est_series.data[:, i]

    timestamps = np.arange(pose_est_series.data[:].shape[0])
    timestamps = timestamps / pose_est_series.rate + pose_est_series.starting_time
    df["timestamps"] = timestamps

    return df


def _parse_spatial_series(spatial_series: SpatialSeries) -> pd.DataFrame:
    """
    Parse data and timestamps of a SpatialSeries .pynwb object

    Parameters
    ----------
    spatial_series :
        pynwb object to parse

    Returns
    -------
    pd.DataFrame :
        Contains x, y, z and timestamp
    """

    if spatial_series.data[:].shape[1] == 2:
        colnames = ["x", "y"]
    elif spatial_series.data[:].shape[1] == 3:
        colnames = ["x", "y", "z"]
    else:
        raise ValueError(
            f"Shape {spatial_series.data[:].shape} is not supported by pynwb. "
            f"Please provide a valid SpatialSeries object"
        )

    df = pd.DataFrame()
    for i, col in enumerate(colnames):
        df[col] = spatial_series.data[:, i]

    df["timestamps"] = spatial_series.timestamps[:]

    return df


def _add_data_to_trial(
    df_to_add_to: pd.DataFrame,
    new_data_column: str,
    df_to_add_from: pd.DataFrame,
    columns_to_read_from: str | list,
    timestamp_column=None,
) -> pd.DataFrame:
    """
    Data-type agnostic function to read data from ParsedNWBfile and add it to pyaldata dataframe

    Parameters
    ----------
    df_to_add_to :
        pd.Dataframe to add new data to. Usually self.pyaldata_df
    new_data_column :
        Name of the new data column
    df_to_add_from :
        pd.Dataframe with raw data from nwb file
    columns_to_read_from :
        Column names from which to extract data
    timestamp_column :
        Timestamp column depending on if you are adding timestamps into pyaldata or not

    Returns
    -------

    """
    for index, row in df_to_add_to.iterrows():
        trial_specific_events = df_to_add_from[
            (df_to_add_from["timestamp_idx"] >= row["idx_trial_start"])
            & (df_to_add_from["timestamp_idx"] <= row["idx_trial_end"])
        ]

        # Add to pyaldata dataframe
        df_to_add_to[new_data_column] = df_to_add_to[new_data_column].astype("object")
        df_to_add_to.at[index, new_data_column] = trial_specific_events[
            columns_to_read_from
        ].to_numpy()

        if timestamp_column is not None:
            df_to_add_to[timestamp_column] = df_to_add_to[timestamp_column].astype("object")
            df_to_add_to.at[index, timestamp_column] = (
                trial_specific_events["timestamp_idx"].to_numpy() - row["idx_trial_start"]
            )

    return df_to_add_to


class ParsedNWBFile:
    def __init__(self, nwbfile_path: Path):
        with NWBHDF5IO(nwbfile_path, mode="r") as io:
            # General
            self.bin_size = 0.01
            self.subject_id = nwbfile_path.name[:4]

            # Nwb file
            self.nwbfile_path = nwbfile_path
            self.nwbfile = io.read()

            # Processing modules
            self.try_to_parse_processing_module("behavior")
            self.try_to_parse_processing_module("ecephys")

            # Initialize Pyaldata dataframe
            self.pyaldata_df = None

        logger.info("Parsed NWB file")

    def try_to_parse_processing_module(self, processing_key: str) -> None:
        """
        Go through nwb processing module and parse 'behavior' and 'ecephys' if they are available

        Parameters
        ----------
        processing_key :
            Module to process. Can be either `behavior` or `ecephys`

        Returns
        -------

        """
        if hasattr(self.nwbfile, "processing"):
            if processing_key in self.nwbfile.processing.keys():
                setattr(
                    self,
                    processing_key,
                    self.nwbfile.processing[processing_key].data_interfaces,
                )
                if processing_key == "behavior":
                    # Pycontrol states and events. This is assumed to always be there if
                    # there is a behavior processin module
                    self.parse_nwb_pycontrol_states()
                    self.parse_nwb_pycontrol_events()

                    # Also try to add motion sensor data if its available
                    self.try_to_parse_motion_sensors()

                    # Anipose data
                    self.try_parsing_anipose_output()

                elif processing_key == "ecephys":
                    # If there is a ephys processing module we assume there is spiking data
                    self.parse_spike_data()

            else:
                logger.warning(
                    f"NWBFile {self.nwbfile.processing.keys()} does not have {processing_key}"
                )
        else:
            logger.warning(
                f"NWBFile {self.nwbfile_path.name} does not have processing module"
            )

        return

    def parse_nwb_pycontrol_states(self) -> None:
        """
        Parse pycontrol output from behavioural processing module of .nwb file.

        Creates a dictionary containing states and the timestamps

        Returns
        -------

        """

        data_dict = {
            col: self.behavior["behavioral_states"][col].data[:]
            for col in self.behavior["behavioral_states"].colnames
        }
        self.pycontrol_states = pd.DataFrame(data_dict)
        return

    def parse_nwb_pycontrol_events(self) -> None:
        """
        Iterate through pycontrol events and create dataframe with values and timestamps.

        Combines print events and normal events

        Returns
        -------

        """

        behav_events_time_series = self.behavior["behavioral_events"].time_series[
            "behavioral events"
        ]
        print_events_time_series = self.behavior["print_events"].time_series

        # First make dataframe with behav events
        df_behav_events = pd.DataFrame()

        # Behavioural event dont have values but print events do so we need to
        # stay consistent with dimension
        df_behav_events["event"] = behav_events_time_series.data[:]
        df_behav_events["value"] = np.full(
            behav_events_time_series.data[:].shape[0], np.nan
        )
        df_behav_events["timestamp"] = behav_events_time_series.timestamps[:]

        # Then make dataframe with print events, and a df for each print event
        # Sounds a bit convoluted but it is converted to .nwb with a different
        # key for each print event
        df_print_events = pd.DataFrame()
        for print_event in print_events_time_series.keys():
            tmp_df = pd.DataFrame()

            tmp_df["event"] = np.full(
                print_events_time_series[print_event].data[:].shape[0], print_event
            )
            tmp_df["value"] = print_events_time_series[print_event].data[:]
            tmp_df["timestamp"] = print_events_time_series[print_event].timestamps[:]

            df_print_events = pd.concat(
                [df_print_events, tmp_df], axis=0, ignore_index=True
            )

        # Concatenate both dataframes
        df_events = pd.concat([df_behav_events, df_print_events], axis=0, ignore_index=True)
        df_events.sort_values(by="timestamp", ascending=True, inplace=True)
        df_events.reset_index(drop=True, inplace=True)
        self.pycontrol_events = df_events

        return

    def try_to_parse_motion_sensors(self) -> None:
        """
        Add motion data to instance as a x, y array

        Returns
        -------

        """
        if "Position" not in self.behavior.keys():
            logger.warning("No motion data available")
            return

        ball_position_spatial_series = self.behavior["Position"].spatial_series[
            "Ball position"
        ]
        self.pycontrol_motion_sensors = _parse_spatial_series(ball_position_spatial_series)
        return

    def try_parsing_anipose_output(self):
        """
        Add anipose data (xyz) or angle to instance as a dictionary with each keypoint

        Returns
        -------

        """
        if "Pose estimation" not in self.behavior.keys():
            logger.warning("No anipose data available")
            return

        anipose_data_dict = self.behavior["Pose estimation"].pose_estimation_series

        parsed_anipose_data_dict = {}
        for key in anipose_data_dict.keys():
            parsed_anipose_data_dict[key] = _parse_pose_estimation_series(
                anipose_data_dict[key]
            )
        self.anipose_data = parsed_anipose_data_dict

        return

    def parse_spike_data(self):
        """
        Parse spiking data from each probe and assign a dictionary of spikes, chan_best, and
        kilosort labels ('good' or 'mua')

        Returns
        -------

        """

        logger.info(f"Parsing spiking data. Found probes {list(self.ecephys.keys())}")
        spike_data_dict = {}

        for probe_units in self.ecephys.keys():
            spike_data_dict[probe_units] = _parse_pynwb_probe(
                probe_units=self.ecephys[probe_units],
                electrode_info=self.nwbfile.electrodes,
                bin_size=self.bin_size,
            )
        self.spike_data = spike_data_dict
        return

    def add_pycontrol_states_to_df(self):
        # TODO: Fix time units
        start_time = 0.0
        end_time = self.pycontrol_states.stop_time.values[-1] / 1000  # To seconds

        self.pyaldata_df["trial_id"] = self.pycontrol_states.start_time.index
        self.pyaldata_df["bin_size"] = self.bin_size

        # Start and stop times of each state
        self.pyaldata_df["idx_trial_start"] = np.floor(
            self.pycontrol_states.start_time.values[:] / 1000 / self.bin_size
        ).astype(int)
        self.pyaldata_df["idx_trial_end"] = (
            np.floor(
                self.pycontrol_states.stop_time.values[:] / 1000 / self.bin_size
            ).astype(int)
            - 1
        )
        self.pyaldata_df["trial_name"] = self.pycontrol_states.state_name[:]

        self.pyaldata_df["trial_length"] = (
            self.pyaldata_df["idx_trial_end"] - self.pyaldata_df["idx_trial_start"] + 1
        )

        return

    def add_pycontrol_events_to_df(self):
        unique_events = self.pycontrol_events["event"].unique()
        for unique_event in unique_events:
            self.pyaldata_df[f"values_{unique_event}"] = np.nan
            self.pyaldata_df[f"idx_{unique_event}"] = np.nan

        # Add timestamp_idx
        self.pycontrol_events["timestamp_idx"] = np.floor(
            self.pycontrol_events.timestamp.values[:] / 1000 / self.bin_size
        ).astype(int)

        # Iterate over states
        for unique_event in unique_events:
            unique_event_df = self.pycontrol_events[
                self.pycontrol_events["event"] == unique_event
            ]
            self.pyaldata_df = _add_data_to_trial(
                df_to_add_to=self.pyaldata_df,
                new_data_column=f"values_{unique_event}",
                df_to_add_from=unique_event_df,
                columns_to_read_from="value",
                timestamp_column=f"idx_{unique_event}",
            )

        return

    def add_motion_sensor_data_to_df(self):
        if hasattr(self, "pycontrol_motion_sensors"):
            # Bin timestamps
            self.pycontrol_motion_sensors["timestamp_idx"] = np.floor(
                self.pycontrol_motion_sensors.timestamps.values[:] / 1000 / self.bin_size
            ).astype(int)

            # Add columns
            self.pyaldata_df["motion_sensor_xy"] = np.nan

            # Add data
            self.pyaldata_df = _add_data_to_trial(
                df_to_add_to=self.pyaldata_df,
                new_data_column="motion_sensor_xy",
                df_to_add_from=self.pycontrol_motion_sensors,
                columns_to_read_from=["x", "y"],
                timestamp_column=None,
            )
        return

    def add_anipose_data_to_df(self):
        if hasattr(self, "anipose_data"):
            for anipose_key, anipose_value in self.anipose_data.items():
                # Bin timestamps
                # TODO: Predefine time units during nwb conversion
                anipose_value["timestamp_idx"] = np.floor(
                    anipose_value.timestamps.values[:] / self.bin_size
                ).astype(int)

                # Add columns
                self.pyaldata_df[anipose_key] = np.nan

                # Add data
                self.pyaldata_df = _add_data_to_trial(
                    df_to_add_to=self.pyaldata_df,
                    new_data_column=anipose_key,
                    df_to_add_from=anipose_value,
                    columns_to_read_from=(
                        "angle" if "angle" in anipose_key else ["x", "y", "z"]
                    ),
                    timestamp_column=None,
                )

        return

    def add_spiking_data_to_df(self):
        if hasattr(self, "spike_data"):
            for probe_key in self.spike_data.keys():
                for brain_area_key, brain_area_spike_data in self.spike_data[
                    probe_key
                ].items():
                    # Add chan best
                    self.pyaldata_df[f"{brain_area_key}_chan_best"] = [
                        brain_area_spike_data["chan_best"]
                    ] * len(self.pyaldata_df)

                    # Add unit guide
                    self.pyaldata_df[f"{brain_area_key}_unit_guide"] = [
                        brain_area_spike_data["unit_guide"]
                    ] * len(self.pyaldata_df)

                    # Add kilosort labels
                    self.pyaldata_df[f"{brain_area_key}_KSLabel"] = [
                        brain_area_spike_data["KSLabel"]
                    ] * len(self.pyaldata_df)

                    self.pyaldata_df[f"{brain_area_key}_spikes"] = np.nan
                    tmp_df = pd.DataFrame(brain_area_spike_data["spikes"].T)  # Transpose
                    tmp_df["timestamp_idx"] = (
                        tmp_df.index
                    )  # Add timestamp for the following function

                    # Add data
                    self.pyaldata_df = _add_data_to_trial(
                        df_to_add_to=self.pyaldata_df,
                        new_data_column=f"{brain_area_key}_spikes",
                        df_to_add_from=tmp_df,
                        columns_to_read_from=[
                            col for col in tmp_df.columns if col != "timestamp_idx"
                        ],
                        timestamp_column=None,
                    )
        return

    def add_mouse_and_session(self):
        self.pyaldata_df["animal"] = [self.subject_id] * len(self.pyaldata_df)
        self.pyaldata_df["session"] = [self.nwbfile_path.name[:-4]] * len(self.pyaldata_df)
        return

    def purge_nan_columns(self, column_subset="values_") -> None:
        """
        Remove columns that are all nans


        Parameters
        ----------
        column_subset :
            String expression to look for in columns to be purged. Defaults to 'values_'

        Returns
        -------

        """
        columns_to_select = [
            col for col in self.pyaldata_df.columns if col.startswith(column_subset)
        ]

        def _is_empty_array_or_nans(value):
            if isinstance(value, np.ndarray):
                if value.ndim != 0 and all(np.isnan(item) for item in value):
                    return True
                elif value.ndim == 0 and not np.isnan(value.item()):
                    return False
            elif value == np.nan:
                return True
            else:
                return False

        for col_name in columns_to_select:
            if self.pyaldata_df[col_name].apply(_is_empty_array_or_nans).all():
                self.pyaldata_df.drop(col_name, axis=1, inplace=True)

        return

    def expand_dim_in_single_bin_trials(self, column_subset="_spikes") -> None:
        """
        Expand 1D arrays in length one trials

        Parameters
        ----------
        column_subset :
            String expression to look for in columns to be expanded. Defaults to 'spikes_'

        Returns
        -------

        """

        def _expand_dim_in_single_bin_trial(value):
            if isinstance(value, np.ndarray):
                return np.expand_dims(value, axis=1)

        trial_length_1_df = self.pyaldata_df.query("trial_length == 1")
        for column in trial_length_1_df.columns:
            if column_subset in column:
                trial_length_1_df[column].apply(_expand_dim_in_single_bin_trial)

        return

    def run_conversion(self):
        """
        Main routine for pyaldata conversion

        Returns
        -------

        """

        # Define all the necessary columns
        columns = [
            "animal",
            "session",
            "trial_id",
            "trial_name",
            "trial_length",
            "bin_size",
            "idx_trial_start",
            "idx_trial_end",
        ]

        # Initialize dataframe
        self.pyaldata_df = pd.DataFrame(columns=columns)

        # Add behaviour data
        self.add_pycontrol_states_to_df()
        self.add_pycontrol_events_to_df()
        self.add_motion_sensor_data_to_df()
        self.add_anipose_data_to_df()

        # Add ecephys data
        self.add_spiking_data_to_df()

        # Add general information
        self.add_mouse_and_session()

        # Purge nan columns
        self.purge_nan_columns()

        # Expand dimensions
        self.expand_dim_in_single_bin_trials()

        logger.info("Session converted to pyaldata format")

        return

    def _partition_and_save_to_mat(self):
        """
        Checks if data_array needs to be partitioned wrt. max matlab 5 file format and saves it
        """
        data_array = self.pyaldata_df.to_records(index=False)

        # Check if size exceeds matlab 5 format
        nbytes = _get_nbytes_from_recarray(recarray=data_array)
        num_partitions = int(np.ceil(nbytes / (2**31)))
        recarray_size = len(data_array)

        assert num_partitions >= 1 and num_partitions < len(data_array)  # General checks

        # If size doesnt exceed, save the array
        if num_partitions == 1:
            # No need to split the data array
            logger.info("Saving file...")
            path_to_save = (
                self.nwbfile_path.parent / f"{self.nwbfile_path.parent.name}_pyaldata.mat"
            )
            scipy.io.savemat(path_to_save, {"pyaldata": data_array})
            return
        else:
            # Partition array
            logger.info(
                f"Session ({nbytes / 2**30:.2f} GB) exceeds matlab 5 format (2 GB) "
            )

            logger.info(f"Partitioning array into {num_partitions} chunks...")
            partition_sizes = [
                (recarray_size // num_partitions)
                + (1 if i < (recarray_size % num_partitions) else 0)
                for i in range(num_partitions)
            ]

            # Split the recarray accordingly
            indices = np.cumsum([0] + partition_sizes)
            arr_partitions = [
                data_array[indices[i] : indices[i + 1]] for i in range(num_partitions)
            ]

            logger.info("Saving file(s)...")
            for i, arr_partition in enumerate(arr_partitions):
                path_to_save = (
                    self.nwbfile_path.parent
                    / f"{self.nwbfile_path.parent.name}_pyaldata_{i}.mat"
                )
                scipy.io.savemat(path_to_save, {"pyaldata": arr_partition})

            return

    def save(self):
        """
        Main saving routine
        """

        if any(self.nwbfile_path.parent.rglob(f"*.mat")):
            # Prompt the user with an interactive menu
            while True:
                user_input = (
                    input("Mat files in directory. Do you want to overwrite them? (y/n): ")
                    .lower()
                    .strip()
                )
                if "y" in user_input:
                    self._partition_and_save_to_mat()
                    logger.info("Session has been overwritten.")
                    break

                elif "n" in user_input:
                    logger.info("Session was not overwritten.")
                    break

                else:
                    logger.info("Please enter 'y' for yes or 'n' for no.")
        else:
            self._partition_and_save_to_mat()
            logger.info(
                f"Saved pyaldata file(s) in {self.nwbfile_path.parent.name} session"
            )
        return


def run_pyaldata_conversion(
    session_path: Path, kilosort_flag: bool, custom_map: bool
) -> None:
    """
    Main pyaldata conversion routine. Creates pyaldata file for a specific session. It will
    also create nwb file if it doesn't exist and kilosort if indicated

    Parameters
    ----------
    session_path : Path
    kilosort_flag : bool
        Whether to run kilosort or not. Defaults to True

    Returns
    -------

    """
    # Get config
    config = _load_config()

    # Check session_path is Path object
    if isinstance(session_path, str):
        session_path = Path(session_path)

    # Get nwb file
    nwbfile_path = config.get_subdirectories_from_pattern(session_path, "*.nwb")
    if not nwbfile_path:
        logger.warning(f"NWB file: {session_path.name}.nwb not found")
        run_nwb_conversion(session_path, kilosort_flag, custom_map)  # Creates .nwb file
        nwbfile_path = session_path / f"{session_path.name}.nwb"

    elif len(nwbfile_path) > 1:
        raise ValueError("Too many nwb files in session folder")

    else:
        nwbfile_path = nwbfile_path[0]
        assert nwbfile_path.exists()

    # Parse nwb data
    parsed_nwbfile = ParsedNWBFile(nwbfile_path)

    # Convert to pyaldata
    parsed_nwbfile.run_conversion()

    # Save in raw
    parsed_nwbfile.save()

    return
