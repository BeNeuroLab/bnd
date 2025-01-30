"""
Kilosort utils during nwb conversion
"""

# TODO: Complete

from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import probeinterface as pi
from neuroconv.datainterfaces import KiloSortSortingInterface
from neuroconv.tools.spikeinterface import add_sorting_to_nwbfile
from neuroconv.utils import DeepDict
from pydantic import DirectoryPath
from pynwb import NWBFile

from bnd import set_logging

logger = set_logging(__name__)


def _try_loading_trajectory_file(raw_recording_path: Path) -> dict | None:
    """
    Load trajectory of probes from pinpoint output files. Return None if no *trajectory.txt
    file is available.

    Parameters
    ----------
    raw_recording_path :
        path to raw session

    Returns
    -------
    Dictionary with trajectory information per probe if the trajectory file is found.
    None if the trajectory file not found.

    Raises
    ------
    FileExistsError
        If more than one trajectory files are found.

    Warns
    -----
    If no trajectory file is found.
    If a loaded probe is not called "imec0" or "imec1".
    """

    pinpoint_trajectory_file = list(raw_recording_path.glob("*trajectory.txt"))
    if not pinpoint_trajectory_file:
        logger.warning(
            "No trajectory file from pinpoint found. If no trajectory file is present"
            " channel map information cannot be loaded"
        )
        return

    elif len(pinpoint_trajectory_file) > 1:
        raise FileExistsError("Too many files found; expected only one.")

    with open(pinpoint_trajectory_file[0], "r") as f:
        trajectory_str = [l.strip() for l in f.readlines() if l.strip() != ""]
        probe_trajectory_pairs = list(zip(trajectory_str[::2], trajectory_str[1::2]))

        # This should be either imec0 or imec1 to match spikeGLX
        probe_names = [
            probe_trajectory_pair[0] for probe_trajectory_pair in probe_trajectory_pairs
        ]

        for idx, probe_name in enumerate(probe_names):
            if probe_name != f"imec{idx}":
                logger.warning(
                    "Pinpoint probes need to named either 'imec0' or 'imec1' to match spikeglx. "
                    "Skipping probe information loading"
                )
                return

        trajectory_dict = {
            probe: trajectory for probe, trajectory in probe_trajectory_pairs
        }

    return trajectory_dict


def _load_channel_map_information_from_pinpoint_probe(
    channel_map_file_path: Path, pinpoint_probe_name: str
) -> pd.DataFrame:
    """
    Extract the brain area of each electrode on a given probe from a
    channel map file saved by Pinpoint.

    Parameters
    ----------
    channel_map_file_path :
        Pinpoint channel_map file to open
    pinpoint_probe_name :
        Key generated in pinpoint to identify the probe

    Returns
    -------
    Dataframe containing brain area of each electrode

    Raises
    ------
    ValueError
        If the given probe name is not found in the channel map.
    """

    with open(channel_map_file_path, "r") as file:
        channel_map_str = file.read().strip()

    channel_map_list = channel_map_str.strip("[]").split('","')
    pinpoint_probe_names_in_channel_map = []
    pinpoint_channel_maps = []
    for probe_map in channel_map_list:
        probe_map = probe_map.replace('"', "")
        pinpoint_probe_names_in_channel_map.append(probe_map.split(":", 1)[0])
        pinpoint_channel_maps.append(probe_map.split(":", 1)[1])

    if pinpoint_probe_name not in pinpoint_probe_names_in_channel_map:
        raise ValueError(
            f"Probes name {pinpoint_probe_name} defined in *trajectory.txt file does not"
            f" match probe_names in *channel_map.txt file {pinpoint_probe_names_in_channel_map}"
        )

    # Split the string by ";" to separate each entry
    index = pinpoint_probe_names_in_channel_map.index(pinpoint_probe_name)
    pinpoint_channel_map = pinpoint_channel_maps[index].split(";")
    data = [
        dict(zip(["id", "area_number", "area_name", "area_color"], entry.split(",")))
        for entry in pinpoint_channel_map
    ]

    return pd.DataFrame(data)


def _create_channel_map(
    pinpoint_trajectory_dict: dict, raw_recording_path: Path
) -> dict | None:
    """
    Function to create a dictionary (channel map) where each probe (keys) has a pd.DataFrame
    (value) relating electrode to brain area. Returns None if no *channel_map.txt file is available.

    Parameters
    ----------
    pinpoint_trajectory_dict :
        Dictionary of pinpoint generated trajectory out (*trajectory.txt) for each probe
    raw_recording_path :
        Path to raw session

    Returns
    -------
    Dictionary of channel map for each probe
        If the channel map file is found.
    None
        If the channel map file is not found or too many channel map files are found.
        If there is a problem loading channel map data.
    """

    pinpoint_channel_map_file = list(raw_recording_path.glob("*channel_map.txt"))
    if not pinpoint_channel_map_file:
        logger.warning("No channel map file from pinpoint found")
        return

    elif len(pinpoint_channel_map_file) > 1:
        logger.warning("Too many channel map files from pinpoint found")
        return

    channel_map = {}
    # Use pinpoint generate probe identifier to match probes between trajectories and channel_maps
    for probe in pinpoint_trajectory_dict.keys():
        try:
            pinpoint_probe_name = pinpoint_trajectory_dict[probe].split(":")[0]
            channel_map[probe] = _load_channel_map_information_from_pinpoint_probe(
                channel_map_file_path=pinpoint_channel_map_file[0],
                pinpoint_probe_name=pinpoint_probe_name,
            )
        except Exception as e:
            logger.warning(f"Problem loading channel map data: {str(e)}")
            return

    return channel_map


class MultiProbeKiloSortInterface(KiloSortSortingInterface):
    """
    Class for handling multi-probe kilosorted outputs
    """

    def __init__(
        self,
        folder_path: DirectoryPath,
        keep_good_only: bool = False,
        verbose: bool = False,
    ):
        """Multiprobe kilosort interface

        Parameters
        ----------
        folder_path : Path
            path to session
        keep_good_only : bool, optional
            keep or not only units labelled as `good`, by default False
        verbose : bool, optional
            verbosity level, by default False
        """
        self.kilosort_folder_paths = list(Path(folder_path).glob("**/sorter_output"))
        self.probe_names = [
            ks_path.parent.name.split("_")[-1] for ks_path in self.kilosort_folder_paths
        ]

        self.kilosort_interfaces = [
            KiloSortSortingInterface(folder_path, keep_good_only, verbose)
            for folder_path in self.kilosort_folder_paths
        ]

        self.folder_path = Path(folder_path)

    def set_aligned_starting_time(self, aligned_starting_time: float):
        for kilosort_interface in self.kilosort_interfaces:
            kilosort_interface.set_aligned_starting_time(aligned_starting_time)

    def add_probe_information_to_nwb(self, nwbfile: NWBFile) -> None:
        """
        Add probe information stored in SpikeGLX and pinpoint to the nwbfile if available

        Attempts to add the *trajectory.txt file which contains general probe information such as entry
        & tip position, angles (yaw, pitch roll), and pinpoint probe identifier (autogenerated by
        pinpoint). It also attempts to add the information about electrode location stored
        in *channel_map.txt if available. This maps each electrode to a brain region. Returns None
        as it adds information directly to the nwbfile

        Parameters
        ----------
        nwbfile :
            NWBFile handle

        Returns
        -------
        None
        """

        raw_recording_path = Path(str(self.folder_path).replace("processed", "raw"))
        meta_filepaths = list(raw_recording_path.rglob("*/*ap.meta"))

        # Try loading trajectory information from pinpoint
        pinpoint_trajectory_dict = _try_loading_trajectory_file(raw_recording_path)

        # If pinpoint_trajectories is available, load channel map
        channel_map_dict = (
            _create_channel_map(pinpoint_trajectory_dict, raw_recording_path)
            if pinpoint_trajectory_dict is not None
            else None
        )

        for probe_name in self.probe_names:
            # Get meta_file_path for probe_name
            meta_filepath = next(
                (path for path in meta_filepaths if probe_name in str(path)), None
            )

            # Load probe object
            probe = pi.read_spikeglx(meta_filepath)

            if probe.get_shank_count() == 1:  # Set shank ids
                probe.set_shank_ids(np.full((probe.get_contact_count(),), 1))
            else:
                raise NotImplementedError("Multishank probes not yet implemented")

            nwbfile.create_device(
                name=probe_name,
                description=probe.annotations["model_name"],  # Neuropixels 1.0
                manufacturer=probe.annotations["manufacturer"],
            )
            nwbfile.create_electrode_group(
                name=probe_name,
                description=f"{probe.annotations['model_name']}. Location is the output from "
                f"pinpoint and corresponds to the targeted brain area",
                location=(
                    pinpoint_trajectory_dict[probe_name]
                    if pinpoint_trajectory_dict
                    else "No pinpoint trajectory"
                ),
                device=nwbfile.devices[probe_name],
            )

            for contact_position, contact_id in zip(
                probe.contact_positions, probe.contact_ids
            ):
                x, y = contact_position
                z = 0.0
                contact_id = int(contact_id.split("e")[1:][0])
                if channel_map_dict is not None:
                    contact_location = channel_map_dict[probe_name].area_name[contact_id]
                else:
                    contact_location = "nan"
                nwbfile.add_electrode(
                    group=nwbfile.electrode_groups[probe_name],
                    x=float(x),
                    y=float(y),
                    z=z,
                    id=contact_id,
                    location=contact_location,
                    reference="Local probe reference: Top of the probe",
                    enforce_unique_id=False,
                )

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: Optional[DeepDict] = None,
    ):
        self.add_probe_information_to_nwb(nwbfile)

        # Kilosort output will be saved in processing and not units
        # units is reserved for the units curated by Phy
        for probe_name, kilosort_interface, kilosort_folder_path in zip(
            self.probe_names, self.kilosort_interfaces, self.kilosort_folder_paths
        ):
            # Load templates
            templates = np.load(kilosort_folder_path / "templates.npy")

            # Copied function from nwb to add templates to nwb
            self.add_one_probe_to_nwbfile(
                sorting_interface=kilosort_interface,
                nwbfile=nwbfile,
                metadata=metadata,
                write_as="processing",
                units_name=f"units_{probe_name}",
                units_description=f"Kilosorted units on {probe_name}",
                waveform_means=templates,
            )

    def get_metadata(self) -> DeepDict:
        return DeepDict()

    @staticmethod
    def add_one_probe_to_nwbfile(
        sorting_interface: KiloSortSortingInterface,
        nwbfile: NWBFile,
        metadata: Optional[DeepDict] = None,
        stub_test: bool = False,
        write_ecephys_metadata: bool = False,
        write_as: Literal["units", "processing"] = "units",
        units_name: str = "units",
        units_description: str = "Autogenerated by neuroconv.",
        waveform_means: Optional[np.ndarray] = None,
    ):
        """
        Primary function for converting the data in a SortingExtractor to NWB format.

        This function is copied from neuroconv and modified to add waveform means (templates)
        to the NWB file.

        Original definition of neuroconv's BaseSortingExtractorInterface.add_to_nwbfile:
        https://github.com/catalystneuro/neuroconv/blob/96c8ed4d76bd734e335acd999c015770f9cfd92a/src/neuroconv/datainterfaces/ecephys/basesortingextractorinterface.py#L282


        Parameters
        ----------
        sorting_interface : KiloSortSortingInterface
            KiloSortSortingInterface for one probe.
        nwbfile :
            Fill the relevant fields within the NWBFile object.
        metadata :
            Information for constructing the NWB file (optional) and units table descriptions.
            Should be of the format::

                metadata["Ecephys"]["UnitProperties"] = dict(name=my_name, description=my_description)
        stub_test :
            If True, will truncate the data to run the conversion faster and take up less memory.
        write_ecephys_metadata :
            Write electrode information contained in the metadata.
        write_as :
            How to save the units table in the nwb file. Options:
            - 'units' will save it to the official NWBFile.Units position; recommended only for the final form of the data.
            - 'processing' will save it to the processing module to serve as a historical provenance for the official table.
        units_name :
            The name of the units table. If write_as=='units', then units_name must also be 'units'.
        units_description :
            Description of where the units in this sorting come from.
        waveform_means :
            Mean waveform (=template) of each unit as recorded from each channel.
            Array of shape (n_units, n_samples, n_channels)
        """

        metadata_copy = deepcopy(metadata)
        if write_ecephys_metadata:
            sorting_interface.add_channel_metadata_to_nwb(
                nwbfile=nwbfile, metadata=metadata_copy
            )

        if stub_test:
            sorting_extractor = sorting_interface.subset_sorting()
        else:
            sorting_extractor = sorting_interface.sorting_extractor

        property_descriptions = dict()
        for metadata_column in metadata_copy["Ecephys"].get("UnitProperties", []):
            property_descriptions.update(
                {metadata_column["name"]: metadata_column["description"]}
            )
            for unit_id in sorting_extractor.get_unit_ids():
                # Special condition for wrapping electrode group pointers to actual object ids rather than string names
                if (
                    metadata_column["name"] == "electrode_group"
                    and nwbfile.electrode_groups
                ):
                    value = nwbfile.electrode_groups[
                        sorting_interface.sorting_extractor.get_unit_property(
                            unit_id=unit_id, property_name="electrode_group"
                        )
                    ]
                    sorting_extractor.set_unit_property(
                        unit_id=unit_id,
                        property_name=metadata_column["name"],
                        value=value,
                    )

        add_sorting_to_nwbfile(
            sorting_extractor,
            nwbfile=nwbfile,
            property_descriptions=property_descriptions,
            write_as=write_as,
            units_name=units_name,
            units_description=units_description,
            waveform_means=waveform_means,
        )
