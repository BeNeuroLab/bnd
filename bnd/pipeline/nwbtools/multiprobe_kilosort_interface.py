"""
Kilosort utils during nwb conversion
"""

# TODO: Complete




from pathlib import Path


class MultiProbeKiloSortInterface:
        
    def __init__(
        self,
        folder_path: Path,
        keep_good_only: bool = False,
        verbose: bool = True,
    ):
        self.kilosort_folder_paths = list(Path(folder_path).glob("**/sorter_output"))
        self.probe_names = [
            ks_path.parent.name.split("_")[-1] for ks_path in self.kilosort_folder_paths
        ]

        self.kilosort_interfaces = [
            KiloSortSortingInterface(folder_path, keep_good_only, verbose)
            for folder_path in self.kilosort_folder_paths
        ]

        self.folder_path = Path(folder_path)
