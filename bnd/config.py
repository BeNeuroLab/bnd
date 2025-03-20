import re
from datetime import datetime
from pathlib import Path

from bnd import set_logging

logger = set_logging(__name__)


def _get_package_path() -> Path:
    """
    Returns the path to the package directory.
    """
    return Path(__file__).absolute().parent.parent


def _get_env_path() -> Path:
    """
    Returns the path to the .env file containing the configuration settings.
    """
    package_path = _get_package_path()
    return package_path / ".env"


def _check_session_directory(session_path):
    return


def _check_is_git_track(repo_path):
    folder = Path(repo_path)  # Convert to Path object
    assert (folder / ".git").is_dir()


def _check_root(root_path: Path):
    assert root_path.exists(), f"{root_path} does not exist."
    assert root_path.is_dir(), f"{root_path} is not a directory."

    files_in_root = [f.stem for f in root_path.iterdir()]

    assert "raw" in files_in_root, f"No raw folder in {root_path}"


class Config:
    """
    Class to load local configuration
    """

    def __init__(self, env_path=_get_env_path()):
        self.REMOTE_PATH = None
        self.LOCAL_PATH = None
        self.REPO_PATH = None
        # Load the actual environment PATHs
        self.load_env(env_path)
        self.datetime_pattern = "%Y_%m_%d_%H_%M"
        self.session_name_re = r"^M\d{3}_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})"
        self.video_formats = (".avi", ".mp4", ".AVI", ".MP4")

    def load_env(self, env_path: Path):
        with open(env_path, "r") as file:
            for line in file:
                # Ignore comments and empty lines
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse key-value pairs
                key, value = map(str.strip, line.split("=", 1))

                # Set as environment variable
                setattr(self, key, Path(value))

    def get_local_animal_path(self, animal_name_like: str) -> Path:
        return self.LOCAL_PATH / "raw" / self.get_animal_name(animal_name_like)

    def get_remote_animal_path(self, animal_name_like: str) -> Path:
        return self.REMOTE_PATH / "raw" / self.get_animal_name(animal_name_like)

    def get_remote_session_path(self, session_name: str) -> Path:
        animal = self.get_animal_name(session_name)
        remote_session_path = self.REMOTE_PATH / "raw" / animal / session_name
        return remote_session_path

    def get_local_session_path(self, session_name: str) -> Path:
        animal = self.get_animal_name(session_name)
        local_session_path = self.LOCAL_PATH / "raw" / animal / session_name
        return local_session_path

    def convert_to_local(self, remote_path: Path) -> Path:
        "convert a remote path to a local path"
        assert str(remote_path).startswith(
            str(self.REMOTE_PATH)
        ), "Path is not in the remote directory"
        return self.LOCAL_PATH / remote_path.relative_to(self.REMOTE_PATH)

    def convert_to_remote(self, local_path: Path) -> Path:
        "convert a local path to a remote path"
        assert str(local_path).startswith(
            str(self.LOCAL_PATH)
        ), "Path is not in the local directory"
        return self.REMOTE_PATH / local_path.relative_to(self.LOCAL_PATH)

    def file_name_ok(self, file_name: str) -> bool:
        """
        Check if the file name follows the session name convention
        And the date-time is valid
        Returns True if it is, False otherwise.
        """
        match = re.match(self.session_name_re, file_name)

        if not match:
            logger.debug("file name not matching the pattern")
            return False  # Doesn't match the expected pattern

        # Extract the datetime parts
        year, month, day, hour, minute = map(int, match.groups())

        # Validate if it forms a correct date-time
        try:
            dt = datetime(year, month, day, hour, minute)
        except ValueError:
            logger.debug("Invalid datetime")
            return False

        # Do not allow future dates
        if dt.date() <= datetime.today().date():
            return True
        else:
            logger.debug("file has future date")
            return False

    @staticmethod
    def get_animal_name(session_name) -> str:
        return session_name[:4]

    @staticmethod
    def get_subdirectories_from_pattern(directory: Path, subdir_pattern: str):
        subdirectory_paths = list(directory.glob(subdir_pattern))

        return subdirectory_paths


def _load_config() -> Config:
    """
    Loads the configuration settings from the .env file and returns it as a Config object.
    """
    if not _get_env_path().exists():
        raise FileNotFoundError("Config file not found. Run `bnd init` to create one.")

    return Config()


def find_file(
    main_path: str | Path, extension: tuple[str | Path] = (".txt",)
) -> list[Path]:
    """
    This function finds all the file types specified by 'extension' in the 'main_path' directory
    and all its subdirectories and their sub-subdirectories etc.,
    and returns a list of all file paths
    'extension' is a list of desired file extensions: ['.dat','.prm']
    """
    if isinstance(main_path, str):
        path = Path(main_path)
    else:
        path = main_path

    if isinstance(extension, str):
        extension = extension.split()  # turning extension into a list with a single element

    return [
        Path(walking[0] / goodfile)
        for walking in path.walk()
        for goodfile in walking[2]
        for ext in extension
        if goodfile.endswith(ext)
    ]


def list_dirs(main_path: str | Path) -> list[str]:
    "List the names (strings) of directories under a path"
    # Create a Path object from the provided path string
    if isinstance(main_path, str):
        p = Path(main_path)
    else:
        p = main_path
    # List all directories in the given path
    directories = [d.name for d in p.iterdir() if d.is_dir()]
    return directories


def list_session_datetime(animal_path: str | Path) -> tuple[list[datetime.date], list[str]]:
    """
    List and sort the datetimes of the sessions in a given path
    animal_path: path to the animal directory containing the session directories: /data/raw/M034/
    Return: - list of datetime objects sorted in ascending order,
            - list of session names in the format M034_2024_07_12_10_00
    """
    datetime_format = _load_config().datetime_pattern
    # List all directories in the given path
    session_list = list_dirs(Path(animal_path))
    session_datetime_list = [
        datetime.strptime(s[5:], datetime_format) for s in session_list
    ]
    session_datetime_list.sort()
    sort_session_list = [
        f"{Path(animal_path).name}_{s.strftime(datetime_format)}"
        for s in session_datetime_list
    ]

    return session_datetime_list, sort_session_list


def get_last_session(animal_path: str | Path) -> str:
    """
    Get the name of the last session for a given animal: M034_2024_07_12_10_00
    animal_path: path to the directory containing the session directories : /data/raw/M034/
    """
    last_session = list_session_datetime(Path(animal_path))[1][-1]
    assert (
        Path(animal_path) / last_session
    ).is_dir(), f"Session {last_session} not found in {animal_path}"

    return last_session
