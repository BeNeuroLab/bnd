from pathlib import Path


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
        self.animal_name_pattern = "M???"

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

    @staticmethod
    def get_animal_name(session_name) -> str:
        return session_name[:4]

    def get_local_session_path(self, session_name: str):
        animal = self.get_animal_name(session_name)
        local_session_path = self.LOCAL_PATH / "raw" / animal / session_name
        return local_session_path

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

