import os
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


def _check_is_git_track(repo_path):
    folder = Path(repo_path)  # Convert to Path object
    assert (folder / ".git").is_dir()


def _check_root(root_path: Path):
    assert root_path.exists(), f"{root_path} does not exist."
    assert root_path.is_dir(), f"{root_path} is not a directory."

    files_in_root = [f.stem for f in root_path.iterdir()]

    assert "raw" in files_in_root, f"No raw folder in {root_path}"


class Config:
    def __init__(self, env_file='.env'):
        self.load_env(env_file)

    def load_env(self, env_file):
        if not os.path.exists(env_file):
            raise FileNotFoundError(f"{env_file} not found")

        with open(env_file, 'r') as file:
            for line in file:
                # Ignore comments and empty lines
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Parse key-value pairs
                key, value = map(str.strip, line.split('=', 1))

                # Set as environment variable
                os.environ[key] = value

    def get(self, key, default=None):
        return os.getenv(key, default)


def _load_config() -> Config:
    """
    Loads the configuration settings from the .env file and returns it as a Config object.
    """
    if not _get_env_path().exists():
        raise FileNotFoundError("Config file not found. Run `bnd init` to create one.")

    return Config()

