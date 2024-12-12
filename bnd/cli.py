from pathlib import Path

import typer
from rich import print

from bnd.config import _check_is_git_track, _check_root, _get_env_path, _get_package_path, \
    _load_config
from bnd.update_bnd import check_for_updates, update_bnd

# Create a Typer app
app = typer.Typer(
    add_completion=False,  # Disable the auto-completion options
)

# ============================== Pipeline functions =======================================


def to_pyal():
    return


def to_nwb():
    return


def ksort():
    return


# ================================== Data Transfer ========================================


@app.command()
def up():
    """
    Upload (raw) experimental data to the remote server of a single session.

    \b
    Example usage to upload data of a given session:
        `bnd up M017_2024_03_12_18_45 -e`  # Uploads ephys
    Example usage to upload data of last session of a given animal:
        `bnd up M017 -e`  # Uploads ephys
    """
    return


@app.command()
def down():
    """
    Download experimental data from the remote server of a single session.

    \b
    Example usage to download data of a given session:
        `bnd dl M017_2024_03_12_18_45 -e`  # Uploads ephys
    """
    return


# =================================== Updating ==========================================

@app.command()
def check_updates():
    """
    Check if there are any new commits on the repo's main branch.
    """
    check_for_updates()


@app.command()
def self_update():
    """
    Update the bnd tool by pulling the latest commits from the repo's main branch.
    """
    update_bnd()


# =================================== Config ============================================


@app.command()
def show_config():
    """
    Show the contents of the config file.
    """
    config = _load_config()
    print(f"bnd source code is at {_get_package_path()}", end="\n\n")
    for attr, value in config.__dict__.items():
        print(f"{attr}: {value}")


@app.command()
def check_config():
    """
    Check that the local and remote root folders have the expected raw and processed folders.
    """
    config = _load_config()

    print(
        "Checking that local and remote root folders have the expected raw and processed folders..."
    )

    _check_root(config.LOCAL_PATH)
    _check_root(config.REMOTE_PATH)

    print("[green]Config looks good.")


@app.command()
def init():
    """
    Create a .env file to store the paths to the local and remote data storage.
    """

    # check if the file exists
    env_path = _get_env_path()

    if env_path.exists():
        print("\n[yellow]Config file already exists.\n")

        check_config()

    else:
        print("\nConfig file doesn't exist. Let's create one.")
        repo_path = _get_package_path()
        _check_is_git_track(repo_path)

        local_path = Path(
            typer.prompt("Enter the absolute path to the root of the local data storage")
        )
        _check_root(local_path)

        remote_path = Path(
            typer.prompt("Enter the absolute path to the root of remote data storage")
        )
        _check_root(remote_path)

        with open(env_path, "w") as f:
            f.write(f"REPO_PATH = {repo_path}\n")
            f.write(f"LOCAL_PATH = {local_path}\n")
            f.write(f"REMOTE_PATH = {remote_path}\n")

        # make sure that it works
        check_config()

        print("[green]Config file created successfully.")


# Main Entry Point
if __name__ == "__main__":
    app()
