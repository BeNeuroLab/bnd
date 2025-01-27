from pathlib import Path

import typer
from rich import print
from typing_extensions import Annotated

from bnd.config import (
    _check_is_git_track,
    _check_root,
    _check_session_directory,
    _get_env_path,
    _get_package_path,
    _load_config,
)
from bnd.data_transfer import download_session, upload_session
from bnd.pipeline import _check_processing_dependencies
from bnd.update_bnd import check_for_updates, update_bnd

# Create a Typer app
app = typer.Typer(
    add_completion=False,  # Disable the auto-completion options
)


# ============================== Pipeline functions =======================================


@app.command()
def to_pyal(
    session_name: Annotated[
        str,
        typer.Argument(help="Session name to convert"),
    ],
    kilosort: Annotated[
        bool,
        typer.Option(
            "--kilosort/--dont-kilosort",
            "-k/-K",
            help="Run kilosort if available (-k) or dont (-K).",
        ),
    ] = True,
    mapping: Annotated[
        str,
        typer.Option(
            "-m",
            "--mapping",
            help="Specify the channel mapping method: 'default' or 'custom.",
            case_sensitive=False,
        ),
    ] = None,
) -> None:
    """
    Convert session data into a pyaldata dataframe and saves it as a .mat

    \b
    If no .nwb file is present it will automatically generate one and if a nwb file is present it will skip it. If you want to generate a new one run `bnd to-nwb`

    \b
    If no kilosorted data is available it will not kilosort by default. If you want to kilosort add the flag `-k`

    \b
    Basic usage:
        `bnd to-pyal M037_2024_01_01_10_00  # Kilosorts data and converts to pyaldata
        `bnd to-pyal M037_2024_01_01_10_00 -m default  # Uses pinpoint mapping output
        `bnd to-pyal M037_2024_01_01_10_00 -m default   # Uses pinpoint mapping output
    """
    _check_processing_dependencies()
    from bnd.pipeline.pyaldata import run_pyaldata_conversion

    # Load config and get session path
    config = _load_config()
    session_path = config.get_local_session_path(session_name)

    # Check session directory
    _check_session_directory(session_path)

    # Run pipeline
    run_pyaldata_conversion(session_path, kilosort, map)

    return


@app.command()
def to_nwb(
    session_name: str,
    kilosort_flag: Annotated[
        bool,
        typer.Option(
            "--kilosort/--dont-kilosort",
            "-k/-K",
            help="Run kilosort if not available (-k) or dont (-K).",
        ),
    ] = False,
    mapping: Annotated[
        str,
        typer.Option(
            "-m",
            "--map",
            help="Specify the channel mapping method: 'default' or 'custom.",
            case_sensitive=False,
        ),
    ] = None,
) -> None:
    """
    Convert session data into a nwb file and saves it as a .nwb

    \b
    If no kilosorted data is available it will not kilosort by default. If you want to kilosort add the flag `-k`

    \b
    Basic usage:
        `bnd to-nwb M037_2024_01_01_10_00`
    """
    # TODO: Add channel map argument: no-map, default-map, custom-map
    _check_processing_dependencies()
    from bnd.pipeline.nwb import run_nwb_conversion

    config = _load_config()
    session_path = config.get_local_session_path(session_name)

    # Check session directory
    _check_session_directory(session_path)

    # Run pipeline
    run_nwb_conversion(session_path, kilosort_flag, mapping)
    return


@app.command()
def ksort(
    session_name: Annotated[
        str,
        typer.Argument(help="Session name to kilosort"),
    ],
) -> None:
    """
    Kilosorts data from a single session.

    \b
    Basic usage:
        `bnd ksort M037_2024_01_01_10_00`
    """
    # this will throw an error if the dependencies are not available
    _check_processing_dependencies()
    from bnd.pipeline.kilosort import run_kilosort_on_session

    config = _load_config()
    session_path = config.get_local_session_path(session_name)

    # Check session directory
    _check_session_directory(session_path)

    # Run pipeline
    run_kilosort_on_session(session_path)
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
    upload_session()
    return


@app.command()
def down():
    """
    Download experimental data from the remote server of a single session.

    \b
    Example usage to download data of a given session:
        `bnd dl M017_2024_03_12_18_45 -e`  # Uploads ephys
    """
    download_session()
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
