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
    get_last_session,
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
    session_name: str = typer.Argument(..., help="Session name to convert"),
    kilosort_flag: bool = typer.Option(
        True,
        "-k/-K",
        "--kilosort/--dont-kilosort",
        help="Run kilosort if available (-k) or dont (-K).",
    ),
    custom_map: bool = typer.Option(
        False,
        "-c/-C",
        "--custom-map/--default-map",
        help="Run conversion with a custom map (-c) or the not (-C)",
    ),
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
        `bnd to-pyal M037_2024_01_01_10_00 -c  # Uses custom mapping
    """
    _check_processing_dependencies()
    from bnd.pipeline.pyaldata import run_pyaldata_conversion

    # Load config and get session path
    config = _load_config()
    session_path = config.get_local_session_path(session_name)

    # Check session directory
    _check_session_directory(session_path)

    # Run pipeline
    run_pyaldata_conversion(session_path, kilosort_flag, custom_map)

    return


@app.command()
def to_nwb(
    session_name: str,
    kilosort_flag: bool = typer.Option(
        True,
        "-k/-K",
        "--kilosort/--dont-kilosort",
        help="Run kilosort if available (-k) or dont (-K).",
    ),
    custom_map: bool = typer.Option(
        False,
        "-c/-C",
        "--custom-map/--default-map",
        help="Run conversion with a custom map (-c) or the not (-C)",
    ),
) -> None:
    """
    Convert session data into a nwb file and saves it as a .nwb

    \b
    If no kilosorted data is available it will not kilosort by default. If you want to kilosort add the flag `-k`

    \b
    Basic usage:
        `bnd to-nwb M037_2024_01_01_10_00`
        `bnd to-nwb M037_2024_01_01_10_00 -c`  # Use custom channel mapping
    """
    # TODO: Add channel map argument: no-map, default-map, custom-map
    # _check_processing_dependencies()
    from bnd.pipeline.nwb import run_nwb_conversion

    config = _load_config()
    session_path = config.get_local_session_path(session_name)

    # Check session directory
    _check_session_directory(session_path)

    # Run pipeline
    run_nwb_conversion(session_path, kilosort_flag, custom_map)
    return


@app.command()
def ksort(session_name: str = typer.Argument(help="Session name to kilosort")) -> None:
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
def up(
    session_or_animal_name: str = typer.Argument(
        help="Animal or session name: M123 or M123_2000_02_03_14_15"
    ),
):
    """
    Upload data to the server. If the file exists on the server, it won't be replaced.
    Every file in the session folder will get uploaded.

    Example usage to upload everything of a given session:
        `bnd up M017_2024_03_12_18_45`
    Upload everything of the last session:
        `bnd up M017`
    """
    if len(session_or_animal_name) > 4:  # session name
        upload_session(session_or_animal_name)
    elif len(session_or_animal_name) == 4:  # animal name
        config = _load_config()
        last_session = get_last_session(
            config.LOCAL_PATH / "raw" / session_or_animal_name
        )
        upload_session(last_session)
    else:
        raise ValueError("Input must be either a session or an animal name.")


@app.command()
def dl(
    session_name: str = typer.Argument(help="Name of session: M123_2000_02_03_14_15"),
    max_size_MB: float = typer.Option(
        0,
        "--max-size",
        help="Maximum size in MB. Any smaller file will be downloaded. Zero mean infinite size.",
    ),
    do_video: bool = typer.Option(
        False,
        "--video/--no-video",
        "-v/-V",
        help="Download video files as well, if they are smaller than `--max-size`. No video files by default.",
    ),
):
    """
    Download experimental data from a given session from the remote server.

    Example usage to download everything:
        `bnd down M017_2024_03_12_18_45 -v` will download everything, including videos
        `bnd down M017_2024_03_12_18_45` will download everything, except videos
        `bnd down M017_2024_03_12_18_45 --max-size=50` will download files smaller than 50MB
    """
    download_session(session_name, max_size_MB, do_video)


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
            typer.prompt(
                "Enter the absolute path to the root of the local data storage"
            )
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
