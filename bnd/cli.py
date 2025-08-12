import shutil
from pathlib import Path
from typing import List

import typer
from typing_extensions import Annotated

from rich import print

from .config import (
    _check_is_git_track,
    _check_root,
    _check_session_directory,
    _get_env_path,
    _get_package_path,
    _load_config,
    get_last_session,
    list_session_datetime,
)
from .data_transfer import download_session, upload_session
from .pipeline import _check_processing_dependencies
from .update_bnd import check_for_updates, update_bnd

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
    validate: bool = typer.Option(
        False,
        "-v/-V",
        "--validate/--no-validate",
        help="Validate spike times alignment (-v) or skip validation (-V)",
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
        `bnd to-pyal M037_2024_01_01_10_00`  # Kilosorts data and converts to pyaldata
        `bnd to-pyal M037_2024_01_01_10_00 -c`  # Uses custom mapping
        `bnd to-pyal M037_2024_01_01_10_00 -v`  # Validates spike alignment
    """
    _check_processing_dependencies()
    from .pipeline.pyaldata import run_pyaldata_conversion

    # Load config and get session path
    config = _load_config()
    session_path = config.get_local_session_path(session_name)

    # Check session directory
    _check_session_directory(session_path)

    # Run pipeline
    run_pyaldata_conversion(session_path, kilosort_flag, custom_map, validate)

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
    validate: bool = typer.Option(
        False,
        "-v/-V",
        "--validate/--no-validate",
        help="Validate spike times alignment after conversion (-v) or skip validation (-V)",
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
        `bnd to-nwb M037_2024_01_01_10_00 -v`  # Validate spike alignment after conversion
    """
    # TODO: Add channel map argument: no-map, default-map, custom-map
    # _check_processing_dependencies()
    from .pipeline.nwb import run_nwb_conversion

    config = _load_config()
    session_path = config.get_local_session_path(session_name)

    # Check session directory
    _check_session_directory(session_path)

    # Run pipeline
    run_nwb_conversion(session_path, kilosort_flag, custom_map)
    
    # Validate spike times after conversion if requested
    if validate:
        from .pipeline.spike_validation import check_negative_spike_times
        nwb_path = session_path / f"{session_name}.nwb"
        
        if nwb_path.exists():
            has_negative, min_spike_time = check_negative_spike_times(nwb_path)
            
            if has_negative:
                print(f"\n[green]✓ Validation passed: Found negative spike times (min = {min_spike_time:.3f}s)[/green]")
                print(f"Alignment between pycontrol and ephys data looks correct.")
            elif min_spike_time is None:
                print(f"\n[yellow]⚠ No spike data found for validation[/yellow]")
            else:
                print(f"\n[red]✗ Validation failed: No negative spike times found (min = {min_spike_time:.3f}s)[/red]")
                print(f"This suggests alignment issues between pycontrol and ephys data.")
                print(f"Consider checking your raw data and re-running conversion.")
        else:
            print(f"\n[red]✗ NWB file not found for validation[/red]")
    
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
    from .pipeline.kilosort import run_kilosort_on_session

    config = _load_config()
    session_path = config.get_local_session_path(session_name)

    # Check session directory
    _check_session_directory(session_path)

    # Run pipeline
    run_kilosort_on_session(session_path)
    return


@app.command()
def validate_spikes(
    session_name: str = typer.Argument(..., help="Session name to validate"),
    delete_if_invalid: bool = typer.Option(
        True,
        "-d/-D",
        "--delete/--no-delete",
        help="Delete files if no negative spikes found (-d) or keep them (-D)",
    ),
    batch: bool = typer.Option(
        False,
        "-b/-B",
        "--batch/--single",
        help="Validate all sessions for this animal (-b) or single session (-B)",
    ),
) -> None:
    """
    Validate spike times alignment in NWB files.
    
    Checks for negative spike times which indicate correct pycontrol-ephys alignment.
    If no negative spikes are found, optionally deletes NWB and pyaldata files.

    \b
    Basic usage:
        `bnd validate-spikes M037_2024_01_01_10_00`  # Validate single session
        `bnd validate-spikes M037_2024_01_01_10_00 -D`  # Check only, don't delete
        `bnd validate-spikes M037 -b`  # Validate all M037 sessions
    """
    _check_processing_dependencies()
    from .pipeline.spike_validation import validate_and_clean_session, batch_validate_sessions
    
    config = _load_config()
    
    if batch:
        # Extract animal name from session
        animal_name = session_name.split("_")[0]
        # Get animal directory and list sessions
        animal_path = config.LOCAL_PATH / "raw" / animal_name
        
        if not animal_path.exists():
            print(f"[red]Animal directory not found: {animal_path}[/red]")
            return
            
        try:
            # Get all sessions for this animal
            _, session_names = list_session_datetime(animal_path)
            session_paths = [config.get_local_session_path(name) for name in session_names]
            
            if not session_paths:
                print(f"[red]No sessions found for animal {animal_name}[/red]")
                return
            
            print(f"[bold]Validating {len(session_paths)} sessions for {animal_name}[/bold]")
            batch_validate_sessions(session_paths, delete_if_no_negative=delete_if_invalid)
            
        except Exception as e:
            print(f"[red]Error getting sessions for {animal_name}: {str(e)}[/red]")
            return
    else:
        # Single session validation
        session_path = config.get_local_session_path(session_name)
        _check_session_directory(session_path)
        
        needs_reconversion = validate_and_clean_session(
            session_path,
            delete_if_no_negative=delete_if_invalid
        )
        
        if needs_reconversion and delete_if_invalid:
            print(f"\n[yellow]Session {session_name} needs re-conversion.[/yellow]")
            print("Run: `bnd to-pyal {session_name}` to re-convert")
    
    return


# ================================== Data Transfer ========================================


@app.command()
def replace_processed(
    session_name: str = typer.Argument(..., help="Session name to replace processed files for"),
    auto_confirm: bool = typer.Option(
        False,
        "-y/-Y",
        "--yes/--no-yes",
        help="Auto-confirm replacement (-y) or prompt for confirmation (-Y)",
    ),
) -> None:
    """
    Replace processed files (.nwb and .mat) on the server.
    
    This command specifically replaces NWB and PyAlData files that may have been
    corrected due to alignment issues. It will overwrite existing files on RDS.
    
    \b
    Basic usage:
        `bnd replace-processed M037_2024_01_01_10_00`  # Replace with confirmation
        `bnd replace-processed M037_2024_01_01_10_00 -y`  # Auto-confirm
    """
    _check_processing_dependencies()
    from .data_transfer import replace_processed_files
    
    config = _load_config()
    session_path = config.get_local_session_path(session_name)
    
    # Check session directory
    _check_session_directory(session_path)
    
    # Find processed files to replace
    processed_files = []
    
    # NWB files
    nwb_files = list(session_path.glob("*.nwb"))
    processed_files.extend(nwb_files)
    
    # PyAlData files (could be partitioned)
    mat_files = list(session_path.glob("*_pyaldata*.mat"))
    
    processed_files.extend(mat_files)
    
    if not processed_files:
        print(f"[yellow]No processed files (.nwb or .mat) found in {session_name}[/yellow]")
        return
    
    # Show what will be replaced
    print(f"[bold]Files to replace on RDS:[/bold]")
    for file in processed_files:
        print(f"  - {file.name}")
    
    # Confirm replacement
    if not auto_confirm:
        response = input(f"\nReplace these {len(processed_files)} files on RDS? (y/n): ").strip().lower()
        if "y" not in response:
            print("[yellow]Operation cancelled.[/yellow]")
            return
    
    # Replace files
    try:
        replace_processed_files(session_name, processed_files)
        print(f"[green]✓ Successfully replaced processed files for {session_name}[/green]")
    except Exception as e:
        print(f"[red]✗ Error replacing files: {str(e)}[/red]")
    
    return


@app.command()
def up(
    session_or_animal_name: str = typer.Argument(
        help="Animal or session name: M123 or M123_2000_02_03_14_15"
    ),
):
    """
    Upload data to the server. If the file exists on the server, it won't be replaced.
    Every file in the session folder will get uploaded.
    If animal name give, the last session will get uploaded.

    \b
    Example usage to upload everything of a given session:
        `bnd up M017_2024_03_12_18_45`
    Upload everything of the last session:
        `bnd up M017`
    """
    if len(session_or_animal_name) > 4:  # session name
        upload_session(session_or_animal_name)
    elif len(session_or_animal_name) == 4:  # animal name
        config = _load_config()
        last_session = get_last_session(config.LOCAL_PATH / "raw" / session_or_animal_name)
        upload_session(last_session)
    else:
        raise ValueError("Input must be either a session or an animal name.")


@app.command()
def dl(
    session_name: str = typer.Argument(help="Name of session: M123_2000_02_03_14_15"),
    file_extension: Annotated[str, typer.Argument(help="One file type to download")] = ".*",
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
    Download data of a given session from the remote server.
    If session exists locally, only missing files will be downloaded.
    if session name is not complete (`M123_2025_02_03`), it will try to find a similar session.

    \b
    Example usage to download everything:
        `bnd dl M017_2024_03_12_18_45 -v` will download everything, including videos
        `bnd dl M017_2024_03_12_18_45` will download everything, except videos
        `bnd dl M017_2024_03_12_18_45 --max-size=50` will download files smaller than 50MB
        `bnd dl M056_2025_03_01 .mat` will download all the '.mat' files from the matching session
    """
    download_session(session_name, file_extension, max_size_MB, do_video)


# =================================== Batch ==========================================


@app.command()
def batch_ks(animal_list: List[str]):
    """
    Download data from all sessions of every animal in animal_list,
    kilosort,
    convert to pyal,
    and upload back to the server.

    \b
    Example usage:
    `yes` in Linux replies to all the prompts with 'yes'.
        `yes | bnd batch-ks M123 M124 M125`
    """
    config = _load_config()
    for animal in animal_list:
        try:
            assert len(animal) == 4, "Animal name must be 4 characters long"

            _, session_list = list_session_datetime(config.REMOTE_PATH / "raw" / animal)

            for session in session_list:
                try:
                    dl(session, max_size_MB=0, do_video=False)
                    to_pyal(session, kilosort_flag=True, custom_map=False)
                    up(session)
                    shutil.rmtree(config.LOCAL_PATH / "raw" / animal / session)
                except Exception as e:
                    print("Error in session:", session)
                    print(e)
                    shutil.rmtree(
                        config.LOCAL_PATH / "raw" / animal / session, ignore_errors=True
                    )
                    continue
        except AssertionError as e:
            print("Error:", animal)
            print(e)
            continue


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
