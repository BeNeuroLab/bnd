"""This module contains functions for uploading and downloading data to and from the server."""

from pathlib import Path
import shutil

from bnd.config import (
    _load_config,
    find_file,
    list_dirs,
    list_session_datetime,
    get_last_session,
)


def upload_session():
    return


def download_session(
    session_name: str,
    max_size_MB: float,
    do_video: bool
) -> None:
    """
    Download a session from the remote server.
    """
    config = _load_config()

    if int(max_size_MB) <= 0:
        max_size = float('inf')
    else:
        max_size = max_size_MB * 1024 * 1024  # convert to bytes

    remote_session_path = config.get_remote_session_path(session_name)
    local_session_path  = config.get_local_session_path(session_name)
    if local_session_path.exists():
        print(f"Session {session_name} already exists locally.")
        return
    remote_files = remote_session_path.rglob("*")
    
    for file in remote_files:
        if file.suffix in config.video_formats and not do_video:
            continue  # skip video files

        if file.stat().st_size < max_size:
            local_file = config.convert_to_local(file)
            assert not local_file.exists(), \
                "Local file already exists. This should not happen."
            # Ensure the destination directory exists
            local_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, local_file)
            print(f"Downloaded {file.name}")
        else:
            print(f"File {file.name} is too large. Skipping.")
    return
