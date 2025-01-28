"""This module contains functions for uploading and downloading data to and from the server."""

from pathlib import Path
import os
import shutil
import sys

from rich.progress import Progress

from bnd.config import (
    _load_config,
    find_file,
    list_dirs,
    list_session_datetime,
    get_last_session,
)

config = _load_config()


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
    if int(max_size_MB) <= 0:
        max_size = float('inf')
    else:
        max_size = max_size_MB / 1024  # convert to bytes
    remote_session_path = config.get_remote_session_path(session_name)
    local_session_path  = config.get_local_session_path(session_name)
    if local_session_path.exists():
        print(f"Session {session_name} already exists locally.")
        return
    remote_files = remote_session_path.rglob("*")
    
    for file in remote_files:
        if file.suffix in config.video_formats:
            continue  # video files dealt with later

        local_file = config.convert_to_local(file)
        assert not local_file.exists(), "Local file already exists. This should not happen."
        
        if file.stat().st_size < max_size:
            with Progress() as progress:
                with progress.open(file, "rb", description=file.name) as src:
                    with open(local_file, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        else:
            print(f"File {file.name} is too large. Skipping.")
    
    # TODO: handle video files
