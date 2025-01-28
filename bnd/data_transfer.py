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
    remote_session_path = config.get_remote_session_path(session_name)
    local_session_path  = config.get_local_session_path(session_name)
    if local_session_path.exists():
        print(f"Session {session_name} already exists locally.")
        return
    remote_files = remote_session_path.rglob("*")
    
    
    with Progress() as progress:
        desc = os.path.basename(sys.argv[1])
        with progress.open(sys.argv[1], "rb", description=desc) as src:
            with open(sys.argv[2], "wb") as dst:
                shutil.copyfileobj(src, dst)