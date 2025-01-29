"""This module contains functions for uploading and downloading data to and from the server."""

import shutil

from bnd import set_logging
from bnd.config import _load_config

logger = set_logging(__name__)


def upload_session(session_name: str) -> None:
    """
    Upload a session to the server.
    Every file in the session folder will get uploaded.
    No file on the server will get overwritten
    """
    config = _load_config()
    remote_session_path = config.get_remote_session_path(session_name)
    local_session_path = config.get_local_session_path(session_name)

    local_files = list(local_session_path.rglob("*"))
    remote_files = list(remote_session_path.rglob("*"))
    assert isinstance(
        remote_files, list
    ), "`remote_files` must be a list, otherwise the list comprehension below will break"
    pending_local_files = [
        file
        for file in local_files
        if config.convert_to_remote(file) not in remote_files and file.is_file()
    ]
    if not pending_local_files:
        logger.info(f"No files to upload.")
        return

    # Check if file names follow the session name convention
    for file in pending_local_files:
        if not config.file_name_ok(file.name):
            logger.warning(f'Unusual file name: "{file.name}"')

    response = input(f"\nUpload session {session_name} (y/n)? ").strip().lower()
    if "n" in response:
        logger.info("Upload aborted.")
        return

    # Upload the files
    for file in pending_local_files:
        remote_file = config.convert_to_remote(file)
        assert not remote_file.exists(), \
            "Remote file already exists. This should not happen."
        # Ensure the destination directory exists
        remote_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, remote_file)
        logger.info(f'Uploaded "{file.name}"')

    logger.info("Upload complete.")


def download_session(session_name: str, max_size_MB: float, do_video: bool) -> None:
    """
    Download a session from the server.
    """
    config = _load_config()

    if int(max_size_MB) <= 0:
        max_size = float("inf")
    else:
        max_size = max_size_MB * 1024 * 1024  # convert to bytes

    remote_session_path = config.get_remote_session_path(session_name)
    local_session_path = config.get_local_session_path(session_name)
    if local_session_path.exists():
        logger.error(f"Session {session_name} already exists locally.")
        return

    # Excluding directories as `rglob()` returns directories as well
    remote_files = [file for file in remote_session_path.rglob("*") if file.is_file()]

    for file in remote_files:
        if file.suffix in config.video_formats and not do_video:
            continue  # skip video files

        if file.stat().st_size < max_size:
            local_file = config.convert_to_local(file)
            assert (
                not local_file.exists()
            ), "Local file already exists. This should not happen."
            # Ensure the destination directory exists
            local_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, local_file)
            logger.info(f"Downloaded {file.name}")
        else:
            logger.warning(f"File {file.name} is too large. Skipping.")
    logger.info("Download complete.")
