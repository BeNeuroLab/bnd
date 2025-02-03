from unittest.mock import MagicMock, patch

import pytest

from bnd.data_transfer import _upload_files


@pytest.fixture
def mock_config(tmp_path):
    """
    Mocks the config object with temporary local and remote paths.
    """
    mock_config = MagicMock()

    # Create temporary directories for local and remote session paths
    local_session_path = tmp_path / "local_session"
    remote_session_path = tmp_path / "remote_session"
    local_session_path.mkdir()
    remote_session_path.mkdir()
    return mock_config, local_session_path, remote_session_path


def test_upload_session_assertion(mock_config):
    """
    Test that an assertion error is raised when a remote file already exists.
    """
    config, local_session_path, remote_session_path = mock_config

    # Create a test file in the local session path
    test_file = local_session_path / "test_file.txt"
    test_file.touch()
    test_files = [test_file]

    # Create the same file in the remote session path to trigger the assertion
    remote_file = remote_session_path / "test_file.txt"
    remote_file.touch()
    remote_files = [remote_file]

    # Mock the _load_config function using unittest.mock.patch
    with patch("bnd.config._load_config", return_value=config):
        with pytest.raises(AssertionError, match="Remote file already exists"):
            _upload_files(test_files, remote_files)
