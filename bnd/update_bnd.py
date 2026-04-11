import platform
import shutil
import subprocess
import warnings
from pathlib import Path

from .logger import set_logging
from .config import _get_package_path

logger = set_logging(__name__)

_REPO_URL = "https://github.com/BeNeuroLab/bnd.git"


def _find_repo_path() -> Path | None:
    """Return the git repo root if bnd was installed from a local clone, else None."""
    pkg = _get_package_path()
    # Walk up looking for .git (editable installs live inside the repo)
    for parent in (pkg, *pkg.parents):
        if (parent / ".git").is_dir():
            return parent
    return None


def _run_git_command(repo_path: Path, command: list[str]) -> str:
    """
    Run a git command in the specified repository and return its output

    Parameters
    ----------
    repo_path : Path
        Path to the git repository to run the command in.
    command : list[str]
        Git command to run, as a list of strings.
        E.g. ["log", "HEAD..origin/main", "--oneline"]

    Returns
    -------
    The output of the git command as a string.
    """
    repo_path = Path(repo_path)

    if not repo_path.is_absolute():
        raise ValueError(f"{repo_path} is not an absolute path")

    if not (repo_path / ".git").exists():
        raise ValueError(f"{repo_path} is not a git repository")

    result = subprocess.run(
        ["git", "-C", repo_path.absolute()] + command, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"Git command failed: {result.stderr}")

    return result.stdout.strip()


def _get_new_commits(repo_path: Path) -> list[str]:
    """
    Check for new commits from origin/main of the specified repository.

    Parameters
    ----------
    repo_path : Path
        Path to the git repository.

    Returns
    -------
    Each new commit as a string in a list.
    """
    repo_path = Path(repo_path)

    # Fetch the latest changes from the remote repository
    _run_git_command(repo_path, ["fetch"])

    # Check if origin/main has new commits compared to the local branch
    new_commits = _run_git_command(repo_path, ["log", "HEAD..origin/main", "--oneline"])

    # filter empty lines and strip whitespaces
    return [commit.strip() for commit in new_commits.split("\n") if commit.strip() != ""]


def check_for_updates() -> bool:
    """
    Check if the package has new commits on the origin/main branch.

    Returns True if new commits are found, False otherwise.
    """
    repo_path = _find_repo_path()

    if repo_path is None:
        print(
            "bnd is not installed from a local git clone.\n"
            "To update, run:\n"
            f'  pipx install --force "bnd @ git+{_REPO_URL}"'
        )
        return False

    new_commits = _get_new_commits(repo_path)

    if len(new_commits) > 0:
        print("New commits found, run `bnd self-update` to update the package.")
        for commit in new_commits:
            print(f" - {commit}")

        return True

    print("No new commits found, package is up to date.")
    return False


def update_bnd(print_new_commits: bool = True) -> None:
    """
    Update bnd. Uses git pull for editable installs, or pipx reinstall otherwise.

    Parameters
    ----------
    print_new_commits

    """
    repo_path = _find_repo_path()

    if repo_path is None:
        # pipx / pip install — can't reinstall ourselves while running
        print(
            "bnd is installed via pipx. To update, run this in your terminal:\n\n"
            f'  pipx install --force "bnd @ git+{_REPO_URL}"\n'
        )
        return

    new_commits = _get_new_commits(repo_path)

    if len(new_commits) > 0:
        print("New commits found, pulling changes...")

        _run_git_command(repo_path, ["pull", "origin", "main"])

        print(1 * "\n")
        print("Package updated successfully.")
        print("\n")

        if print_new_commits:
            print("New commits:")
            for commit in new_commits:
                print(f" - {commit}")
    else:
        print("Package appears to be up to date, no new commits found.")
