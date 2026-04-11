"""
Pipeline package
"""


def _check_processing_dependencies():
    try:
        from .kilosort import run_kilosort_on_session
        from .nwb import run_nwb_conversion
        from .pyaldata import run_pyaldata_conversion
    except ImportError as e:
        raise ImportError(
            f"Missing processing dependencies: {e}.\n"
            "Install them with:\n"
            '  pipx install --force "bnd[processing] @ git+https://github.com/BeNeuroLab/bnd.git"\n'
            "or:\n"
            '  pip install "bnd[processing]"'
        ) from e
    return
