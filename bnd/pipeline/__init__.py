"""
Pipeline package
"""


def _check_processing_dependencies():
    try:
        from .kilosort import run_kilosort_on_session
        from .nwb import run_nwb_conversion
        from .pyaldata import run_pyaldata_conversion
    except Exception as e:
        raise ImportError(
            f"Could not import processing dependencies: {e}. Update your environment "
            "with `conda env update -n bnd --file=processing_env.yml`"
        )
    return
