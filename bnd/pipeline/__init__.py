"""
Pipeline package
"""


def _check_processing_dependencies():
    try:
        from bnd.pipeline.kilosort import run_kilosort_on_session
        from bnd.pipeline.nwb import run_nwb_conversion
        from bnd.pipeline.pyaldata import run_pyaldata_conversion
    except Exception as e:
        raise ImportError(
            "Could not import processing dependencies. Update your environment "
            "with `conda env update -n bnd --file=processing_env.yml`"
        )
    return
