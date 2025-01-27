from bnd import set_logging

logger = set_logging(__name__)


def run_nwb_conversion(session_path, kilosort_flag, map):
    if map is not None and map not in ["default", "custom"]:
        raise ValueError(
            f"Argument {map} does not match expected options 'default' or 'custom'"
    )
    return
