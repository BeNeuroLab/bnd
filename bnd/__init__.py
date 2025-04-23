import logging
import warnings

# Package interface
from .cli import dl, up, to_pyal, batch_ks
from .data_transfer import upload_session, download_session, download_animal

__all__ = [dl, up, to_pyal, batch_ks, upload_session, download_session, download_animal]

# Create a logger for the package
def set_logging(
    file_name: str,
) -> logging.Logger:
    """
    Set project-wide logging

    Parameters
    ----------
    file_name: str
        Name of the module being logged

    Returns
    -------
    logger: logging.Logger
        logger object
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logging.captureWarnings(True)

    logger = logging.getLogger(file_name)

    def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
        logger.warning(f"{category.__name__}: {message}")

    # Set the custom handler
    warnings.showwarning = custom_warning_handler

    return logger
