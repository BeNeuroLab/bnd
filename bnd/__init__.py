import logging
import warnings


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.captureWarnings(True)

    logger = logging.getLogger(file_name)

    def custom_warning_handler(
        message, category, filename, lineno, file=None, line=None
    ):
        logger.warning(f"{category.__name__}: {message}")

    # Set the custom handler
    warnings.showwarning = custom_warning_handler

    return logger
