import inspect
import os
from pathlib import Path

from loguru import logger

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def add_file_handler(name=None, log_dir=LOG_DIR):
    """
    Add a handler to the logger.
    If no name is provided, the name of the calling script will be used.
    Each call will return a new, independent logger instance.

    Args:
        name (str, optional): The name of the logger, default to None
        log_dir (str, optional): The directory to save the log file, default to LOG_DIR

    """
    # If no name is provided, the name of the calling script will be used.
    if name is None:
        frame = inspect.stack()[1]
        caller_path = frame.filename
        name = Path(caller_path).stem

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")
    logger.add(log_file, rotation="10 MB")
