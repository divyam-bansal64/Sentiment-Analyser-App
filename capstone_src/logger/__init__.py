import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Constants for log configuration
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of backup log files to keep

# Construct log file path at project root
ROOT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR_PATH = ROOT_DIR / LOG_DIR
LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOG_DIR_PATH / LOG_FILE

def configure_logger():
    """
    Configures logging with a rotating file handler and a console handler.
    Prevents adding duplicate handlers if configured multiple times.
    """
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handler creation on multiple imports/calls
    if not logger.handlers:
        formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

        # File handler with rotation
        file_handler = RotatingFileHandler(
            LOG_FILE_PATH, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

# Configure the logger on module import
configure_logger()