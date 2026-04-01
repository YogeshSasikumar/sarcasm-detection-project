"""
logger.py - Centralized logging for the Sarcasm Detection System
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from utils.config import LOG_FILE, LOG_LEVEL, LOGS_DIR

os.makedirs(LOGS_DIR, exist_ok=True)

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with rotating file handler + console handler.
    Call once per module: logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Rotating file handler (5 MB max, 3 backups)
    try:
        fh = RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except (OSError, PermissionError):
        logger.warning("Could not create file handler for %s", LOG_FILE)

    logger.propagate = False
    return logger
