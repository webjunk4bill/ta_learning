"""Logging setup using Loguru."""

import sys
from loguru import logger


import sys
from pathlib import Path
from loguru import logger

def init_logger(debug: bool = False) -> None:
    """Configure Loguru to log to console and file."""
    logger.remove()

    level = "DEBUG" if debug else "INFO"

    # Console logging
    logger.add(sys.stdout, level=level)

    # File logging
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/ta_learning.log", level=level, rotation="1 MB", enqueue=True)

