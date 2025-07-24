"""Logging setup using Loguru."""

import sys
from loguru import logger


def init_logger(debug: bool = False) -> None:
    """Configure Loguru based on debug flag."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if debug else "INFO")

