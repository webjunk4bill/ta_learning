"""Logging setup using Loguru with Rich formatting."""

from loguru import logger
from rich.logging import RichHandler


def init_logger(level: str = "INFO") -> None:
    """Configure Loguru to use RichHandler."""
    logger.remove()
    logger.add(RichHandler(markup=True, rich_tracebacks=True), level=level)

