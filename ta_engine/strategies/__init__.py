"""Strategy package providing registry access."""

from .base import Strategy
from .registry import REGISTRY, get_strategy

__all__ = ["Strategy", "REGISTRY", "get_strategy"]
