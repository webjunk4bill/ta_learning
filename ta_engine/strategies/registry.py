from __future__ import annotations

from .base import Strategy
from .ema_crossover import EmaCrossover


REGISTRY: dict[str, Strategy] = {
    "ema_crossover": EmaCrossover(),
}


def get_strategy(name: str) -> Strategy:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown strategy: {name}")
