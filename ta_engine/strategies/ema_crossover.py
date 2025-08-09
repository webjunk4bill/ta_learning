from __future__ import annotations

import pandas as pd

from .base import Strategy


class EmaCrossover(Strategy):
    """Exponential moving average crossover strategy."""

    def run(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        short_window = int(config.get("short_window", 12))
        long_window = int(config.get("long_window", 26))
        if df.empty or len(df) < max(short_window, long_window):
            raise ValueError("Insufficient data for EMA windows")

        out = df.copy()
        out["Price"] = out["Close"]
        out["EMA_short"] = out["Close"].ewm(span=short_window, adjust=False).mean()
        out["EMA_long"] = out["Close"].ewm(span=long_window, adjust=False).mean()

        prev_short = out["EMA_short"].shift(1)
        prev_long = out["EMA_long"].shift(1)
        bullish = (prev_short <= prev_long) & (out["EMA_short"] > out["EMA_long"])
        bearish = (prev_short >= prev_long) & (out["EMA_short"] < out["EMA_long"])

        out["Crossover"] = "none"
        out.loc[bullish, "Crossover"] = "bullish"
        out.loc[bearish, "Crossover"] = "bearish"

        out["Signal"] = "neutral"
        out.loc[bullish, "Signal"] = "long"
        out.loc[bearish, "Signal"] = "short"

        return out[["Price", "EMA_short", "EMA_long", "Crossover", "Signal"]]
