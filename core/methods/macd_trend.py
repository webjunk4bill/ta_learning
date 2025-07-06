import pandas as pd
import numpy as np
from core.indicators.macd import macd


def analyze(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_window: int = 9,
) -> pd.DataFrame:
    """MACD Trend Following strategy.

    Buy when MACD crosses above the signal line.
    Sell when MACD crosses below the signal line.
    Adds columns 'MACD', 'MACD_signal', 'MACD_hist', 'signal', and 'reason'.
    """
    df = macd(df, fast=fast, slow=slow, signal=signal_window)

    macd_col = "MACD"
    sig_col = "MACD_signal"

    prev_macd = df[macd_col].shift(1)
    prev_sig = df[sig_col].shift(1)

    cross_up = (df[macd_col] > df[sig_col]) & (prev_macd <= prev_sig)
    cross_down = (df[macd_col] < df[sig_col]) & (prev_macd >= prev_sig)

    df["signal"] = 0
    df.loc[cross_up, "signal"] = 1
    df.loc[cross_down, "signal"] = -1

    df["reason"] = ""
    df.loc[cross_up, "reason"] = "MACD crossed above signal"
    df.loc[cross_down, "reason"] = "MACD crossed below signal"

    return df

