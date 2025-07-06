import pandas as pd
import numpy as np
from core.indicators.moving_averages import sma


def analyze(
    df: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
) -> pd.DataFrame:
    """Moving Average Crossover strategy.

    Buy when short SMA crosses above long SMA.
    Sell when short SMA crosses below long SMA.
    Adds columns 'SMA_{short_window}', 'SMA_{long_window}', 'signal', and 'reason'.
    """
    df = sma(df, window=short_window)
    df = sma(df, window=long_window)

    short_col = f"SMA_{short_window}"
    long_col = f"SMA_{long_window}"

    prev_short = df[short_col].shift(1)
    prev_long = df[long_col].shift(1)

    cross_up = (df[short_col] > df[long_col]) & (prev_short <= prev_long)
    cross_down = (df[short_col] < df[long_col]) & (prev_short >= prev_long)

    df["signal"] = 0
    df.loc[cross_up, "signal"] = 1
    df.loc[cross_down, "signal"] = -1

    df["reason"] = ""
    df.loc[cross_up, "reason"] = f"{short_col} crossed above {long_col}"
    df.loc[cross_down, "reason"] = f"{short_col} crossed below {long_col}"

    return df

