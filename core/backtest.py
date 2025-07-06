import pandas as pd
import numpy as np


def backtest_signals(df: pd.DataFrame, initial_capital: float = 1_000.0) -> pd.Series:
    """
    Toy back-tester.

    • Enter long when df['signal'] == 1, short when == -1.
    • Hold the position until the signal flips or returns to 0.
    • Position size = 1 unit (no leverage).
    • Returns an equity curve Series.

    Args
    ----
    df : DataFrame with 'Close' and 'signal'.
    initial_capital : starting account value.

    Returns
    -------
    equity : pd.Series aligned with df.index.
    """
    df = df.sort_index().copy()

    # Price returns
    pct_ret = df["Close"].pct_change().fillna(0.0)

    # Build dynamic position including entry and double-down scaling
    entry = df.get("entry_signal", pd.Series(0, index=df.index))
    dd    = df.get("double_down",    pd.Series(0, index=df.index))

    position = []
    prev_pos = 0
    for sig, ent, ddown in zip(df["signal"], entry, dd):
        if sig == 0:
            pos = 0
        elif ent != 0:
            pos = ent
        elif ddown != 0:
            pos = ddown
        else:
            pos = prev_pos
        position.append(pos)
        prev_pos = pos
    position = pd.Series(position, index=df.index)
  
    # Strategy return per bar
    strat_ret = pct_ret * position.shift(1).fillna(0)

    # Equity curve
    equity = (1 + strat_ret).cumprod() * initial_capital
    equity.name = "Equity"
    return equity