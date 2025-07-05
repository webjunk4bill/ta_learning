import pandas as pd
from core.indicators.moving_averages import sma
from core.indicators.rsi import rsi

def analyze(
    df: pd.DataFrame,
    sma_window: int = 20,
    rsi_window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0
) -> pd.DataFrame:
    """
    Apply a mean-reversion strategy:
    - Compute SMA and RSI.
    - Signal =  1 for buy when price < SMA and RSI < oversold.
    - Signal = -1 for sell when price > SMA and RSI > overbought.
    - Otherwise signal = 0.
    Adds columns: 'SMA_{sma_window}', 'RSI_{rsi_window}', 'signal', 'reason'.
    """
    # Calculate indicators
    df = sma(df, window=sma_window)
    df = rsi(df, window=rsi_window)

    # Define column names
    sma_col = f"SMA_{sma_window}"
    rsi_col = f"RSI_{rsi_window}"

    # Initialize signal and reason columns
    df['signal'] = 0
    df['reason'] = ''

    # Define conditions
    buy_cond = (df['Close'] < df[sma_col]) & (df[rsi_col] < oversold)
    sell_cond = (df['Close'] > df[sma_col]) & (df[rsi_col] > overbought)

    # Apply signals
    df.loc[buy_cond, 'signal'] = 1
    df.loc[sell_cond, 'signal'] = -1

    # Add reasons
    df.loc[buy_cond, 'reason'] = f"Price below {sma_col} and {rsi_col} below {oversold}"
    df.loc[sell_cond, 'reason'] = f"Price above {sma_col} and {rsi_col} above {overbought}"

    return df
