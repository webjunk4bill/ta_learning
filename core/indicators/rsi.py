import pandas as pd

def rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index (RSI)

    Adds a column 'RSI_{window}' to the DataFrame.
    """
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use exponential moving average for smoothing
    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    col_name = f"RSI_{window}"
    df[col_name] = 100 - (100 / (1 + rs))
    return df