import pandas as pd

def sma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Simple Moving Average (SMA)

    Adds a column 'SMA_{window}' to the DataFrame.
    """
    col_name = f"SMA_{window}"
    df[col_name] = df["Close"].rolling(window=window, min_periods=1).mean()
    return df

def ema(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Exponential Moving Average (EMA)

    Adds a column 'EMA_{window}' to the DataFrame.
    """
    col_name = f"EMA_{window}"
    df[col_name] = df["Close"].ewm(span=window, adjust=False).mean()
    return df