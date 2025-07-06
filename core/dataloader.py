import pandas as pd
from typing import Dict, Any
from loguru import logger

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV data from the given path, parse dates, and set the first column as the index.
    Expects CSV with a datetime index column (e.g., 'Date') and columns: Open, High, Low, Close, Volume.
    """
    logger.debug(f"Loading data from {path}")
    # Read CSV; expect columns 'date', 'price', 'volume', or standard OHLCV
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.rename(columns={'date': 'Date', 'price': 'Close', 'volume': 'Volume'})
    df.set_index('Date', inplace=True)

    # If Open/High/Low are missing, use Close for all
    for col in ('Open', 'High', 'Low'):
        if col not in df.columns:
            df[col] = df['Close']

    return df

def resample_df(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample a minute-level DataFrame to the specified timeframe (e.g., '1D', '1H', '15T').
    Aggregates OHLC and sums Volume.
    """
    # Normalize deprecated frequency aliases: 'H'→'h', 'T'→'min'
    timeframe_norm = timeframe
    if timeframe_norm.endswith(("H", "h")):
        timeframe_norm = timeframe_norm[:-1] + "h"  # e.g., '1H' → '1h'
    elif timeframe_norm.endswith(("T", "t")):
        timeframe_norm = timeframe_norm[:-1] + "min"  # e.g., '15T' → '15min'

    ohlc: Dict[str, Any] = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    logger.debug(f"Resampling to {timeframe}")
    df = df.sort_index()
    df_resampled = df.resample(timeframe_norm).agg(ohlc).dropna()  # type: ignore[arg-type]
    return df_resampled
