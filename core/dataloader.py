import pandas as pd
from typing import Dict, Any

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV data from the given path, parse dates, and set the first column as the index.
    Expects CSV with a datetime index column (e.g., 'Date') and columns: Open, High, Low, Close, Volume.
    """
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
    ohlc: Dict[str, Any] = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    df = df.sort_index()
    df_resampled = df.resample(timeframe).agg(ohlc).dropna()  # type: ignore[arg-type]
    return df_resampled
