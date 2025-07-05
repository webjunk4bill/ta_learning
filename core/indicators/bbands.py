import pandas as pd

def bollinger(df: pd.DataFrame, window: int = 20, n_sigma: float = 2.0) -> pd.DataFrame:
    sma = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()

    df[f"BB_M_{window}"] = sma
    df[f"BB_U_{window}"] = sma + n_sigma * std
    df[f"BB_L_{window}"] = sma - n_sigma * std
    return df