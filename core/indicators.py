import pandas as pd


def sma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Simple Moving Average"""
    col_name = f"SMA_{window}"
    df[col_name] = df["Close"].rolling(window=window, min_periods=1).mean()
    return df


def ema(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Exponential Moving Average"""
    col_name = f"EMA_{window}"
    df[col_name] = df["Close"].ewm(span=window, adjust=False).mean()
    return df


def rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Relative Strength Index"""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    df[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    return df


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence"""
    fast_ema = df["Close"].ewm(span=fast, adjust=False).mean()
    slow_ema = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    df["MACD"] = macd_line
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = macd_line - signal_line
    return df


def bollinger(df: pd.DataFrame, window: int = 20, n_sigma: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands"""
    sma = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    df[f"BB_M_{window}"] = sma
    df[f"BB_U_{window}"] = sma + n_sigma * std
    df[f"BB_L_{window}"] = sma - n_sigma * std
    return df


def compute_indicators(df: pd.DataFrame, config: dict) -> dict:
    """Compute multiple indicators based on a config dict."""
    results: dict = {}
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    if "rsi" in config:
        window = config["rsi"].get("window", 14)
        rsi_df = rsi(df.copy(), window=window)
        results["rsi"] = rsi_df[f"RSI_{window}"].tolist()

    if "macd" in config:
        mc = config["macd"]
        fast = mc.get("fast", 12)
        slow = mc.get("slow", 26)
        sig = mc.get("signal", 9)
        macd_df = macd(df.copy(), fast=fast, slow=slow, signal=sig)
        results["macd"] = {
            "macd_line": macd_df["MACD"].tolist(),
            "signal_line": macd_df["MACD_signal"].tolist(),
            "histogram": macd_df["MACD_hist"].tolist(),
        }

    if "bollinger" in config:
        bc = config["bollinger"]
        window = bc.get("window", 20)
        std_dev = bc.get("std_dev", 2)
        boll_df = bollinger(df.copy(), window=window, n_sigma=std_dev)
        results["bollinger"] = {
            "upper": boll_df[f"BB_U_{window}"].tolist(),
            "lower": boll_df[f"BB_L_{window}"].tolist(),
            "mid": boll_df[f"BB_M_{window}"].tolist(),
        }

    return results
