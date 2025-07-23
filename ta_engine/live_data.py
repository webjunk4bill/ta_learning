import ccxt
import pandas as pd


def fetch_ohlcv(symbol: str, exchange_name: str, timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
