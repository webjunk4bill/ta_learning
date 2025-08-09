from ccxt.base.errors import ExchangeNotAvailable
import requests

import ccxt
import pandas as pd


def fetch_ohlcv(symbol: str, exchange_name: str, timeframe: str = "1h", limit: int = 200, strict: bool = True) -> pd.DataFrame:
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class()

        exchange.load_markets()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")

        if df.empty:
            raise ValueError(
                f"Received no data from {exchange_name} for {symbol} (0 rows)"
            )

        if strict and df.shape[0] < 10:
            raise ValueError(
                f"Received insufficient data from {exchange_name} for {symbol} ({df.shape[0]} rows)"
            )

        return df

    except ExchangeNotAvailable as e:
        raise ValueError(f"Exchange '{exchange_name}' is not available: {str(e)}")
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"HTTP error when accessing {exchange_name}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error fetching data from {exchange_name}: {str(e)}")
