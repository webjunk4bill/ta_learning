from typing import Any, Dict, Literal, Optional, Callable, Tuple

Method = Literal["GET", "POST"]


class Preset:
    def __init__(
        self,
        name: str,
        description: str,
        method: Method,
        path: str,
        default_body: Optional[Dict[str, Any]] = None,
        default_params: Optional[Dict[str, Any]] = None,
        execute_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        self.name = name
        self.description = description
        self.method = method
        self.path = path
        self.default_body = default_body or {}
        self.default_params = default_params or {}
        self.execute_fn = execute_fn


PRESETS: Dict[str, Preset] = {}


def _add(p: Preset) -> None:
    PRESETS[p.name] = p


_add(
    Preset(
        name="summary_default_btc_1h_4h",
        description="Multi-timeframe summary (1h vs 4h) for BTC on Binance.US with default RSI/MACD/Bollinger settings.",
        method="POST",
        path="/summary_signal",
        default_body={
            "symbol": "BTC/USDT",
            "exchange": "binanceus",
            "resolutions": {"short_term": "1h", "long_term": "4h"},
            "indicators": {
                "rsi": {"window": 14},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "bollinger": {"window": 20, "std_dev": 2},
            },
            "limit": 200,
        },
    )
)

_add(
    Preset(
        name="indicators_btc_1h_default",
        description="Compute RSI(14), MACD(12,26,9), Bollinger(20,2) for BTC 1h on Binance.US.",
        method="POST",
        path="/compute_indicators",
        default_body={
            "symbol": "BTC/USDT",
            "exchange": "binanceus",
            "resolution": "1h",
            "indicators": {
                "rsi": {"window": 14},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "bollinger": {"window": 20, "std_dev": 2},
            },
            "limit": 200,
        },
    )
)

_add(
    Preset(
        name="ema_crossover_btc_1h",
        description="EMA crossover (12/26) strategy for BTC 1h on Binance.US.",
        method="POST",
        path="/run_strategy",
        default_body={
            "symbol": "BTC/USDT",
            "exchange": "binanceus",
            "resolution": "1h",
            "limit": 500,
            "config": {"strategy": "ema_crossover", "short_window": 12, "long_window": 26},
        },
    )
)

_add(
    Preset(
        name="ohlcv_btc_1h_200",
        description="Fetch last 200 BTC/USDT 1h candles from Binance.US.",
        method="GET",
        path="/ohlcv",
        default_params={
            "symbol": "BTC/USDT",
            "exchange": "binanceus",
            "resolution": "1h",
            "limit": 200,
        },
    )
)

_add(
    Preset(
        name="news_default",
        description="Fetch recent crypto headlines (CryptoPanic).",
        method="GET",
        path="/news",
    )
)

