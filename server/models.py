from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from datetime import datetime

from pydantic import BaseModel, Field

SignalSide = Literal["long", "short", "neutral"]


# ---------- Summary Signal ----------

class Detail(BaseModel):
    method: str = Field(..., example="MACD")
    signal: SignalSide = Field(..., example="long")
    confidence: float = Field(..., ge=0, le=1, example=0.78)


class Block(BaseModel):
    summary: SignalSide = Field(..., example="long")
    confidence: float = Field(..., ge=0, le=1, example=0.76)
    details: List[Detail]


class SummarySignalRequest(BaseModel):
    symbol: str = Field(..., example="BTC/USDT")
    exchange: str = Field(..., example="binanceus")
    resolutions: Dict[str, str] = Field(
        ..., example={"short_term": "1h", "long_term": "4h"}
    )
    indicators: Dict[str, Any] = Field(
        ...,
        example={
            "rsi": {"window": 14},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger": {"window": 20, "std_dev": 2},
        },
    )
    limit: int = Field(200, example=200)


class SummarySignalResponse(BaseModel):
    short_term: Block
    long_term: Block

    model_config = {
        "json_schema_extra": {
            "example": {
                "short_term": {
                    "summary": "long",
                    "confidence": 0.78,
                    "details": [
                        {"method": "MACD", "signal": "long", "confidence": 0.82},
                        {"method": "RSI", "signal": "neutral", "confidence": 0.50},
                        {"method": "Bollinger", "signal": "neutral", "confidence": 0.60},
                    ],
                },
                "long_term": {
                    "summary": "neutral",
                    "confidence": 0.52,
                    "details": [
                        {"method": "MACD", "signal": "neutral", "confidence": 0.55},
                        {"method": "RSI", "signal": "neutral", "confidence": 0.50},
                        {"method": "Bollinger", "signal": "short", "confidence": 0.58},
                    ],
                },
            }
        }
    }


# ---------- Compute Indicators ----------

class MACDSeries(BaseModel):
    macd_line: List[Optional[float]]
    signal_line: List[Optional[float]]
    histogram: List[Optional[float]]


class BollingerSeries(BaseModel):
    upper: List[Optional[float]]
    mid: List[Optional[float]]
    lower: List[Optional[float]]


class ComputeIndicatorsRequest(BaseModel):
    symbol: str = Field(..., example="BTC/USDT")
    exchange: str = Field(..., example="binanceus")
    resolution: str = Field(..., example="1h")
    indicators: Dict[str, Any] = Field(
        ...,
        example={
            "rsi": {"window": 14},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger": {"window": 20, "std_dev": 2},
        },
    )
    limit: int = Field(200, example=200)


class ComputeIndicatorsResponse(BaseModel):
    symbol: str = Field(..., example="BTC/USDT")
    exchange: str = Field(..., example="binanceus")
    indicators: Dict[str, Any]

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "BTC/USDT",
                "exchange": "binanceus",
                "indicators": {
                    "rsi": [None, None, 32.1, 45.3, 51.8],
                    "macd": {
                        "macd_line": [None, -22.5, -10.1, 3.4, 9.2],
                        "signal_line": [None, -18.2, -8.0, 1.2, 6.5],
                        "histogram": [None, -4.3, -2.1, 2.2, 2.7],
                    },
                    "bollinger": {
                        "upper": [None, None, 118500.1, 118620.4, 118740.9],
                        "mid": [None, None, 117900.0, 118010.2, 118120.0],
                        "lower": [None, None, 117300.2, 117399.9, 117499.1],
                    },
                },
            }
        }
    }


# ---------- Run Strategy (EMA crossover) ----------

class StrategyRow(BaseModel):
    Date: datetime = Field(..., example="2025-07-23T12:00:00Z")
    Open: float = Field(..., example=118512.24)
    High: float = Field(..., example=118653.81)
    Low: float = Field(..., example=117775.02)
    Close: float = Field(..., example=117782.88)
    Volume: float = Field(..., example=0.89481)
    Signal: SignalSide = Field(..., example="long")
    EMA_short: Optional[float] = Field(None, example=118120.5)
    EMA_long: Optional[float] = Field(None, example=118040.2)
    Crossover: Optional[Literal["bullish", "bearish", "none"]] = Field(
        "none", example="bullish"
    )


class RunStrategyRequest(BaseModel):
    symbol: Optional[str] = Field(None, example="BTC/USDT")
    exchange: Optional[str] = Field(None, example="binanceus")
    resolution: str = Field("1h", example="1h")
    limit: int = Field(500, example=500)
    config: Dict[str, Any] = Field(
        ..., example={"strategy": "ema_crossover", "short_window": 12, "long_window": 26}
    )
    filepath: Optional[str] = Field(None, example="/path/to/local.csv")


class RunStrategyResponse(BaseModel):
    rows: List[StrategyRow]

    model_config = {
        "json_schema_extra": {
            "example": {
                "rows": [
                    {
                        "Date": "2025-07-23T10:00:00Z",
                        "Open": 118026.49,
                        "High": 118667.70,
                        "Low": 117893.42,
                        "Close": 118427.30,
                        "Volume": 2.21387,
                        "Signal": "neutral",
                        "EMA_short": 118180.1,
                        "EMA_long": 118120.7,
                        "Crossover": "none",
                    },
                    {
                        "Date": "2025-07-23T11:00:00Z",
                        "Open": 118383.81,
                        "High": 118667.66,
                        "Low": 118258.49,
                        "Close": 118508.27,
                        "Volume": 0.36994,
                        "Signal": "long",
                        "EMA_short": 118220.4,
                        "EMA_long": 118140.2,
                        "Crossover": "bullish",
                    },
                    {
                        "Date": "2025-07-23T12:00:00Z",
                        "Open": 118512.24,
                        "High": 118653.81,
                        "Low": 117775.02,
                        "Close": 117782.88,
                        "Volume": 0.89481,
                        "Signal": "short",
                        "EMA_short": 118150.2,
                        "EMA_long": 118160.9,
                        "Crossover": "bearish",
                    },
                ]
            }
        }
    }


# ---------- OHLCV ----------

class OhlcvRow(BaseModel):
    Date: datetime = Field(..., example="2025-07-23T14:00:00Z")
    Open: float = Field(..., example=117400.00)
    High: float = Field(..., example=118045.45)
    Low: float = Field(..., example=117395.17)
    Close: float = Field(..., example=118045.45)
    Volume: float = Field(..., example=0.17734)


class OhlcvResponse(BaseModel):
    rows: List[OhlcvRow]

    model_config = {
        "json_schema_extra": {
            "example": {
                "rows": [
                    {
                        "Date": "2025-07-23T12:00:00Z",
                        "Open": 118512.24,
                        "High": 118653.81,
                        "Low": 117775.02,
                        "Close": 117782.88,
                        "Volume": 0.89481,
                    },
                    {
                        "Date": "2025-07-23T13:00:00Z",
                        "Open": 117929.37,
                        "High": 118191.86,
                        "Low": 117362.00,
                        "Close": 117494.54,
                        "Volume": 1.12609,
                    },
                    {
                        "Date": "2025-07-23T14:00:00Z",
                        "Open": 117400.00,
                        "High": 118045.45,
                        "Low": 117395.17,
                        "Close": 118045.45,
                        "Volume": 0.17734,
                    },
                ]
            }
        }
    }


# ---------- News (CryptoPanic) ----------

class NewsItem(BaseModel):
    source: str = Field(..., example="CoinDesk")
    title: str = Field(..., example="Bitcoin Surges as Spot ETFs See Record Inflows")
    url: str = Field(..., example="https://example.com/article")
    published_at: datetime = Field(..., example="2025-07-23T14:05:00Z")
    tags: List[str] = Field(default_factory=list, example=["BTC", "ETF", "Market"])


class NewsResponse(BaseModel):
    items: List[NewsItem]

    model_config = {
        "json_schema_extra": {
            "example": {
                "items": [
                    {
                        "source": "CoinDesk",
                        "title": "Bitcoin Surges as Spot ETFs See Record Inflows",
                        "url": "https://example.com/article",
                        "published_at": "2025-07-23T14:05:00Z",
                        "tags": ["BTC", "ETF", "Market"],
                    },
                    {
                        "source": "The Block",
                        "title": "Altcoins Follow BTC Higher Amid Broader Risk Rally",
                        "url": "https://example.com/altcoins",
                        "published_at": "2025-07-23T13:45:00Z",
                        "tags": ["ETH", "ALT", "Sentiment"],
                    },
                ]
            }
        }
    }


# ---------- Presets (from Task C) ----------

class PresetInfo(BaseModel):
    name: str = Field(..., example="summary_default_btc_1h_4h")
    description: str = Field(
        ..., example="Multi-timeframe summary (1h vs 4h) for BTC on Binance.US with default RSI/MACD/Bollinger settings."
    )
    method: Literal["GET", "POST"] = Field(..., example="POST")
    path: str = Field(..., example="/summary_signal")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "summary_default_btc_1h_4h",
                "description": "Multi-timeframe summary (1h vs 4h) for BTC on Binance.US with default RSI/MACD/Bollinger settings.",
                "method": "POST",
                "path": "/summary_signal",
            }
        }
    }


class PresetDetail(PresetInfo):
    default_body: Optional[Dict[str, Any]] = Field(
        None,
        example={
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
    default_params: Optional[Dict[str, Any]] = Field(
        None,
        example={
            "symbol": "BTC/USDT",
            "exchange": "binanceus",
            "resolution": "1h",
            "limit": 200,
        },
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "summary_default_btc_1h_4h",
                "description": "Multi-timeframe summary (1h vs 4h) for BTC on Binance.US with default RSI/MACD/Bollinger settings.",
                "method": "POST",
                "path": "/summary_signal",
                "default_body": {
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
                "default_params": {
                    "symbol": "BTC/USDT",
                    "exchange": "binanceus",
                    "resolution": "1h",
                    "limit": 200,
                },
            }
        }
    }


class PresetExecuteRequest(BaseModel):
    name: str = Field(..., example="summary_default_btc_1h_4h")
    overrides_body: Optional[Dict[str, Any]] = Field(
        default=None, example={"exchange": "coinbase", "symbol": "BTC/USD"}
    )
    overrides_params: Optional[Dict[str, Any]] = Field(
        default=None, example={"limit": 50}
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "summary_default_btc_1h_4h",
                "overrides_body": {"exchange": "coinbase", "symbol": "BTC/USD"},
                "overrides_params": {"limit": 50},
            }
        }
    }
