from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class PresetInfo(BaseModel):
    name: str
    description: str
    method: Literal["GET", "POST"]
    path: str


class PresetDetail(PresetInfo):
    default_body: Optional[Dict[str, Any]] = None
    default_params: Optional[Dict[str, Any]] = None


class PresetExecuteRequest(BaseModel):
    name: str = Field(description="Preset name, e.g., 'summary_default_btc_1h_4h'")
    overrides_body: Optional[Dict[str, Any]] = Field(
        default=None, description="Partial overrides merged into default body"
    )
    overrides_params: Optional[Dict[str, Any]] = Field(
        default=None, description="Partial overrides merged into default params"
    )


SignalSide = Literal["long", "short", "neutral"]


class Detail(BaseModel):
    method: str
    signal: SignalSide
    confidence: float = Field(ge=0, le=1)


class Block(BaseModel):
    summary: SignalSide
    confidence: float = Field(ge=0, le=1)
    details: List[Detail]


class SummarySignalRequest(BaseModel):
    symbol: str
    exchange: str
    resolutions: Dict[str, str]
    indicators: Dict[str, Any]
    limit: int = 200


class SummarySignalResponse(BaseModel):
    short_term: Block
    long_term: Block

    model_config = {
        "json_schema_extra": {
            "example": {
                "short_term": {
                    "summary": "long",
                    "confidence": 0.8,
                    "details": [
                        {"method": "RSI", "signal": "long", "confidence": 0.9},
                        {"method": "MACD", "signal": "neutral", "confidence": 0.5},
                    ],
                },
                "long_term": {
                    "summary": "short",
                    "confidence": 0.6,
                    "details": [
                        {"method": "RSI", "signal": "short", "confidence": 0.7},
                        {"method": "Bollinger", "signal": "short", "confidence": 0.5},
                    ],
                },
            }
        }
    }


class ComputeIndicatorsRequest(BaseModel):
    symbol: str
    exchange: str
    resolution: str
    indicators: Dict[str, Any]
    limit: int = 200


class MACDSeries(BaseModel):
    macd_line: List[Optional[float]]
    signal_line: List[Optional[float]]
    histogram: List[Optional[float]]


class BollingerSeries(BaseModel):
    upper: List[Optional[float]]
    mid: List[Optional[float]]
    lower: List[Optional[float]]


class ComputeIndicatorsResponse(BaseModel):
    symbol: str
    exchange: str
    indicators: Dict[str, Any]

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "BTC/USDT",
                "exchange": "binanceus",
                "indicators": {
                    "macd": {
                        "macd_line": [0.1, 0.2],
                        "signal_line": [0.05, 0.1],
                        "histogram": [0.05, 0.1],
                    },
                    "bollinger": {
                        "upper": [100.0, 101.0],
                        "mid": [99.0, 100.0],
                        "lower": [98.0, 99.0],
                    },
                },
            }
        }
    }


class RunStrategyRequest(BaseModel):
    symbol: Optional[str] = None
    exchange: Optional[str] = None
    resolution: str = "1h"
    limit: int = 500
    config: Dict[str, Any]
    filepath: Optional[str] = None


class StrategyRow(BaseModel):
    Date: datetime
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    Signal: SignalSide
    EMA_short: Optional[float] = None
    EMA_long: Optional[float] = None
    Crossover: Optional[Literal["bullish", "bearish", "none"]] = "none"


class RunStrategyResponse(BaseModel):
    rows: List[StrategyRow]

    model_config = {
        "json_schema_extra": {
            "example": {
                "rows": [
                    {
                        "Date": "2024-01-01T00:00:00Z",
                        "Open": 100.0,
                        "High": 105.0,
                        "Low": 99.0,
                        "Close": 102.0,
                        "Volume": 1234.5,
                        "Signal": "long",
                        "EMA_short": 101.0,
                        "EMA_long": 100.5,
                        "Crossover": "bullish",
                    },
                    {
                        "Date": "2024-01-01T01:00:00Z",
                        "Open": 102.0,
                        "High": 106.0,
                        "Low": 101.0,
                        "Close": 104.0,
                        "Volume": 1300.0,
                        "Signal": "neutral",
                        "EMA_short": 102.0,
                        "EMA_long": 101.0,
                        "Crossover": "none",
                    },
                ]
            }
        }
    }


class OhlcvRow(BaseModel):
    Date: datetime
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float


class OhlcvResponse(BaseModel):
    rows: List[OhlcvRow]

    model_config = {
        "json_schema_extra": {
            "example": {
                "rows": [
                    {
                        "Date": "2024-01-01T00:00:00Z",
                        "Open": 100.0,
                        "High": 105.0,
                        "Low": 99.0,
                        "Close": 102.0,
                        "Volume": 1234.5,
                    },
                    {
                        "Date": "2024-01-01T01:00:00Z",
                        "Open": 102.0,
                        "High": 106.0,
                        "Low": 101.0,
                        "Close": 104.0,
                        "Volume": 1300.0,
                    },
                ]
            }
        }
    }
