from fastapi import FastAPI
from pydantic import BaseModel
from core.news import fetch_latest_news
from ta_engine.data import load_price_data
from ta_engine.live_data import fetch_ohlcv
from ta_engine.strategies import run_strategy
from core.indicators import compute_indicators
from core.signals import timeframe_summary
import numpy as np

# --- Load config and initialize logger ---
import yaml
from core.logger import init_logger

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

debug = config.get("general", {}).get("debug", False)
init_logger(debug=debug)
print(f"DEBUG MODE: {debug}")

app = FastAPI()


class StrategyRequest(BaseModel):
    filepath: str | None = None
    config: dict
    symbol: str | None = None
    exchange: str | None = None
    resolution: str | None = "1h"
    limit: int | None = 200


class IndicatorRequest(BaseModel):
    symbol: str
    exchange: str
    resolution: str = "1h"
    indicators: dict
    limit: int | None = 200


class SummarySignalRequest(BaseModel):
    symbol: str
    exchange: str
    resolutions: dict
    indicators: dict
    limit: int | None = 200


@app.post("/run_strategy")
def run_strategy_api(req: StrategyRequest):
    try:
        if req.symbol and req.exchange:
            live_df = fetch_ohlcv(
                req.symbol,
                req.exchange,
                timeframe=req.resolution or "1h",
                limit=req.limit or 200,
            )
            df = (
                live_df.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                        "date": "Date",
                    }
                ).set_index("Date")
            )
        else:
            df = load_price_data(req.filepath)

        result = run_strategy(df, req.config)
        return result.to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}


@app.post("/summary_signal")
def summary_signal(req: SummarySignalRequest):
    try:
        results = {}
        for name, tf in req.resolutions.items():
            live_df = fetch_ohlcv(
                req.symbol,
                req.exchange,
                timeframe=tf,
                limit=req.limit or 200,
            )
            df = (
                live_df.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                        "date": "Date",
                    }
                ).set_index("Date")
            )
            results[name] = timeframe_summary(df, req.indicators, debug=debug)

        return results
    except Exception as e:
        return {"error": str(e)}


@app.get("/news")
def get_news():
    """Return recent news headlines from CryptoPanic."""
    try:
        return fetch_latest_news(20)
    except Exception as e:
        return {"error": str(e)}


@app.post("/compute_indicators")
def compute_indicators_api(req: IndicatorRequest):
    try:
        live_df = fetch_ohlcv(
            req.symbol,
            req.exchange,
            timeframe=req.resolution or "1h",
            limit=req.limit or 200,
        )
        df = (
            live_df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                    "date": "Date",
                }
            ).set_index("Date")
        )

        indicators = compute_indicators(df, req.indicators)
        
        def replace_nans(obj):
            if isinstance(obj, dict):
                return {k: replace_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_nans(v) for v in obj]
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            return obj

        safe_indicators = replace_nans(indicators)

        return {
            "symbol": req.symbol,
            "exchange": req.exchange,
            "indicators": safe_indicators,
        }

    except Exception as e:
        return {"error": str(e)}
