from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import os
from core.news import fetch_latest_news
from ta_engine.data import load_price_data
from ta_engine.live_data import fetch_ohlcv
from ta_engine.strategies import run_strategy
from core.indicators import compute_indicators
from core.signals import timeframe_summary
import numpy as np

_ENV_FILE = ".env"

# --- Load config and initialize logger ---
import yaml
from core.logger import init_logger

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

debug = config.get("general", {}).get("debug", False)
init_logger(debug=debug)
print(f"DEBUG MODE: {debug}")

def _read_env_var(key: str) -> str | None:
    """Read a single variable from the local .env file."""
    if os.path.exists(_ENV_FILE):
        with open(_ENV_FILE) as f:
            for line in f:
                if line.strip().startswith(f"{key}="):
                    return line.strip().split("=", 1)[1]
    return None


# Load API key once at startup
API_KEY = (
    os.getenv("API_KEY")
    or os.getenv("GPT_API_KEY")
    or _read_env_var("API_KEY")
    or config.get("security", {}).get("api_key", "")
)


api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def verify_api_key(x_api_key: str = Depends(api_key_header)):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )

        
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


@app.post("/run_strategy", dependencies=[Depends(verify_api_key)])
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


@app.post("/summary_signal", dependencies=[Depends(verify_api_key)])
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


@app.get("/news", dependencies=[Depends(verify_api_key)])
def get_news():
    """Return recent news headlines from CryptoPanic."""
    try:
        return fetch_latest_news(20)
    except Exception as e:
        return {"error": str(e)}


@app.post("/compute_indicators", dependencies=[Depends(verify_api_key)])
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


# Health check endpoint
@app.get("/ping")
def ping():
    return {"status": "ok"}


from datetime import datetime
import subprocess


@app.get("/version")
def version():
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        )
    except Exception:
        sha = "unknown"
    return {"version": datetime.utcnow().isoformat() + "Z", "commit": sha}


@app.get("/ohlcv", dependencies=[Depends(verify_api_key)])
def ohlcv(symbol: str, exchange: str, resolution: str = "1h", limit: int = 200):
    df = fetch_ohlcv(symbol, exchange, timeframe=resolution, limit=limit)
    return df.tail(limit).to_dict(orient="records")
