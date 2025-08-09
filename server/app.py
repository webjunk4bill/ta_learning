from fastapi import FastAPI, Depends, Security, HTTPException, status
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
import pandas as pd
from loguru import logger

_ENV_FILE = ".env"

# --- Load config and initialize logger ---
import yaml
from core.logger import init_logger

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

debug = config.get("general", {}).get("debug", False)
init_logger(debug=debug)
print(f"DEBUG MODE: {debug}")

# --- Helper: Robust OHLCV normalization ---
def normalize_prices_df(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Normalize a raw OHLCV DataFrame into columns [Date, Open, High, Low, Close, Volume] with UTC Date index.
    Handles common variations for timestamp and column casing. Raises a clear error if essentials are missing.
    """
    try:
        if debug:
            logger.debug(f"[normalize] context={context} incoming_cols={list(df.columns)}")

        # Build a mapping of lowercase name -> original column name
        lower_map = {c.lower(): c for c in df.columns}

        # Detect timestamp column
        ts_col = None
        for cand in ("date", "datetime", "timestamp"):
            if cand in lower_map:
                ts_col = lower_map[cand]
                break
        if ts_col is None:
            raise ValueError("Missing timestamp column: expected one of ['date','datetime','timestamp']")

        ts = df[ts_col]
        # Convert to UTC datetime
        if pd.api.types.is_integer_dtype(ts) or pd.api.types.is_float_dtype(ts):
            # Heuristic: ms vs s
            sample = float(ts.iloc[-1]) if len(ts) else 0.0
            unit = "ms" if sample > 1e12 else "s"
            dt = pd.to_datetime(ts, unit=unit, utc=True)
        else:
            dt = pd.to_datetime(ts, utc=True, errors="coerce")

        if dt.isna().all():
            raise ValueError("Timestamp conversion failed: all NaT after parsing")

        # Detect price/volume columns
        def need(name: str) -> str:
            if name in lower_map:
                return lower_map[name]
            raise ValueError(f"Missing required column: {name}")

        o_col = need("open")
        h_col = need("high")
        l_col = need("low")
        c_col = need("close")
        v_col = need("volume")

        out = pd.DataFrame({
            "Date": dt,
            "Open": pd.to_numeric(df[o_col], errors="coerce"),
            "High": pd.to_numeric(df[h_col], errors="coerce"),
            "Low": pd.to_numeric(df[l_col], errors="coerce"),
            "Close": pd.to_numeric(df[c_col], errors="coerce"),
            "Volume": pd.to_numeric(df[v_col], errors="coerce"),
        })
        out = out.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

        if debug:
            logger.debug(f"[normalize] context={context} mapped={{'open': o_col, 'high': h_col, 'low': l_col, 'close': c_col, 'volume': v_col, 'ts': ts_col}} rows={len(out)} range=({out.index.min()}, {out.index.max()})")

        return out
    except Exception as e:
        logger.error(f"[normalize] context={context} failed: {e}")
        raise

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


def verify_api_key(x_api_key: str = Security(api_key_header)):
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
    """
    Run a strategy on live OHLCV (via CCXT) or a local CSV (if filepath is provided). Returns per-candle outputs based on the supplied strategy config.
    """
    try:
        if req.symbol and req.exchange:
            live_df = fetch_ohlcv(
                req.symbol,
                req.exchange,
                timeframe=req.resolution or "1h",
                limit=req.limit or 200,
            )
            df = normalize_prices_df(live_df, context="run_strategy")
        else:
            df = load_price_data(req.filepath)

        result = run_strategy(df, req.config)
        return result.to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}


@app.post("/summary_signal", dependencies=[Depends(verify_api_key)])
def summary_signal(req: SummarySignalRequest):
    """
    Analyze multiple timeframes and indicators to produce short_term and long_term signal summaries with confidence scores and per-indicator details.
    """
    try:
        results = {}
        for name, tf in req.resolutions.items():
            live_df = fetch_ohlcv(
                req.symbol,
                req.exchange,
                timeframe=tf,
                limit=req.limit or 200,
            )
            df = normalize_prices_df(live_df, context=f"summary_signal:{name}")
            results[name] = timeframe_summary(df, req.indicators, debug=debug)

        return results
    except Exception as e:
        return {"error": str(e)}


@app.get("/news", dependencies=[Depends(verify_api_key)])
def get_news():
    """
    Fetch recent crypto headlines from CryptoPanic (up to 20). Each item includes source, title, url, published_at, and tags.
    """
    try:
        return fetch_latest_news(20)
    except Exception as e:
        return {"error": str(e)}


@app.post("/compute_indicators", dependencies=[Depends(verify_api_key)])
def compute_indicators_api(req: IndicatorRequest):
    """
    Compute technical indicators (e.g., RSI, MACD, Bollinger) over live OHLCV for the requested symbol/exchange/timeframe. NaN values are serialized as null.
    """
    try:
        live_df = fetch_ohlcv(
            req.symbol,
            req.exchange,
            timeframe=req.resolution or "1h",
            limit=req.limit or 200,
        )
        df = normalize_prices_df(live_df, context="compute_indicators")

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
    """
    Health check endpoint used by uptime monitors and Render health checks.
    """
    return {"status": "ok"}


from datetime import datetime
import subprocess
from fastapi.openapi.utils import get_openapi


@app.get("/version")
def version():
    """
    Return build metadata including a UTC timestamp and short git commit hash, if available.
    """
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        )
    except Exception:
        sha = "unknown"
    return {"version": datetime.utcnow().isoformat() + "Z", "commit": sha}


@app.get("/ohlcv", dependencies=[Depends(verify_api_key)])
def ohlcv(symbol: str, exchange: str, resolution: str = "1h", limit: int = 200):
    """
    Fetch recent OHLCV candles from the specified exchange via CCXT and return the latest rows as an array of records.
    """
    try:
        # Pass strict=False to fetch_ohlcv for this endpoint
        raw = fetch_ohlcv(symbol, exchange, timeframe=resolution, limit=limit, strict=False)
        df = normalize_prices_df(raw, context="ohlcv")
        return df.tail(limit).to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}



# Custom OpenAPI schema with public Render URL and API key security
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="TA Learning API",
        version="1.0.0",
        description="Signals, indicators, OHLCV, and news endpoints for TA Learning.",
        routes=app.routes,
    )
    # Set public base URL so tools know where to send requests
    schema["servers"] = [{"url": "https://ta-learning.onrender.com"}]
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi
