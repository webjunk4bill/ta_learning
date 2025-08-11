from fastapi import FastAPI, Depends, Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from typing import List
import os
from core.news import fetch_latest_news
from ta_engine.data import load_price_data
from ta_engine.live_data import fetch_ohlcv
from ta_engine.strategies.registry import get_strategy
from core.indicators import compute_indicators
from core.signals import timeframe_summary
import numpy as np
import pandas as pd
from loguru import logger
from .presets import PRESETS, Preset
from .models import (
    SummarySignalRequest,
    SummarySignalResponse,
    ComputeIndicatorsRequest,
    ComputeIndicatorsResponse,
    RunStrategyRequest,
    RunStrategyResponse,
    StrategyRow,
    OhlcvResponse,
    OhlcvRow,
    NewsResponse,
    PresetInfo,
    PresetDetail,
    PresetExecuteRequest,
)

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


def replace_nans(obj):
    if isinstance(obj, dict):
        return {k: replace_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_nans(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and np.isnan(obj):
        return None
    return obj


@app.post(
    "/run_strategy",
    dependencies=[Depends(verify_api_key)],
    response_model=RunStrategyResponse,
)
def run_strategy_api(req: RunStrategyRequest):
    """Run a strategy and return per-candle outputs."""
    try:
        if req.symbol and req.exchange:
            live_df = fetch_ohlcv(
                req.symbol,
                req.exchange,
                timeframe=req.resolution or "1h",
                limit=req.limit or 500,
            )
            df = normalize_prices_df(live_df, context="run_strategy")
        else:
            df = load_price_data(req.filepath)

        strat = get_strategy(req.config.get("strategy"))
        result_df = strat.run(df, req.config)
        merged = df.join(result_df, how="left").drop(columns=["Price"], errors="ignore")
        rows = merged.reset_index().to_dict(orient="records")
        rows = replace_nans(rows)
        return {"rows": rows}

    except Exception as e:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post(
    "/summary_signal",
    dependencies=[Depends(verify_api_key)],
    response_model=SummarySignalResponse,
)
def summary_signal(req: SummarySignalRequest):
    """Analyze multiple timeframes to produce short and long term signal summaries."""
    try:
        results = {}
        for name in ("short_term", "long_term"):
            tf = req.resolutions.get(name)
            if tf:
                live_df = fetch_ohlcv(
                    req.symbol,
                    req.exchange,
                    timeframe=tf,
                    limit=req.limit or 200,
                )
                df = normalize_prices_df(live_df, context=f"summary_signal:{name}")
                results[name] = timeframe_summary(df, req.indicators, debug=debug)
        return replace_nans(results)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get(
    "/news",
    response_model=NewsResponse,
    dependencies=[Depends(verify_api_key)],
)
def get_news():
    """Fetch recent crypto headlines from CryptoPanic."""
    try:
        news_list = fetch_latest_news(20)
        items = [
            {
                "source": i["source"],
                "title": i["title"],
                "url": i["url"],
                "published_at": i["published_at"],
                "tags": i.get("tags", []),
            }
            for i in news_list
        ]
        return {"items": items}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post(
    "/compute_indicators",
    dependencies=[Depends(verify_api_key)],
    response_model=ComputeIndicatorsResponse,
)
def compute_indicators_api(req: ComputeIndicatorsRequest):
    """Compute technical indicators over live OHLCV data."""
    try:
        live_df = fetch_ohlcv(
            req.symbol,
            req.exchange,
            timeframe=req.resolution or "1h",
            limit=req.limit or 200,
        )
        df = normalize_prices_df(live_df, context="compute_indicators")

        indicators = compute_indicators(df, req.indicators)
        safe_indicators = replace_nans(indicators)

        return {
            "symbol": req.symbol,
            "exchange": req.exchange,
            "indicators": safe_indicators,
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# Preset routes
@app.get("/presets", response_model=List[PresetInfo], dependencies=[Depends(verify_api_key)])
def list_presets():
    """List available presets with brief descriptions and target endpoints."""
    return [
        PresetInfo(name=p.name, description=p.description, method=p.method, path=p.path)
        for p in PRESETS.values()
    ]


@app.get(
    "/presets/{name}",
    response_model=PresetDetail,
    dependencies=[Depends(verify_api_key)],
)
def get_preset(name: str):
    """Return details for a specific preset, including default payload/params."""
    p = PRESETS.get(name)
    if not p:
        return JSONResponse(status_code=404, content={"error": f"Preset '{name}' not found"})
    return PresetDetail(
        name=p.name,
        description=p.description,
        method=p.method,
        path=p.path,
        default_body=p.default_body,
        default_params=p.default_params,
    )


@app.post("/presets/execute", dependencies=[Depends(verify_api_key)])
def execute_preset(req: PresetExecuteRequest):
    """Execute a preset by name. Overrides (if provided) are merged into default body/params."""
    p = PRESETS.get(req.name)
    if not p:
        return JSONResponse(status_code=404, content={"error": f"Preset '{req.name}' not found"})

    body = {**(p.default_body or {})}
    params = {**(p.default_params or {})}
    if req.overrides_body:
        body.update(req.overrides_body)
    if req.overrides_params:
        params.update(req.overrides_params)

    if p.execute_fn:
        try:
            return p.execute_fn({"body": body, "params": params})
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    try:
        if p.method == "GET" and p.path == "/ohlcv":
            df = fetch_ohlcv(
                params.get("symbol", "BTC/USDT"),
                params.get("exchange", "binanceus"),
                timeframe=params.get("resolution", "1h"),
                limit=int(params.get("limit", 200)),
                strict=False,
            )
            norm = normalize_prices_df(df, context=f"preset:{p.name}")
            return norm.tail(int(params.get("limit", 200))).to_dict(orient="records")

        if p.method == "GET" and p.path == "/news":
            return get_news()

        if p.method == "POST" and p.path == "/summary_signal":
            return summary_signal(SummarySignalRequest(**body))

        if p.method == "POST" and p.path == "/compute_indicators":
            return compute_indicators_api(ComputeIndicatorsRequest(**body))

        if p.method == "POST" and p.path == "/run_strategy":
            return run_strategy_api(RunStrategyRequest(**body))

        return JSONResponse(status_code=400, content={"error": f"Unsupported preset path: {p.path}"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


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


@app.get(
    "/ohlcv",
    dependencies=[Depends(verify_api_key)],
    response_model=OhlcvResponse,
)
def ohlcv(symbol: str, exchange: str, resolution: str = "1h", limit: int = 200):
    """Fetch recent OHLCV candles from the specified exchange via CCXT."""
    try:
        raw = fetch_ohlcv(symbol, exchange, timeframe=resolution, limit=limit, strict=False)
        df = normalize_prices_df(raw, context="ohlcv")
        rows = df.tail(limit).reset_index().to_dict(orient="records")
        rows = replace_nans(rows)
        return {"rows": rows}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})



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
