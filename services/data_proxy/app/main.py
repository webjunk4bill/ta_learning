from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os, time
import ccxt
import pandas_ta as ta

app = FastAPI(title="TA Data Proxy", version="0.1.0")

EXCHANGES = {}

def get_exchange(name: str):
    name = name.lower()
    if name not in EXCHANGES:
        if name == "binanceus":
            EXCHANGES[name] = ccxt.binanceus()
        elif name == "kraken":
            EXCHANGES[name] = ccxt.kraken()
        else:
            raise HTTPException(400, f"Unsupported exchange '{name}'")
    return EXCHANGES[name]

def ohlcv_df(exchange: str, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    ex = get_exchange(exchange)
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not data:
        raise HTTPException(404, "No OHLCV returned")
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

@app.get("/health")
def health():
    return {"ok": True, "time": int(time.time())}

@app.get("/ohlcv")
def get_ohlcv(symbol: str, exchange: str, timeframe: str, limit: int = 1000):
    df = ohlcv_df(exchange, symbol, timeframe, limit)
    return {"candles": df.to_dict(orient="records")}

class IndicatorInputs(BaseModel):
    symbol: str
    exchange: str
    timeframe: str
    inputs: dict = {}

@app.post("/indicators")
def post_indicators(body: IndicatorInputs):
    df = ohlcv_df(body.exchange, body.symbol, body.timeframe, 2000)
    out = {}
    inp = body.inputs or {}

    # RSI
    if "rsi" in inp:
        L = int(inp["rsi"].get("length", 14))
        out["rsi"] = float(ta.rsi(df["close"], length=L).iloc[-1])

    # MACD
    if "macd" in inp:
        p = inp["macd"]; f=p.get("fast",12); s=p.get("slow",26); sig=p.get("signal",9)
        macd = ta.macd(df["close"], fast=f, slow=s, signal=sig).iloc[-1].to_dict()
        out["macd"] = {
            "macd": float(macd.get(f"MACD_{f}_{s}_{sig}")),
            "signal": float(macd.get(f"MACDs_{f}_{s}_{sig}")),
            "hist": float(macd.get(f"MACDh_{f}_{s}_{sig}"))
        }

    # Bollinger Bands
    if "bb" in inp:
        l = inp["bb"].get("length",20); sd = inp["bb"].get("stdev",2)
        bb = ta.bbands(df["close"], length=l, std=sd).iloc[-1].to_dict()
        out["bb"] = {
            "lower": float(bb.get(f"BBL_{l}_{sd}.0")),
            "basis": float(bb.get(f"BBM_{l}_{sd}.0")),
            "upper": float(bb.get(f"BBU_{l}_{sd}.0"))
        }

    # EMA (array)
    if "ema" in inp:
        out["ema"] = {str(n): float(ta.ema(df["close"], length=int(n)).iloc[-1]) for n in inp["ema"]}

    # ATR
    if "atr" in inp:
        L = int(inp["atr"].get("length",14))
        out["atr"] = float(ta.atr(df["high"], df["low"], df["close"], length=L).iloc[-1])

    # ADX
    if "adx" in inp:
        L = int(inp["adx"].get("length",14))
        out["adx"] = float(ta.adx(df["high"], df["low"], df["close"], length=L)[f"ADX_{L}"].iloc[-1])

    return {"as_of": df["ts"].iloc[-1].isoformat(), "values": out}

@app.get("/cryptopanic")
def get_cryptopanic(asset: str, lookback_days: int = 3):
    # TODO: wire to existing CryptoPanic integration
    return {"items": []}

@app.get("/reddit/search")
def reddit_search(subreddits: str, query: str = "", lookback_hours: int = 48, limit: int = 100):
    # TODO: wire to Reddit worker/API
    return {"posts": [], "comments": []}
