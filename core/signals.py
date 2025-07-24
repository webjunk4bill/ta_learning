import pandas as pd
from typing import Dict, Any, List, Tuple

from .indicators import rsi, macd, bollinger


Signal = Dict[str, Any]


def _rsi_signal(df: pd.DataFrame, window: int = 14, debug: bool = False) -> Signal:
    df = rsi(df.copy(), window=window)
    value = df[f"RSI_{window}"].iloc[-1]
    if value < 30:
        signal = "long"
        confidence = min((30 - value) / 30, 1.0)
    elif value > 70:
        signal = "short"
        confidence = min((value - 70) / 30, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5
    if debug:
        print(f"RSI={value:.2f} → Signal={signal}, Confidence={confidence:.2f}")
    return {"method": "RSI", "signal": signal, "confidence": round(float(confidence), 2)}


def _macd_signal(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, debug: bool = False) -> Signal:
    df = macd(df.copy(), fast=fast, slow=slow, signal=signal)
    hist = df["MACD_hist"].iloc[-1]
    recent = df["MACD_hist"].tail(20)
    max_recent = max(abs(recent.max()), abs(recent.min()), 1e-6)
    if hist > 0:
        sig = "long"
        confidence = min(abs(hist) / max_recent, 1.0)
    elif hist < 0:
        sig = "short"
        confidence = min(abs(hist) / max_recent, 1.0)
    else:
        sig = "neutral"
        confidence = 0.0
    if debug:
        print(f"MACD histogram={hist:.5f} → Signal={sig}, Confidence={confidence:.2f}")
    return {"method": "MACD", "signal": sig, "confidence": round(float(confidence), 2)}


def _bollinger_signal(df: pd.DataFrame, window: int = 20, std_dev: float = 2.0, debug: bool = False) -> Signal:
    df = bollinger(df.copy(), window=window, n_sigma=std_dev)
    close = df["Close"].iloc[-1]
    upper = df[f"BB_U_{window}"].iloc[-1]
    lower = df[f"BB_L_{window}"].iloc[-1]
    width = upper - lower if upper != lower else 1
    if close < lower:
        sig = "long"
        confidence = min((lower - close) / width, 1.0)
    elif close > upper:
        sig = "short"
        confidence = min((close - upper) / width, 1.0)
    else:
        sig = "neutral"
        mid = (upper + lower) / 2
        confidence = max(1 - abs(close - mid) / (width / 2), 0.4)
        confidence = min(confidence, 0.6)
    if debug:
        print(f"Close={close:.2f}, Lower={lower:.2f}, Upper={upper:.2f} → Signal={sig}, Confidence={confidence:.2f}")
    return {"method": "Bollinger", "signal": sig, "confidence": round(float(confidence), 2)}


def indicator_signals(df: pd.DataFrame, config: Dict[str, Any], debug: bool = False) -> List[Signal]:
    """Compute signals for each indicator defined in config."""
    results: List[Signal] = []
    if "rsi" in config:
        window = config["rsi"].get("window", 14)
        results.append(_rsi_signal(df, window=window, debug=debug))
    if "macd" in config:
        mc = config["macd"]
        results.append(
            _macd_signal(
                df,
                fast=mc.get("fast", 12),
                slow=mc.get("slow", 26),
                signal=mc.get("signal", 9),
                debug=debug,
            )
        )
    if "bollinger" in config:
        bc = config["bollinger"]
        results.append(
            _bollinger_signal(
                df,
                window=bc.get("window", 20),
                std_dev=bc.get("std_dev", 2.0),
                debug=debug,
            )
        )
    return results


def _summarize(details: List[Signal]) -> Tuple[str, float]:
    """Return overall signal and confidence from individual indicator signals."""
    if not details:
        return "neutral", 0.0

    sign_map = {"long": 1, "short": -1}
    directional = [d for d in details if d["signal"] in sign_map]
    if not directional:
        return "neutral", 0.5

    weighted = sum(sign_map[d["signal"]] * d["confidence"] for d in directional)
    total_conf = sum(d["confidence"] for d in directional)

    score = weighted / total_conf
    if score > 0.1:
        summary = "long"
    elif score < -0.1:
        summary = "short"
    else:
        summary = "neutral"
    return summary, round(abs(score), 2)


def timeframe_summary(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    details = indicator_signals(df, config)
    summary, conf = _summarize(details)
    return {"summary": summary, "confidence": conf, "details": details}
