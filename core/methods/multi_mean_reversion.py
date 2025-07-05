"""
Multi‑timeframe mean‑reversion module.

Hierarchy:
    • Trend   : Daily   – direction via SMA‑100
    • Setup   : Hourly  – oversold / overbought via RSI‑14
    • Trigger : 15‑min  – mean‑reversion entry using the existing single‑TF
                           logic from core.methods.mean_reversion.analyze
"""

from __future__ import annotations
import pandas as pd
import numpy as np

# Re‑use the single‑timeframe trigger
from core.methods.mean_reversion import analyze as trigger_analyze

# Indicator helpers
from core.indicators.moving_averages import sma
from core.indicators.rsi import rsi
from core.indicators.bbands import bollinger


# ---------------------------------------------------------------------------
# Daily‑level TREND
# ---------------------------------------------------------------------------


def trend_analyze(df: pd.DataFrame, sma_window: int = 100) -> pd.DataFrame:
    """
    Adds:
        • SMA_{window}
        • trend  (+1 if Close > SMA, else ‑1)

    Args:
        df: Daily dataframe.
        sma_window: Look‑back for defining trend.

    Returns:
        Same dataframe with trend columns.
    """
    df = sma(df, window=sma_window)
    sma_col = f"SMA_{sma_window}"
    df["trend"] = (df["Close"] > df[sma_col]).astype(int).replace({0: -1})
    return df


# ---------------------------------------------------------------------------
# Hourly‑level SETUP
# ---------------------------------------------------------------------------


def zone_analyze(
    df: pd.DataFrame,
    rsi_window: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    bb_window: int = 20,
    bb_sigma: float = 2.0,
) -> pd.DataFrame:
    """
    Adds:
        • RSI_{window}
        • BB_M|U|L_{bb_window}
        • zone : {'oversold', 'overbought', 'neutral'}

    Oversold  = Close < lower Bollinger band AND RSI below rsi_oversold.
    Overbought = Close > upper Bollinger band AND RSI above rsi_overbought.
    """
    # --- RSI ---
    df = rsi(df, window=rsi_window)
    rsi_col = f"RSI_{rsi_window}"

    # --- Bollinger Bands ---
    df = bollinger(df, window=bb_window, n_sigma=bb_sigma)
    lower_band = f"BB_L_{bb_window}"
    upper_band = f"BB_U_{bb_window}"

    # Define oversold / overbought using BOTH RSI and Bollinger extremes
    oversold_cond = (df["Close"] < df[lower_band]) & (df[rsi_col] < rsi_oversold)
    overbought_cond = (df["Close"] > df[upper_band]) & (df[rsi_col] > rsi_overbought)

    df["zone"] = np.select(
        [oversold_cond, overbought_cond],
        ["oversold", "overbought"],
        default="neutral",
    )
    return df


# ---------------------------------------------------------------------------
# 15‑min TRIGGER  (re‑uses trigger_analyze)
# ---------------------------------------------------------------------------


def trigger(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper for the existing mean‑reversion analyze() on 15‑minute bars.
    """
    return trigger_analyze(df)


# ---------------------------------------------------------------------------
# Cross‑timeframe FILTER
# ---------------------------------------------------------------------------


def multi_tf_filter(
    m15_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep 15‑min signals only when:

        • direction matches daily trend  (uptrend → long only, downtrend → short only)
        • timestamp falls within an hourly oversold/overbought zone matching the signal

    Signal mapping:
        +1 (long)  valid in daily up‑trend and hourly 'oversold'
        -1 (short) valid in daily down‑trend and hourly 'overbought'

    Returns:
        Filtered 15‑minute dataframe with invalid signals set to 0.
    """
    # Forward‑fill daily & hourly context to 15‑minute index
    daily_trend = daily_df["trend"].reindex(m15_df.index, method="ffill")
    hourly_zone = hourly_df["zone"].reindex(m15_df.index, method="ffill")

    # Validity masks
    long_ok = (daily_trend == 1) & (hourly_zone == "oversold")
    short_ok = (daily_trend == -1) & (hourly_zone == "overbought")

    valid_mask = ((m15_df["signal"] == 1) & long_ok) | (
        (m15_df["signal"] == -1) & short_ok
    )

    m15_df.loc[~valid_mask, "signal"] = 0
    m15_df.loc[~valid_mask, "reason"] = ""  # clear reasons for filtered signals
    return m15_df