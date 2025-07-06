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
from loguru import logger

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
    logger.info("multi_tf_filter start: {} rows", len(m15_df))

    # Align higher timeframe context to 15-minute index
    hourly_zone = hourly_df["zone"].reindex(m15_df.index, method="ffill")
    trend = daily_df["trend"].reindex(m15_df.index, method="ffill")

    hr_sma_cols = [c for c in hourly_df.columns if c.startswith("SMA_")]
    if not hr_sma_cols:
        hr_sma_cols = [c for c in hourly_df.columns if c.startswith("BB_M_")]
    hr_sma = (
        hourly_df[hr_sma_cols[0]].reindex(m15_df.index, method="ffill")
        if hr_sma_cols
        else pd.Series(np.nan, index=m15_df.index)
    )

    # Prepare columns
    m15_df["entry_signal"] = 0
    m15_df["double_down"] = 0
    m15_df["entry_price"] = np.nan
    m15_df["reason"] = ""

    position = 0
    entry_price = None
    size = 0
    entry_idxs: list[pd.Timestamp] = []
    exit_idxs: list[pd.Timestamp] = []
    threshold = 0.01

    for idx, row in m15_df.iterrows():
        price = row["Close"]
        raw_sig = row["signal"]
        zone = hourly_zone.loc[idx]
        dir_trend = trend.loc[idx]

        valid_entry = 0
        if raw_sig == 1 and dir_trend == 1 and zone == "oversold":
            valid_entry = 1
        elif raw_sig == -1 and dir_trend == -1 and zone == "overbought":
            valid_entry = -1

        # --------------------
        # Exit logic first
        # --------------------
        if position != 0:
            exit_bb = False
            if not pd.isna(hr_sma.loc[idx]):
                exit_bb = price >= hr_sma.loc[idx] if position == 1 else price <= hr_sma.loc[idx]
            exit_profit = price >= entry_price * 1.02 if position == 1 else price <= entry_price * 0.98

            if exit_bb or exit_profit:
                position = 0
                entry_price = None
                size = 0
                exit_idxs.append(idx)
                m15_df.at[idx, "reason"] = "Exit at hourly SMA or ±2% target"

        # --------------------
        # Entry / scaling logic
        # --------------------
        if position == 0:
            if valid_entry != 0:
                position = valid_entry
                entry_price = price
                size = 1
                m15_df.at[idx, "entry_signal"] = position
                entry_idxs.append(idx)
        else:
            if valid_entry == position:
                size += 1
                entry_price = (entry_price * (size - 1) + price) / size
                m15_df.at[idx, "double_down"] = 2 if position == 1 else -2
            else:
                if position == 1 and price < entry_price * (1 - threshold):
                    size += 1
                    entry_price = (entry_price * (size - 1) + price) / size
                    m15_df.at[idx, "double_down"] = 2
                elif position == -1 and price > entry_price * (1 + threshold):
                    size += 1
                    entry_price = (entry_price * (size - 1) + price) / size
                    m15_df.at[idx, "double_down"] = -2

        m15_df.at[idx, "signal"] = position
        if entry_price is not None:
            m15_df.at[idx, "entry_price"] = entry_price

    # Force close any open position at end of data
    if position != 0:
        exit_idxs.append(m15_df.index[-1])
        m15_df.at[m15_df.index[-1], "reason"] = "Exit end-of-data"

    logger.info("Entry signals at indices: {}", entry_idxs)
    logger.info("Exit signals at indices: {}", exit_idxs)
    dd_long_idxs = m15_df.index[m15_df["double_down"] == 2].tolist()
    dd_short_idxs = m15_df.index[m15_df["double_down"] == -2].tolist()
    logger.info("Double-down long at: {}", dd_long_idxs)
    logger.info("Double-down short at: {}", dd_short_idxs)

    return m15_df