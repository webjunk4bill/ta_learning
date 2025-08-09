import pandas as pd
from core.dataloader import resample_df
from core.methods.mean_reversion import analyze
from core.methods.multi_mean_reversion import (
    trend_analyze,
    zone_analyze,
    trigger,
    multi_tf_filter,
)
from core.backtest import backtest_signals


def run_strategy(df: pd.DataFrame, config: dict, debug: bool = False) -> pd.DataFrame:
    """Run trading strategy based on configuration."""
    if config.get("strategy") == "passthrough_only":
        return df[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()

    gen = config.get("general", {})
    stf = config.get("single_tf", {})
    mtf = config.get("multi_tf", {})

    if gen.get("multi_tf"):
        daily_df = resample_df(df, "1D")
        hourly_df = resample_df(df, "1H")
        m15_df = resample_df(df, "15T")

        daily_df = trend_analyze(daily_df, sma_window=mtf.get("trend_sma_window", 100))
        hourly_df = zone_analyze(
            hourly_df,
            rsi_window=mtf.get("zone_rsi_window", 14),
            rsi_oversold=mtf.get("zone_oversold", 30.0),
            rsi_overbought=mtf.get("zone_overbought", 70.0),
            bb_window=mtf.get("bb_window", 20),
            bb_sigma=mtf.get("bb_sigma", 2.0),
        )
        m15_df = trigger(m15_df)
        m15_df = multi_tf_filter(m15_df, hourly_df, daily_df)
        equity = backtest_signals(m15_df)
        m15_df["Equity"] = equity
        return m15_df

    else:
        results = []
        for tf in stf.get("timeframes", []):
            df_tf = resample_df(df, tf)
            res = analyze(
                df_tf,
                sma_window=stf.get("sma_window", 20),
                rsi_window=stf.get("rsi_window", 14),
                oversold=stf.get("oversold", 30.0),
                overbought=stf.get("overbought", 70.0),
            )
            res["Timeframe"] = tf
            results.append(res)
            print(results)
            if not results:
                raise ValueError("Strategy generated no output — check input data or strategy logic.")
        return pd.concat(results)
