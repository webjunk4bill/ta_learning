# main.py
import argparse
import yaml
from core.dataloader import load_data, resample_df
from core.methods.mean_reversion import analyze
from core.methods.multi_mean_reversion import (
    trend_analyze,
    zone_analyze,
    trigger,
    multi_tf_filter,
)
from core.visualizer import plot_signals, plot_multi_tf

def parse_args():
    parser = argparse.ArgumentParser(description="Technical Analysis Learning CLI")
    parser.add_argument(
        "--config", "-c",
        help="YAML config file (default: config.yaml)",
        default="config.yaml",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    # Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    gen = cfg["general"]
    stf = cfg["single_tf"]
    mtf = cfg["multi_tf"]

    # Load and optionally filter data
    raw_df = load_data(gen["file"])
    raw_df = raw_df.loc[gen["start_date"] : gen["end_date"]]

    if gen.get("multi_tf"):
        # Multi-timeframe pipeline
        daily_df  = resample_df(raw_df, "1D")
        hourly_df = resample_df(raw_df, "1H")
        m15_df    = resample_df(raw_df, "15T")

        daily_df  = trend_analyze(daily_df, sma_window=mtf["trend_sma_window"])
        hourly_df = zone_analyze(
            hourly_df,
            rsi_window=mtf["zone_rsi_window"],
            rsi_oversold=mtf["zone_oversold"],
            rsi_overbought=mtf["zone_overbought"],
            bb_window=mtf["bb_window"],
            bb_sigma=mtf["bb_sigma"],
        )
        m15_df    = trigger(m15_df)
        m15_df    = multi_tf_filter(m15_df, hourly_df, daily_df)

        plot_multi_tf(daily_df, hourly_df, m15_df, symbol=gen["symbol"])
    else:
        # Single-timeframe loop
        for tf in stf["timeframes"]:
            df_tf = resample_df(raw_df, tf)
            result = analyze(
                df_tf,
                sma_window=stf["sma_window"],
                rsi_window=stf["rsi_window"],
                oversold=stf["oversold"],
                overbought=stf["overbought"],
            )
            plot_signals(result, symbol=gen["symbol"], timeframe=tf)

if __name__ == "__main__":
    main()
