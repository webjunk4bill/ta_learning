# main.py
import argparse
import yaml
from rich.console import Console
from core.dataloader import load_data, resample_df
from core.methods.mean_reversion import analyze
from core.methods.multi_mean_reversion import (
    trend_analyze,
    zone_analyze,
    trigger,
    multi_tf_filter,
)
from core.visualizer import plot_signals, plot_multi_tf
from core.logger import init_logger
from loguru import logger
import sys

# Reconfigure Loguru to emit INFO+ logs to standard error
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
from core.backtest import backtest_signals

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Technical Analysis Learning CLI")
    parser.add_argument(
        "--config", "-c",
        help="YAML config file (default: config.yaml)",
        default="config.yaml",
    )
    return parser.parse_args()

def main():
    init_logger()
    console.print("[bold green]Starting analysis...[/bold green]")
    args = parse_args()
    # Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config from {}", args.config)
    gen = cfg["general"]
    stf = cfg["single_tf"]
    mtf = cfg["multi_tf"]

    # Load and optionally filter data
    logger.info("Loading data range {start} to {end}", start=gen["start_date"], end=gen["end_date"])
    raw_df = load_data(gen["file"])
    raw_df = raw_df.loc[gen["start_date"] : gen["end_date"]]

    if gen.get("multi_tf"):
        logger.info("Running multi-timeframe analysis")
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
        equity = backtest_signals(m15_df)
        plot_multi_tf(daily_df, hourly_df, m15_df, equity=equity, symbol=gen["symbol"])

    else:
        logger.info("Running single-timeframe analysis")
        # Single-timeframe loop
        for tf in stf["timeframes"]:
            logger.debug(f"Processing timeframe {tf}")
            df_tf = resample_df(raw_df, tf)
            result = analyze(
                df_tf,
                sma_window=stf["sma_window"],
                rsi_window=stf["rsi_window"],
                oversold=stf["oversold"],
                overbought=stf["overbought"],
            )
            plot_signals(result, symbol=gen["symbol"], timeframe=tf)

    logger.success("Analysis complete")

if __name__ == "__main__":
    main()
