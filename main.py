# main.py
import argparse
import yaml
from rich.console import Console
from loguru import logger

from core.visualizer import plot_multi_tf
from core.logger import init_logger
from core.dataloader import resample_df

from ta_engine.data import load_price_data
from ta_engine.legacy_strategies import run_strategy
from ta_engine.plotting import plot_signals
from core.methods.multi_mean_reversion import trend_analyze, zone_analyze

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
    console.print("[bold green]Starting analysis...[/bold green]")
    args = parse_args()
    # Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    gen = cfg["general"]
    stf = cfg["single_tf"]
    mtf = cfg["multi_tf"]

    init_logger(debug=gen.get("debug", False))
    debug = gen.get("debug", False)
    logger.info("Loaded config from {}", args.config)

    # Load and optionally filter data
    logger.info("Loading data range {start} to {end}", start=gen["start_date"], end=gen["end_date"])
    raw_df = load_price_data(gen["file"])
    raw_df = raw_df.loc[gen["start_date"] : gen["end_date"]]

    result = run_strategy(raw_df, cfg, debug=debug)

    if gen.get("multi_tf"):
        # result is 15-minute dataframe with Equity column
        equity = result.pop("Equity")
        daily_df = trend_analyze(
            resample_df(raw_df, "1D"),
            sma_window=mtf["trend_sma_window"],
        )
        hourly_df = zone_analyze(
            resample_df(raw_df, "1H"),
            rsi_window=mtf["zone_rsi_window"],
            rsi_oversold=mtf["zone_oversold"],
            rsi_overbought=mtf["zone_overbought"],
            bb_window=mtf["bb_window"],
            bb_sigma=mtf["bb_sigma"],
        )
        plot_multi_tf(
            daily_df,
            hourly_df,
            result,
            equity=equity,
            symbol=gen["symbol"],
        )
    else:
        timeframes = result["Timeframe"].unique()
        for tf in timeframes:
            df_tf = result[result["Timeframe"] == tf]
            plot_signals(df_tf, df_tf["signal"]) 

    logger.success("Analysis complete")

if __name__ == "__main__":
    main()
