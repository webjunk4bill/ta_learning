# main.py
import argparse
from core.dataloader import load_data, resample_df
from core.methods.mean_reversion import analyze
from core.visualizer import plot_signals

def parse_args():
    parser = argparse.ArgumentParser(description="Technical Analysis Learning CLI")
    parser.add_argument("--method", default="mean_reversion", choices=["mean_reversion"], help="Trading method to apply")
    parser.add_argument("--file", required=True, help="Path to CSV file with OHLCV data")
    parser.add_argument("--sma-window", type=int, default=20, help="SMA window size")
    parser.add_argument("--rsi-window", type=int, default=14, help="RSI window size")
    parser.add_argument("--oversold", type=float, default=30.0, help="RSI oversold threshold")
    parser.add_argument("--overbought", type=float, default=70.0, help="RSI overbought threshold")
    parser.add_argument("--timeframes", default="1D", help="Comma-separated list of timeframes (e.g., '1D,1H,15T')")
    parser.add_argument("--symbol", default="", help="Ticker symbol for plot titles")
    return parser.parse_args()

def main():
    args = parse_args()
    raw_df = load_data(args.file)
    for tf in args.timeframes.split(","):
        df_tf = resample_df(raw_df, tf)
        result = analyze(
            df_tf,
            sma_window=args.sma_window,
            rsi_window=args.rsi_window,
            oversold=args.oversold,
            overbought=args.overbought
        )
        plot_signals(result, symbol=args.symbol, timeframe=tf)

if __name__ == "__main__":
    main()
