#!/usr/bin/env python
import argparse
import datetime as dt
from pathlib import Path

# from openai import OpenAI


def run(args):
    date = dt.date.today().isoformat()
    outdir = Path("reports") / date
    outdir.mkdir(parents=True, exist_ok=True)
    for asset in args.assets.split(","):
        fname_base = f"{asset}-{args.exchange}"
        (outdir / f"{fname_base}.json").write_text("{}")
        (outdir / f"{fname_base}.md").write_text("TODO: call GPT and render report")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True, help="Comma separated list, e.g. ETH/USDC,BTC/USDT")
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--timeframes", default="4h,1d")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
