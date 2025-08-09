import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ta_engine.strategies.ema_crossover import EmaCrossover


def test_ema_crossover_generates_signals():
    dates = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    prices = [1, 2, 3, 2, 1]
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": [100] * len(prices),
        },
        index=dates,
    )

    strat = EmaCrossover()
    result = strat.run(df, {"short_window": 2, "long_window": 3})

    assert {"EMA_short", "EMA_long", "Crossover"}.issubset(result.columns)
    assert result.loc[dates[1], "Signal"] == "long"
    assert result.loc[dates[-1], "Signal"] == "short"
