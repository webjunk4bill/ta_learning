import os
import sys
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

# Set API key before importing app
sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ["API_KEY"] = "test"

from server.app import app  # noqa: E402


def test_run_strategy_route(tmp_path):
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC"),
            "Open": [1, 2, 3, 2, 1],
            "High": [1, 2, 3, 2, 1],
            "Low": [1, 2, 3, 2, 1],
            "Close": [1, 2, 3, 2, 1],
            "Volume": [100] * 5,
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    client = TestClient(app)
    payload = {
        "filepath": str(csv_path),
        "config": {
            "strategy": "ema_crossover",
            "short_window": 2,
            "long_window": 3,
        },
    }
    resp = client.post("/run_strategy", json=payload, headers={"x-api-key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 3
    for row in data[-3:]:
        assert {
            "Signal",
            "Price",
            "EMA_short",
            "EMA_long",
            "Crossover",
        }.issubset(row.keys())
        assert row["Signal"] in {"long", "short", "neutral"}
