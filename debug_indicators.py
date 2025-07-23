import requests
import json

payload = {
    "symbol": "BTC/USDT",
    "exchange": "binanceus",
    "resolution": "1h",
    "indicators": {
        "rsi": {"window": 14},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "bollinger": {"window": 20, "std_dev": 2},
    },
}

resp = requests.post("http://localhost:8000/compute_indicators", json=payload)
print("Status:", resp.status_code)
print(json.dumps(resp.json(), indent=2))
