import requests
import json

from core.logger import init_logger
init_logger(debug=True)

payload = {
    "symbol": "BTC/USDT",
    "exchange": "binanceus",
    "resolutions": {"short_term": "1h", "long_term": "4h"},
    "indicators": {
        "rsi": {"window": 14},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "bollinger": {"window": 20, "std_dev": 2},
    },
}

resp = requests.post("http://localhost:8000/summary_signal", json=payload)
print("Status:", resp.status_code)
print(json.dumps(resp.json(), indent=2))
