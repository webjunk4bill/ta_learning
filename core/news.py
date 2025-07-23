import os
import json
from typing import List, Dict

import requests


_DEF_ENV_FILE = ".env"


def _load_api_key() -> str | None:
    """Load CryptoPanic API key from environment or .env file."""
    key = os.environ.get("CRYPTOPANIC_API_KEY")
    if key:
        return key
    if os.path.exists(_DEF_ENV_FILE):
        with open(_DEF_ENV_FILE) as f:
            for line in f:
                if line.strip().startswith("CRYPTOPANIC_API_KEY="):
                    return line.strip().split("=", 1)[1]
    return None


def fetch_latest_news(limit: int = 20) -> List[Dict]:
    """Fetch latest crypto news from CryptoPanic."""
    key = _load_api_key()
    if not key:
        raise ValueError("CryptoPanic API key not found in environment or .env")

    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": key,
        "currencies": "BTC,ETH",
        "filter": "hot",
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    headlines = []
    for post in data.get("results", [])[:limit]:
        headlines.append(
            {
                "source": post.get("source", {}).get("title") or post.get("source"),
                "title": post.get("title"),
                "url": post.get("url"),
                "published_at": post.get("published_at"),
                "tags": [c.get("code") for c in post.get("currencies", [])],
            }
        )

    return headlines
