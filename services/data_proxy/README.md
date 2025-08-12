uvicorn app.main:app --reload --port 8001
# GET http://localhost:8001/health
# GET http://localhost:8001/ohlcv?symbol=ETH/USDC&exchange=kraken&timeframe=4h&limit=1000
