### Example: ETH/USDC on Kraken (4h & 1d)

**Tool calls**
1. `GET /ohlcv?symbol=ETH/USDC&exchange=kraken&timeframe=4h&limit=1000`
2. `GET /ohlcv?symbol=ETH/USDC&exchange=kraken&timeframe=1d&limit=1000`
3. `POST /indicators` with body `{ "symbol":"ETH/USDC", "exchange":"kraken", "timeframe":"4h", "inputs":{"rsi":{},"macd":{},"bb":{},"ema":[20,50,200],"atr":{},"adx":{}} }`
4. `GET /cryptopanic?asset=ETH`
5. `GET /reddit/search?subreddits=ethtrader,ethereum`

**Summary**
Market is in placeholder uptrend; sentiment mixed.

```json
{
  "as_of":"2024-01-01T00:00:00Z",
  "asset":"ETH/USDC",
  "exchange":"kraken",
  "timeframes":["4h","1d"],
  "market_state":{
    "trend":{"4h":"up","1d":"up"},
    "volatility_regime":{},
    "structure":{}
  },
  "indicators":{
    "rsi":55.0,
    "macd":{"macd":1.0,"signal":0.5,"hist":0.5},
    "bb":{"lower":1000.0,"basis":1100.0,"upper":1200.0},
    "ema":{"20":1100.0,"50":1050.0,"200":900.0},
    "atr":10.0,
    "adx":20.0
  },
  "sentiment":{"cryptopanic":[],"reddit":[]},
  "outlook":{"short_term":{"if":"price holds above 1100","then":"target 1200","confidence":0.6},"long_term":{"if":"price breaks 1300","then":"target 1500","confidence":0.5}}
}
```
