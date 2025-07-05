# core/visualizer.py
import matplotlib.pyplot as plt

def plot_signals(df, symbol: str = "", timeframe: str = ""):
    """
    Plot Close price and SMA, and mark buy/sell signals.
    """
    # Identify SMA column
    sma_cols = [col for col in df.columns if col.startswith("SMA_")]
    sma_col = sma_cols[0] if sma_cols else None

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label="Close")
    if sma_col:
        plt.plot(df.index, df[sma_col], label=sma_col)

    # Plot buy and sell signals
    buys = df[df["signal"] == 1]
    sells = df[df["signal"] == -1]
    plt.scatter(buys.index, buys["Close"], marker="^", label="Buy", alpha=1)
    plt.scatter(sells.index, sells["Close"], marker="v", label="Sell", alpha=1)

    # Formatting
    title = f"{symbol} | Mean Reversion | {timeframe}"
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

