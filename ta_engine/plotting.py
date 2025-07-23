import pandas as pd
import matplotlib.pyplot as plt


def plot_signals(df: pd.DataFrame, signals: pd.Series) -> None:
    """Plot price series with buy/sell signal markers."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label="Close")

    buys = signals[signals == 1].index
    sells = signals[signals == -1].index
    plt.scatter(buys, df.loc[buys, "Close"], marker="^", color="green", label="Buy")
    plt.scatter(sells, df.loc[sells, "Close"], marker="v", color="red", label="Sell")

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
