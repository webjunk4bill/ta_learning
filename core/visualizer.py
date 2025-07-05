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

    # Annotate buy signals with reasons
    for idx, row in buys.iterrows():
        reason = row.get("reason", "")
        if reason:
            plt.annotate(
                reason,
                (idx, row["Close"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                rotation=0
            )

    # Annotate sell signals with reasons
    for idx, row in sells.iterrows():
        reason = row.get("reason", "")
        if reason:
            plt.annotate(
                reason,
                (idx, row["Close"]),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=8,
                rotation=0
            )

    # Formatting
    title = f"{symbol} | Mean Reversion | {timeframe}"
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


def plot_multi_tf(
    daily_df,
    hourly_df,
    m15_df,
    symbol: str = ""
):
    """
    Three‑panel chart for multi‑time‑frame mean‑reversion analysis.

    Row 1: Daily trend with up/down shading.
    Row 2: Hourly RSI zones (oversold / overbought markers).
    Row 3: 15‑minute price with filtered entry signals.

    Args:
        daily_df : DataFrame containing columns ['Close', 'trend'].
        hourly_df: DataFrame containing columns ['Close', 'zone'].
        m15_df   : DataFrame containing columns ['Close', 'signal'].
        symbol   : Ticker symbol for chart title.
    """
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 2]},
    )

    # ------------------ Daily trend ------------------
    ax0 = axes[0]
    ax0.plot(daily_df.index, daily_df["Close"], label="Close")

    up_trend = daily_df["trend"] == 1
    down_trend = daily_df["trend"] == -1

    ax0.fill_between(
        daily_df.index,
        daily_df["Close"].min(),
        daily_df["Close"].max(),
        where=up_trend,
        alpha=0.1,
        color="green",
        label="Up‑trend",
    )
    ax0.fill_between(
        daily_df.index,
        daily_df["Close"].min(),
        daily_df["Close"].max(),
        where=down_trend,
        alpha=0.1,
        color="red",
        label="Down‑trend",
    )
    ax0.set_title(f"{symbol} | Daily Trend")
    ax0.legend()
    ax0.grid(True)

    # ------------------ Hourly setup ------------------
    ax1 = axes[1]
    ax1.plot(hourly_df.index, hourly_df["Close"], label="Close")
    # Plot Bollinger Bands if present
    bb_upper_cols = [c for c in hourly_df.columns if c.startswith("BB_U_")]
    bb_lower_cols = [c for c in hourly_df.columns if c.startswith("BB_L_")]
    if bb_upper_cols and bb_lower_cols:
        up_col = bb_upper_cols[0]
        lo_col = bb_lower_cols[0]
        ax1.plot(hourly_df.index, hourly_df[up_col], color="grey", linewidth=0.8, alpha=0.7, label="Upper BB")
        ax1.plot(hourly_df.index, hourly_df[lo_col], color="grey", linewidth=0.8, alpha=0.7, label="Lower BB")

    oversold = hourly_df["zone"] == "oversold"
    overbought = hourly_df["zone"] == "overbought"

    ax1.scatter(
        hourly_df[oversold].index,
        hourly_df[oversold]["Close"],
        marker="^",
        color="green",
        label="Hourly oversold",
    )
    ax1.scatter(
        hourly_df[overbought].index,
        hourly_df[overbought]["Close"],
        marker="v",
        color="red",
        label="Hourly overbought",
    )
    ax1.set_title("Hourly RSI Zones")
    ax1.legend()
    ax1.grid(True)

    # ------------------ 15‑minute triggers ------------------
    ax2 = axes[2]
    ax2.plot(m15_df.index, m15_df["Close"], label="Close")
    # Plot Bollinger Bands if present
    bb_upper_cols = [c for c in m15_df.columns if c.startswith("BB_U_")]
    bb_lower_cols = [c for c in m15_df.columns if c.startswith("BB_L_")]
    if bb_upper_cols and bb_lower_cols:
        up_col = bb_upper_cols[0]
        lo_col = bb_lower_cols[0]
        ax2.plot(m15_df.index, m15_df[up_col], color="grey", linewidth=0.8, alpha=0.7, label="Upper BB")
        ax2.plot(m15_df.index, m15_df[lo_col], color="grey", linewidth=0.8, alpha=0.7, label="Lower BB")

    longs = m15_df["signal"] == 1
    shorts = m15_df["signal"] == -1

    ax2.scatter(
        m15_df[longs].index,
        m15_df[longs]["Close"],
        marker="^",
        color="green",
        label="Long entry",
    )
    ax2.scatter(
        m15_df[shorts].index,
        m15_df[shorts]["Close"],
        marker="v",
        color="red",
        label="Short entry",
    )
    ax2.set_title("15‑Minute Entries (Filtered)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

