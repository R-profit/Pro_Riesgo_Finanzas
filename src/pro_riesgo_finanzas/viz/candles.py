# %% ======================================================
# Candlestick y Cruce EMA/RSI (backend offscreen, sin plt.show)
# ======================================================
import os
import pandas as pd  # type: ignore
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import mplfinance as mpf  # type: ignore

def plot_candles(df: pd.DataFrame, symbol: str, outdir: str = "outputs", last_n: int = 120) -> str:
    """
    Guarda un gráfico de velas de los últimos `last_n` registros.
    Requiere columnas: Open, High, Low, Close (y Volume para barra inferior).
    """
    os.makedirs(outdir, exist_ok=True)
    req = {"Open", "High", "Low", "Close"}
    if df is None or df.empty or not req.issubset(df.columns):
        return ""

    plot_df = df.tail(last_n).copy()
    plot_df.index = pd.to_datetime(plot_df.index)

    fig, _ = mpf.plot(
        plot_df, type="candle", style="charles", volume=True, returnfig=True,
        title=f"Últimos {last_n} — {symbol}"
    )
    path = os.path.join(outdir, f"candles_{symbol.replace('=', '_').replace('/', '_')}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path

def plot_ema_rsi(df: pd.DataFrame, symbol: str, outdir: str = "outputs") -> str:
    """
    Precio + EMA20/EMA50 y panel con RSI. Calcula EMA/RSI si no existen.
    Usa Adj Close si está disponible, si no Close.
    """
    os.makedirs(outdir, exist_ok=True)
    price = df.get("Adj Close", df.get("Close"))
    if price is None or price.empty:
        return ""

    price = price.astype(float).dropna().tail(365)
    if price.empty:
        return ""

    ema20 = df.get("EMA_20", price.ewm(span=20, adjust=False).mean())
    ema50 = df.get("EMA_50", price.ewm(span=50, adjust=False).mean())

    # RSI mínimo sin dependencia adicional
    delta = price.diff()
    up = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rsi = 100 - (100 / (1 + (up / down)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax1.plot(price.index, price.values, label="Precio", linewidth=1)
    ax1.plot(ema20.index, ema20.values, label="EMA 20", linewidth=1.4)
    ax1.plot(ema50.index, ema50.values, label="EMA 50", linewidth=1.4)
    ax1.set_title(f"Cruce EMA(20/50) + RSI — {symbol}")
    ax1.grid(True, alpha=0.3); ax1.legend(loc="upper left")

    ax2.plot(rsi.index, rsi.values, linewidth=1)
    ax2.axhline(70, color="red", linestyle="--", alpha=0.6)
    ax2.axhline(30, color="blue", linestyle="--", alpha=0.6)
    ax2.set_ylim(0, 100); ax2.set_ylabel("RSI"); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, f"ema_rsi_{symbol.replace('=', '_').replace('/', '_')}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path

