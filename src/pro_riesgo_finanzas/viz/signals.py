# %% ============================================
# VIZ: señales EMA/MACD y Bollinger/RSI
# - Sin descargas; recibe DataFrame con columnas del core
# - Por defecto guarda PNG en outputs/; show=True para plt.show()
# se puede llamar desde cualquier scripts app o noteboook
# ===============================================
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt

def plot_ema_macd(df: pd.DataFrame, title: str = "EMA+MACD con señales"):
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(13,7),sharex=True,gridspec_kw={'height_ratios':[3,1]})
    ax1.plot(df.index, df["close"], color='black', label='Precio')
    if "EMA_short" in df and "EMA_long" in df:
        ax1.plot(df.index, df["EMA_short"], '--', label='EMA Corta')
        ax1.plot(df.index, df["EMA_long"],  '--', label='EMA Larga')
    if "Buy_EMA" in df:
        ax1.scatter(df.index, df["Buy_EMA"],  marker='^', color='green', s=60, label='Compra EMA')
    if "Sell_EMA" in df:
        ax1.scatter(df.index, df["Sell_EMA"], marker='v', color='red',   s=60, label='Venta EMA')
    ax1.grid(True, linestyle=':')
    ax1.legend(loc='upper left'); ax1.set_title(title)

    need = {"MACD","Signal_Line","MACD_Hist"}
    if need.issubset(df.columns):
        ax2.plot(df.index, df["MACD"], label="MACD")
        ax2.plot(df.index, df["Signal_Line"], '--', label="Señal")
        ax2.bar(df.index, df["MACD_Hist"], alpha=0.35, label="Hist")
        ax2.grid(True, linestyle=':'); ax2.legend(loc='upper left')
    plt.tight_layout(); plt.show()

def plot_bollinger_rsi(df: pd.DataFrame, title: str = "Bollinger+RSI"):
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(13,7),sharex=True)
    ax1.plot(df.index, df["close"], color='black', label='Precio')
    for c, ls in [("BOL_M",'--'),("BOL_U",':'),("BOL_D",':')]:
        if c in df.columns:
            ax1.plot(df.index, df[c], ls, label=c)
    for col, marker, colr, lab in [
        ("Buy_BOLL", '^', 'green', 'Buy'),
        ("Sell_BOLL", 'v', 'red', 'Sell'),
        ("Buy_BOLL_SQUEEZE", 'P', 'lime', 'Buy Sq'),
        ("Sell_BOLL_SQUEEZE", 'X', 'darkorange', 'Sell Sq')
    ]:
        if col in df.columns:
            ax1.scatter(df.index, df[col], marker=marker, color=colr, s=60, label=lab)
    ax1.grid(True, linestyle=':'); ax1.legend(loc='upper left'); ax1.set_title(title)

    if "RSI" in df.columns:
        ax2.plot(df.index, df["RSI"], label="RSI")
        ax2.axhline(70, color='red', linestyle='--', alpha=0.6)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.6)
        ax2.fill_between(df.index, 70, 30, color='gray', alpha=0.06)
        ax2.grid(True, linestyle=':'); ax2.legend(loc='upper left')
    plt.tight_layout(); plt.show()
