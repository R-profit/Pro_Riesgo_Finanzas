#señales de compra y venta con ema
# modules/ema_strategy.py
import numpy as np
import pandas as pd

def apply_ema_strategy(df, short_window=5, long_window=20):
    """Calcula EMAs, señales y columnas Buy/Sell"""
    df['EMA_short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=long_window, adjust=False).mean()

    signals = [0]
    for i in range(1, len(df)):
        short = df['EMA_short'][i]
        long = df['EMA_long'][i]
        prev_short = df['EMA_short'][i-1]
        prev_long = df['EMA_long'][i-1]

        if prev_short < prev_long and short >= long:
            signals.append(1)
        elif prev_short > prev_long and short <= long:
            signals.append(-1)
        else:
            signals.append(0)

    df['Signals'] = signals
    df['Buy'] = np.where(df['Signals'] == 1, df['Close'], np.nan)
    df['Sell'] = np.where(df['Signals'] == -1, df['Close'], np.nan)
    return df

def backtest_strategy(df, tp=0.03, sl=0.01):
    """Simula evolución del capital con TP/SL"""
    N = len(df)
    equity = [100]
    pos = 0
    price = 0

    for i in range(1, N):
        equity.append(equity[i-1])
        if pos == 1:
            if df['Close'][i] >= price*(1+tp):
                pos = 0
                equity[i] *= (1+tp)
            elif df['Close'][i] <= price*(1-sl):
                pos = 0
                equity[i] *= (1-sl)

        elif pos == -1:
            if df['Close'][i] <= price*(1-tp):
                pos = 0
                equity[i] *= (1+tp)
            elif df['Close'][i] >= price*(1+sl):
                pos = 0
                equity[i] *= (1-sl)

        else:
            if df['Signals'][i] != 0:
                pos = df['Signals'][i]
                price = df['Close'][i]

    df['Equity'] = equity
    return df
