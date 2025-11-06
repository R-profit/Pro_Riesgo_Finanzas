# estrategia
# modules/eth_analysis.py
import yfinance as yf
import pandas as pd
import numpy as np

def load_eth_data(period='1mo', interval='1h'):
    """Descarga datos de ETH/USD desde Yahoo Finance"""
    df = yf.download('ETH-USD', period=period, interval=interval)
    df.dropna(inplace=True)
    return df

def calculate_indicators(df):
    """Calcula indicadores técnicos básicos"""
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'])
    return df

def compute_rsi(series, window=14):
    """Cálculo del RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def analyze_eth():
    """Función principal de análisis"""
    df = load_eth_data()
    df = calculate_indicators(df)

    # Señal simple de ejemplo
    df['Signal'] = np.where(df['SMA_20'] > df['SMA_50'], 'BUY', 'SELL')
    return df
