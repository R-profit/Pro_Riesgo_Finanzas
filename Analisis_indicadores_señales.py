## Señales indicadores sma, ema, bandas de bollinger, macd
import yfinance as yfin
import pandas as pd
import ta

def load_data(symbol='BTC-USD', start='2020-07-01', end='2021-07-01'):
    """Descarga datos históricos de un activo."""
    df = yfin.download(symbol, start=start, end=end)
    return df

def add_sma(df, short=5, long=20):
    """Agrega medias móviles simples."""
    df['SMA_short'] = df['Adj Close'].rolling(window=short).mean()
    df['SMA_long'] = df['Adj Close'].rolling(window=long).mean()
    return df

def add_ema(df, short=5, long=20):
    """Agrega medias móviles exponenciales."""
    df['EMA_short'] = df['Adj Close'].ewm(span=short, adjust=False).mean()
    df['EMA_long'] = df['Adj Close'].ewm(span=long, adjust=False).mean()
    return df

def add_bollinger(df, window=20, dev=2):
    """Agrega Bandas de Bollinger."""
    bb = ta.volatility.BollingerBands(close=df['Close'], window=window, window_dev=dev)
    df['BOL_M'] = bb.bollinger_mavg()
    df['BOL_U'] = bb.bollinger_hband()
    df['BOL_D'] = bb.bollinger_lband()
    return df

def add_macd(df, fast=12, slow=26):
    """Agrega el indicador MACD."""
    macd = ta.trend.MACD(close=df['Close'], window_fast=fast, window_slow=slow)
    df['MACD'] = macd.macd()
    return df

def prepare_indicators(symbol='BTC-USD', start='2020-07-01', end='2021-07-01'):
    """Pipeline completo para generar los indicadores."""
    df = load_data(symbol, start, end)
    df = add_sma(df)
    df = add_ema(df)
    df = add_bollinger(df)
    df = add_macd(df)
    return df
