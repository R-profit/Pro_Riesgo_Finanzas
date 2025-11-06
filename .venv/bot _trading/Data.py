#descarga y preparacion de datos
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def descargar_datos(ticker="AMZN", start="2020-08-01", end="2021-03-31"):
    data = yf.download(ticker, start=start, end=end).dropna()
    data = data[['Close']]
    return data

def dividir_datos(data, test_ratio=0.07):
    h = round(len(data) * test_ratio)
    train = data[:-h]
    test = data[-h:]
    return train, test

def graficar_datos(data, titulo="Precios de Cierre"):
    plt.figure(figsize=(10, 4))
    plt.plot(data['Close'])
    plt.title(titulo)
    plt.ylabel('Precio ($)')
    plt.show()

# modules/data_loader.py
import pandas_datareader as pdr
import yfinance as yf

def get_fred_data(symbol, start, end):
    """Descarga datos desde FRED (por ejemplo, petróleo Brent)"""
    df = pdr.get_data_fred(symbol, start=start, end=end)
    df = df.dropna()
    df.columns = ['Close']
    return df

def get_yahoo_data(symbol, start, end):
    """Descarga datos desde Yahoo Finance (por ejemplo, S&P 500 o criptos)"""
    df = yf.download(symbol, start=start, end=end)
    df = df.dropna()
    return df
