#Simula operaciones y mide resultados de la estratregia
# ===========================================
# Módulo de Backtesting del Bot de Trading
# ===========================================
# Autor: Rafael Medina Arias (R_Profit)
# Descripción:
# Simula operaciones de trading con datos históricos,
# aplicando estrategias definidas por el usuario.
# ===========================================

import pandas as pd
import numpy as np

# ===========================================
# FUNCIONES PRINCIPALES
# ===========================================

def ejecutar_backtest(data, strategy_func, capital_inicial=10000, lote=1):
    """
    Ejecuta un backtest básico con señales de compra/venta.

    Parámetros:
    - data: DataFrame con precios e indicadores.
    - strategy_func: función que genera señales ('Buy', 'Sell', 'Hold').
    - capital_inicial: capital de partida.
    - lote: número de unidades a operar por transacción.

    Retorna:
    - data con columna 'Position' y 'Portfolio'
    - métricas del rendimiento
    """
    data = data.copy()
    data['Signal'] = strategy_func(data)
    data['Position'] = 0  # 1 = compra, -1 = venta, 0 = fuera

    # Determinar posiciones
    data.loc[data['Signal'] == 'Buy', 'Position'] = 1
    data.loc[data['Signal'] == 'Sell', 'Position'] = -1

    # Calcular retornos
    data['Market_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Market_Return'] * data['Position'].shift(1)

    # Calcular evolución del portafolio
    data['Portfolio'] = (1 + data['Strategy_Return']).cumprod() * capital_inicial

    # Métricas
    retorno_total = (data['Portfolio'].iloc[-1] - capital_inicial) / capital_inicial
    max_drawdown = calcular_drawdown(data['Portfolio'])
    sharpe = calcular_sharpe(data['Strategy_Return'])
    win_rate = calcular_winrate(data['Strategy_Return'])

    metricas = {
        'Retorno Total (%)': round(retorno_total * 100, 2),
        'Max Drawdown (%)': round(max_drawdown * 100, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Win Rate (%)': round(win_rate * 100, 2)
    }

    return data, metricas

# ===========================================
# FUNCIONES AUXILIARES
# ===========================================

def calcular_drawdown(portfolio):
    """Calcula el máximo drawdown de un portafolio."""
    roll_max = portfolio.cummax()
    drawdown = (portfolio - roll_max) / roll_max
    return drawdown.min()

def calcular_sharpe(strategy_returns, risk_free_rate=0.02):
    """Calcula el Sharpe ratio anualizado."""
    mean_return = strategy_returns.mean() * 252
    std_return = strategy_returns.std() * np.sqrt(252)
    if std_return == 0:
        return 0
    return (mean_return - risk_free_rate) / std_return

def calcular_winrate(strategy_returns):
    """Porcentaje de operaciones con retorno positivo."""
    wins = np.sum(strategy_returns > 0)
    total = np.sum(strategy_returns != 0)
    return wins / total if total > 0 else 0

# ===========================================
# TEST LOCAL
# ===========================================
if __name__ == "__main__":
    import yfinance as yf

    # Ejemplo con una estrategia simple (SMA crossover)
    def sma_strategy(data):
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        signals = np.where(data['SMA_20'] > data['SMA_50'], 'Buy',
                  np.where(data['SMA_20'] < data['SMA_50'], 'Sell', 'Hold'))
        return signals

    df = yf.download('ETH-USD', start='2023-01-01', end='2025-01-01')
    df, metricas = ejecutar_backtest(df, sma_strategy)
    print(metricas)
