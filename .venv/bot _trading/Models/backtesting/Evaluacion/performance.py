#ratio sharpe
# modules/performance.py
import numpy as np

def calculate_sharpe(df, risk_free=0.01):
    """Calcula ratio de Sharpe anualizado"""
    ret = np.log(df['Equity']).diff().mean() * 252
    vol = np.log(df['Equity']).diff().std() * np.sqrt(252)
    sharpe_ratio = (ret - risk_free) / vol
    return sharpe_ratio
