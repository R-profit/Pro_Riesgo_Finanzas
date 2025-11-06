#Metricas RMSE, MAPE, Sharpe,Drawdown etc.
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from math import sqrt

def calcular_metricas(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mape

def comparar_resultados(resultados_dict):
    resultados = pd.DataFrame(resultados_dict)
    print(resultados)
    return resultados
