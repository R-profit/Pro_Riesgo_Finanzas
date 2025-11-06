# entrada principal de la aplicaion
from data_handler import descargar_datos, dividir_datos, graficar_datos
from models.arima_models import entrenar_arima, entrenar_autoarima, pronosticar
from models.exp_smoothing_models import entrenar_suavizamiento
from models.neural_network_model import preparar_datos_red, entrenar_red_neuronal
from evaluation import calcular_metricas, comparar_resultados

# === Descarga de datos ===
data = descargar_datos("AMZN", "2020-08-01", "2021-03-31")
graficar_datos(data, "Precios de Cierre - AMZN")
train, test = dividir_datos(data)

# === ARIMA ===
arima_model = entrenar_arima(train)
auto_model = entrenar_autoarima(train)
forecast_arima = pronosticar(arima_model, len(test))
forecast_auto = auto_model.predict(n_periods=len(test))

# === Suavizamiento Exponencial ===
ses, holt, hw, ets = entrenar_suavizamiento(train)
forecast_ses = ses.forecast(len(test))
forecast_holt = holt.forecast(len(test))
forecast_hw = hw.forecast(len(test))
forecast_ets = ets.forecast(len(test))

# === Red Neuronal ===
X_train, y_train = preparar_datos_red(train)
nn = entrenar_red_neuronal(X_train, y_train)
X_test, y_test = preparar_datos_red(data)
forecast_nn = nn.predict(X_test[-len(test):])

# === Métricas ===
resultados = {
    'Modelo': ['ARIMA(7,1,3)', 'Auto ARIMA', 'SES', 'Holt', 'HW', 'ETS', 'NeuralNet'],
    'RMSE': [],
    'MAPE': []
}

modelos_y_pred = [
    forecast_arima, forecast_auto, forecast_ses, forecast_holt,
    forecast_hw, forecast_ets, forecast_nn
]

for pred in modelos_y_pred:
    rmse, mape = calcular_metricas(test['Close'], pred)
    resultados['RMSE'].append(rmse)
    resultados['MAPE'].append(mape)

comparar_resultados(resultados)

# toma lo indicadores tecnicos

from indicators.technical_indicators import prepare_indicators
import matplotlib.pyplot as plt

# Cargar datos e indicadores
df = prepare_indicators('BTC-USD', start='2020-07-01', end='2021-07-01')

# Mostrar algunas columnas
print(df[['Adj Close', 'SMA_short', 'SMA_long', 'EMA_short', 'EMA_long', 'MACD']].tail())

# Ejemplo de gráfico
plt.figure(figsize=(16,8))
plt.title('BTC-USD: SMA & EMA')
plt.plot(df['Adj Close'], label='Adj Close', alpha=0.8)
plt.plot(df['SMA_short'], label='SMA 5', alpha=0.5)
plt.plot(df['SMA_long'], label='SMA 20', alpha=0.5)
plt.plot(df['EMA_short'], label='EMA 5', alpha=0.4)
plt.plot(df['EMA_long'], label='EMA 20', alpha=0.4)
plt.legend()
plt.show()

#analiza el activo
from analysis.eth_analysis import ejecutar_analisis

# Ejecutar análisis antes del backtesting o toma de decisiones
data, retorno_anual, vol_anual, resumen = ejecutar_analisis('ETH-USD', start='2022-01-01', end='2025-01-01')

#conecta analisis estreteias y bactesting
from analysis.eth_analysis import ejecutar_analisis
from backtesting.backtester import ejecutar_backtest

# Paso 1: Analizar activo
data, _, _, _ = ejecutar_analisis('ETH-USD', start='2023-01-01', end='2025-01-01', graficar=False)

# Paso 2: Definir estrategia (ejemplo)
def estrategia_basica(data):
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    signals = np.where(data['EMA_20'] > data['EMA_50'], 'Buy',
              np.where(data['EMA_20'] < data['EMA_50'], 'Sell', 'Hold'))
    return signals

# Paso 3: Ejecutar backtesting
data_bt, metricas = ejecutar_backtest(data, estrategia_basica)
print(metricas)


#Trae La estretegia
# main.py
from modules.eth_analysis import analyze_eth

def main():
    eth_df = analyze_eth()
    print(eth_df.tail())

    # Aquí puedes enviar señales al bot o al módulo de ejecución
    last_signal = eth_df['Signal'].iloc[-1]
    print(f"Última señal ETH/USD: {last_signal}")

if __name__ == "__main__":
    main()
    
    #estrategia 2
# main.py
from modules.data_loader import get_fred_data, get_yahoo_data
from modules.ema_strategy import apply_ema_strategy, backtest_strategy
from modules.performance import calculate_sharpe
from modules.visualization import plot_signals, compare_equity

def main():
    # Datos Brent (FRED)
    df = get_fred_data('DCOILBRENTEU', '2016-01-01', '2021-01-01')
    df = apply_ema_strategy(df)
    df = backtest_strategy(df)

    # Datos S&P 500 (Yahoo)
    df2 = get_yahoo_data('^GSPC', '2016-01-01', '2021-01-01')
    df2['Equity'] = (100 / df2['Close'][0]) * df2['Close']

    # Resultados
    sharpe = calculate_sharpe(df)
    print(f"Sharpe Ratio estrategia: {sharpe:.3f}")

    # Visualización
    plot_signals(df, title='Estrategia EMA Brent Oil')
    compare_equity(df, df2, 'Estrategia EMA', 'S&P 500')

if __name__ == "__main__":
    main()
    
# trae el comparador de activos
from modules.data_loader import get_data
from modules.plotter import plot_candlestick
from modules.config import ASSETS

def main():
    # CEMEX
    cemex_data = get_data(**ASSETS["CEMEX"])
    plot_candlestick(cemex_data, title="Velas japonesas CEMEX")

    # ECOPETROL diario
    ecopetrol_data = get_data(**ASSETS["ECOPETROL"])
    plot_candlestick(ecopetrol_data, title="Velas japonesas ECOPETROL")

    # ECOPETROL horario (1 semana)
    eco_hourly = get_data(**ASSETS["ECOPETROL_HOURLY"])
    plot_candlestick(eco_hourly, title="ECOPETROL 1 semana", mav=(5, 20))

if __name__ == "__main__":
    main()
