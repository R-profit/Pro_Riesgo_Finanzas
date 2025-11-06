# Señales indicadores SMA, EMA, Bandas de Bollinger, MACD y RSI
# ===============================================================
# 🔒 ANÁLISIS TÉCNICO SEGURO
# Autor: Rafy (R_Profit)
# Descripción: Cálculo de indicadores, generación de señales, visualización y análisis de rendimiento.
# ===============================================================


# %% ============================================
# IMPORTACIONES Y CONFIGURACIÓN GLOBAL
# ===============================================
import yfinance as yfin  # type: ignore
import pandas as pd      # type: ignore
import numpy as np       # type: ignore
import ta                # type: ignore
import datetime
import os
import traceback
import matplotlib.pyplot as plt  # type: ignore
import warnings
from typing import Optional

warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4f}'.format


# %% ============================================
# CLASE DE SEGURIDAD Y TRAZABILIDAD
# ===============================================
class SafeLogger:
    """
    Controlador de seguridad y trazabilidad.
    Registra eventos, errores y resultados de ejecución.
    """

    def __init__(self, log_file='logs_trading.txt'):
        self.log_file = log_file
        if '/' in log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, message, level="INFO"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{level}] {timestamp} - {message}"
        print(entry)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(entry + "\n")

    def safe_run(self, func, *args, **kwargs):
        """Ejecuta una función capturando excepciones."""
        try:
            result = func(*args, **kwargs)
            self.log(f"Función '{func.__name__}' ejecutada correctamente.")
            return result
        except Exception as e:
            error_info = traceback.format_exc()
            self.log(f"Error en '{func.__name__}': {str(e)}\n{error_info}", level="ERROR")
            return None


# %% ============================================
# DESCARGA Y VALIDACIÓN DE DATOS
# ===============================================
class MarketData:
    """
    Gestiona la descarga de datos financieros desde Yahoo Finance
    y asegura su integridad antes del análisis.
    """

    def __init__(self, symbol='EUR/USD', start='2024-01-01', end='2025-10-01', logger=None):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.data = pd.DataFrame()
        self.logger = logger or SafeLogger()

    def load_data(self):
        self.logger.log(f"Descargando datos de {self.symbol} desde {self.start} hasta {self.end}...")
        try:
            df = yfin.download(self.symbol, start=self.start, end=self.end, progress=False)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            if df.empty:
                raise ValueError("No se descargaron datos o el símbolo no existe.")

            df = df.dropna().copy()
            self.data = df
            self.logger.log(f"Datos descargados correctamente: {len(df)} registros. Columnas: {list(df.columns)}")
        except Exception as e:
            self.logger.log(f"Error en la descarga: {str(e)}", level="ERROR")
        return self.data

    def get_data(self):
        if self.data.empty:
            self.logger.log("Intento de acceso a datos vacíos.", level="WARNING")
            raise ValueError("No hay datos cargados. Ejecuta primero load_data().")
        return self.data


# %% ============================================
# INDICADORES TÉCNICOS PRINCIPALES
# ===============================================
class Indicators:
    """
    Calcula indicadores técnicos clásicos:
    SMA, EMA, Bandas de Bollinger, MACD y RSI.
    """

    def __init__(self, data: pd.DataFrame, logger=None):
        self.df = data.copy()
        self.logger = logger or SafeLogger()

    def sma(self, short=5, long=20):
        self.logger.log("Calculando SMA...")
        self.df['SMA_short'] = self.df['Close'].rolling(window=short).mean()
        self.df['SMA_long'] = self.df['Close'].rolling(window=long).mean()
        return self

    def ema(self, short=5, long=20):
        self.logger.log("Calculando EMA...")
        self.df['EMA_short'] = self.df['Close'].ewm(span=short, adjust=False).mean()
        self.df['EMA_long'] = self.df['Close'].ewm(span=long, adjust=False).mean()
        return self

    def bollinger(self, window=20, dev=2):
        self.logger.log("Calculando Bandas de Bollinger...")
        bb = ta.volatility.BollingerBands(close=self.df['Close'], window=window, window_dev=dev)
        self.df['BOL_M'] = bb.bollinger_mavg()
        self.df['BOL_U'] = bb.bollinger_hband()
        self.df['BOL_D'] = bb.bollinger_lband()
        return self

    def macd(self, fast=12, slow=26, signal=9):
        self.logger.log("Calculando MACD...")
        macd = ta.trend.MACD(close=self.df['Close'], window_fast=fast, window_slow=slow, window_sign=signal)
        self.df['MACD'] = macd.macd()
        self.df['Signal_Line'] = macd.macd_signal()
        self.df['MACD_Hist'] = macd.macd_diff()
        return self

    def rsi(self, window=14):
        self.logger.log("Calculando RSI...")
        self.df['RSI'] = ta.momentum.RSIIndicator(close=self.df['Close'], window=window).rsi()
        return self

    def compute_all(self):
        """Ejecución segura y completa de todos los indicadores."""
        self.logger.safe_run(self.sma)
        self.logger.safe_run(self.ema)
        self.logger.safe_run(self.bollinger)
        self.logger.safe_run(self.macd)
        self.logger.safe_run(self.rsi)
        self.logger.log("Todos los indicadores Actualizados correctamente.")
        return self.df


# %% ============================================
# CELDA 7 — SEÑALES (EMA+MACD y BOLLINGER+RSI)
# ===============================================
class Signals:
    """Genera señales confirmadas por cruce EMA+MACD y por coincidencia Bollinger+RSI."""

    def __init__(self, df: pd.DataFrame, logger: Optional[object] = None):
        self.df = df.copy()
        self.logger = logger or SafeLogger()

    def apply_ema_macd_signals(self, short_window=5, long_window=20):
        self.logger.log("Calculando señales EMA+MACD...")
        df = self.df

        df['EMA_short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
        df['EMA_long'] = df['Close'].ewm(span=long_window, adjust=False).mean()

        df['Signals_EMA'] = 0
        for i in range(1, len(df)):
            cross_up = (df['EMA_short'].iat[i-1] < df['EMA_long'].iat[i-1]) and (df['EMA_short'].iat[i] >= df['EMA_long'].iat[i])
            cross_down = (df['EMA_short'].iat[i-1] > df['EMA_long'].iat[i-1]) and (df['EMA_short'].iat[i] <= df['EMA_long'].iat[i])
            macd_conf_buy = df['MACD'].iat[i] > df['Signal_Line'].iat[i]
            macd_conf_sell = df['MACD'].iat[i] < df['Signal_Line'].iat[i]

            if cross_up and macd_conf_buy:
                df['Signals_EMA'].iat[i] = 1
            elif cross_down and macd_conf_sell:
                df['Signals_EMA'].iat[i] = -1

        df['Buy_EMA'] = np.where(df['Signals_EMA'] == 1, df['Close'], np.nan)
        df['Sell_EMA'] = np.where(df['Signals_EMA'] == -1, df['Close'], np.nan)
        self.logger.log("Señales EMA+MACD generadas correctamente.")
        return df

    def apply_bollinger_rsi_signals(self):
        self.logger.log("Calculando señales Bollinger+RSI...")
        df = self.df
        df['Signals_BOLL'] = 0

        for i in range(1, len(df)):
            if df['Close'].iat[i] <= df['BOL_D'].iat[i] and df['RSI'].iat[i] < 30:
                df['Signals_BOLL'].iat[i] = 1
            elif df['Close'].iat[i] >= df['BOL_U'].iat[i] and df['RSI'].iat[i] > 70:
                df['Signals_BOLL'].iat[i] = -1

        df['Buy_BOLL'] = np.where(df['Signals_BOLL'] == 1, df['Close'], np.nan)
        df['Sell_BOLL'] = np.where(df['Signals_BOLL'] == -1, df['Close'], np.nan)
        self.logger.log("Señales Bollinger+RSI generadas correctamente.")
        return df

    def run_pipeline(self):
        """Ejecuta todas las señales y calcula capital simulado."""
        df = self.apply_ema_macd_signals()
        df = self.apply_bollinger_rsi_signals()
        df['Equity'] = 100 + (df['Signals_EMA'].cumsum() * df['Close'].pct_change().fillna(0))
        self.df = df
        self.logger.log("Pipeline de señales completado.")
        return df


# %% ============================================
# CELDA 8 — VISUALIZACIÓN DE RESULTADOS
# ===============================================
class Visualizer:
    """Visualizador técnico seguro con etiquetas de precio."""

    def __init__(self, df: pd.DataFrame, logger: Optional[object] = None):
        self.df = df.copy()
        self.logger = logger or SafeLogger()

    def plot_price_ema_signals(self):
        """Gráfico: Precio + EMAs + Señales EMA+MACD (etiquetas)."""
        try:
            self.logger.log("Generando gráfico EMA+MACD...")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

            ax1.plot(self.df.index, self.df['Close'], color='blue', label='Precio')
            ax1.plot(self.df.index, self.df['EMA_short'], '--', color='orange', label='EMA Corta')
            ax1.plot(self.df.index, self.df['EMA_long'], '--', color='magenta', label='EMA Larga')

            ax1.scatter(self.df.index, self.df['Buy_EMA'], color='green', marker='^', s=80, label='Compra (EMA+MACD)')
            ax1.scatter(self.df.index, self.df['Sell_EMA'], color='red', marker='v', s=80, label='Venta (EMA+MACD)')

            for idx, price in self.df['Buy_EMA'].dropna().items():
                ax1.text(idx, price, f"{price:.2f}", color='green', fontsize=8)
            for idx, price in self.df['Sell_EMA'].dropna().items():
                ax1.text(idx, price, f"{price:.2f}", color='red', fontsize=8)

            ax1.set_title("Precio + EMAs + Señales Confirmadas MACD")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.5)

            ax2.plot(self.df.index, self.df['MACD'], label='MACD', color='purple')
            ax2.plot(self.df.index, self.df['Signal_Line'], '--', label='Señal', color='orange')
            ax2.bar(self.df.index, self.df['MACD_Hist'], color='gray', alpha=0.4, label='Histograma')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.logger.log(f"Error en plot_price_ema_signals: {e}", level="ERROR")

    def plot_bollinger(self):
        """Gráfico: Bandas de Bollinger + RSI + Señales Confirmadas."""
        try:
            self.logger.log("Generando gráfico Bollinger+RSI...")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

            ax1.plot(self.df.index, self.df['Close'], color='blue', label='Precio')
            ax1.plot(self.df.index, self.df['BOL_M'], '--', color='orange', label='Media')
            ax1.plot(self.df.index, self.df['BOL_U'], ':', color='magenta', label='Banda Superior')
            ax1.plot(self.df.index, self.df['BOL_D'], ':', color='cyan', label='Banda Inferior')
            ax1.fill_between(self.df.index, self.df['BOL_D'], self.df['BOL_U'], color='gray', alpha=0.1)

            ax1.scatter(self.df.index, self.df['Buy_BOLL'], color='green', marker='^', s=80, label='Compra (BOLL+RSI)')
            ax1.scatter(self.df.index, self.df['Sell_BOLL'], color='red', marker='v', s=80, label='Venta (BOLL+RSI)')

            for idx, price in self.df['Buy_BOLL'].dropna().items():
                ax1.text(idx, price, f"{price:.2f}", color='green', fontsize=8)
            for idx, price in self.df['Sell_BOLL'].dropna().items():
                ax1.text(idx, price, f"{price:.2f}", color='red', fontsize=8)

            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.5)

            ax2.plot(self.df.index, self.df['RSI'], color='purple', label='RSI')
            ax2.axhline(70, color='red', linestyle='--')
            ax2.axhline(30, color='green', linestyle='--')
            ax2.fill_between(self.df.index, 70, 30, color='gray', alpha=0.1)
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.logger.log(f"Error en plot_bollinger: {e}", level="ERROR")

# %%
# %% 
# ANÁLISIS DE RENDIMIENTO

class PerformanceAnalyzer:
    """Analiza rendimiento, retornos y drawdown."""

    def __init__(self, df, logger=None, initial_capital=100):
        self.df = df.copy()
        self.logger = logger or SafeLogger()
        self.initial_capital = initial_capital
        self.results = {}

    def compute_returns(self):
        self.df['Returns'] = self.df['Equity'].pct_change().fillna(0)
        self.results['mean_return'] = self.df['Returns'].mean()
        self.results['volatility'] = self.df['Returns'].std()
        self.results['total_return'] = (self.df['Equity'].iloc[-1] / self.initial_capital) - 1

    def compute_drawdown(self):
        self.df['Rolling_Max'] = self.df['Equity'].cummax()
        self.df['Drawdown'] = (self.df['Equity'] - self.df['Rolling_Max']) / self.df['Rolling_Max']
        self.results['max_drawdown'] = self.df['Drawdown'].min()

    def run_full_analysis(self):
        self.compute_returns()
        self.compute_drawdown()
        plt.figure(figsize=(12, 5))
        plt.plot(self.df.index, self.df['Equity'], color='blue', linewidth=2, label='Equity')
        plt.title("Evolución del Capital")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print(f"\n $$$ Resultado ¡ Analisa ! $$$")
        print(f"Retorno total: {self.results['total_return']*100:.2f}%")
        print(f"Retorno promedio diario: {self.results['mean_return']*100:.3f}%")
        print(f"Volatilidad diaria: {self.results['volatility']*100:.3f}%")
        print(f"Máximo Drawdown: {self.results['max_drawdown']*100:.2f}%")


# %%

# %%
# PIPELINE COMPLETO Y EJECUCIÓN FINAL — Rafy (R_Profit)
# ============================================

if __name__ == "__main__":
    log = SafeLogger()
    log.log("=== INICIO PIPELINE COMPLETO ===")

    # 1️⃣ DESCARGA DE DATOS
    data = MarketData(symbol='GC=F', start='2024-01-01', end='2025-10-01', logger=log)
    df_raw = data.load_data()

    if df_raw is None or df_raw.empty:
        log.log("❌ No se descargaron datos. Abortando ejecución.", level="ERROR")
        raise SystemExit()

    log.log(f"Datos brutos cargados: {df_raw.shape}")

    # 2️⃣ CÁLCULO DE INDICADORES
    indicators = Indicators(df_raw, logger=log)
    df_indicators = indicators.compute_all()

    if df_indicators is None or df_indicators.empty:
        log.log("❌ Falló el cálculo de indicadores. Abortando ejecución.", level="ERROR")
        raise SystemExit()

    log.log(f"Indicadores calculados correctamente: {df_indicators.shape}")

    # 3️⃣ GENERACIÓN DE SEÑALES
    try:
        signals = Signals(df_indicators, logger=log)
        df_signals = signals.run_pipeline().copy()

        if df_signals is None or df_signals.empty:
            raise ValueError("El DataFrame de señales está vacío.")
        log.log(f"Señales generadas: {df_signals.shape}")
        log.log(f"Columnas finales disponibles: {list(df_signals.columns)}")

    except Exception as e:
        log.log(f"❌ Error al generar señales: {e}", level="ERROR")
        raise SystemExit()

    # 4️⃣ VISUALIZACIÓN TÉCNICA
    try:
        visual = Visualizer(df_signals, logger=log)

        log.log("Generando visualizaciones técnicas...")
        visual.plot_price_ema_signals()    # EMA + MACD + etiquetas confirmadas
        visual.plot_bollinger()            # Bollinger + RSI + etiquetas confirmadas

    except Exception as e:
        log.log(f"❌ Error al generar visualizaciones: {e}", level="ERROR")

    # 5️⃣ ANÁLISIS DE RENDIMIENTO
    try:
        analyzer = PerformanceAnalyzer(df_signals, logger=log)
        analyzer.run_full_analysis()
    except Exception as e:
        log.log(f"❌ Error al analizar el rendimiento: {e}", level="ERROR")

    # 6️⃣ TRAZABILIDAD FINAL
    log.log(f"Últimas filas del DataFrame procesado:\n{df_signals.tail(5)}")
    log.log("✅ Pipeline ejecutado correctamente.")
    log.log("=== FIN DEL PIPELINE ===")

# %%
