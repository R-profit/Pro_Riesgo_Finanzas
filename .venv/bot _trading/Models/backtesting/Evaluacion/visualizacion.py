# visualizacion
# modules/visualization.py
import matplotlib.pyplot as plt

def plot_signals(df, title="Estrategia EMA"):
    plt.figure(figsize=(16,8))
    plt.plot(df['Close'], alpha=0.5, label='Precio')
    plt.plot(df['EMA_short'], color='green', alpha=0.4)
    plt.plot(df['EMA_long'], color='red', alpha=0.4)
    plt.scatter(df.index, df['Buy'], color='green', marker='^', label='Compra')
    plt.scatter(df.index, df['Sell'], color='red', marker='v', label='Venta')
    plt.title(title)
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.legend()
    plt.show()

def compare_equity(df1, df2, label1='Estrategia', label2='Mercado'):
    plt.figure(figsize=(16,8))
    plt.plot(df1['Equity'], label=label1)
    plt.plot(df2['Equity'], label=label2)
    plt.title('Comparación de capital')
    plt.xlabel('Fecha')
    plt.ylabel('USD')
    plt.legend()
    plt.show()

#grafico de velas japonesas de comparacion de activos
import mplfinance as mpf

def plot_candlestick(data, title="Gráfico de velas", mav=None):
    """
    Grafica velas japonesas con volumen y estilo 'yahoo'.
    """
    mpf.plot(data,
             type='candle',
             volume=True,
             figratio=(16, 8),
             style='yahoo',
             title=title,
             mav=mav)

#comparador de act con portafolio
import matplotlib.pyplot as plt

def graficar_comparacion(df1, df2, label1='Activo 1', label2='Activo 2', columna='Close'):
    """Grafica la comparación de precios entre dos activos."""
    plt.figure(figsize=(16,8))
    plt.title(f'{label1} vs {label2}')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de cierre')
    plt.plot(df1[columna], color='red')
    plt.plot(df2[columna], color='blue')
    plt.legend([label1, label2], loc='lower right')
    plt.show()

def graficar_escalados(df1, df2, label1='Activo 1', label2='Activo 2'):
    """Grafica los precios escalados a 100."""
    plt.figure(figsize=(16,8))
    plt.title(f'{label1} vs {label2} (Precios escalados)')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de cierre escalado (base 100)')
    plt.plot(df1['Close_100'], color='red')
    plt.plot(df2['Close_100'], color='blue')
    plt.legend([label1, label2], loc='lower right')
    plt.show()
#%%
# %% 
# VISUALIZACIÓN TÉCNICA DEL ACTIVO
# ============================================
# Autor: Rafy (R_Profit)
# Descripción:
# Muestra la evolución del precio con sus indicadores técnicos:
# EMAs, Bandas de Bollinger, MACD y las señales Buy/Sell generadas.
# ============================================

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

class Visualizer:
    """
    Clase encargada de generar visualizaciones técnicas del activo,
    manteniendo coherencia con el modelo de seguridad y trazabilidad.
    """

    def __init__(self, df, logger=None):
        self.df = df.copy()
        self.logger = logger or SafeLogger()

    def plot_price_ema_signals(self):
        """Gráfico principal: Precio + EMAs + Señales Buy/Sell"""
        try:
            self.logger.log("Generando gráfico Precio + EMAs + Señales Buy/Sell...")
            fig, ax = plt.subplots(figsize=(14, 6))

            # Precio base
            ax.plot(self.df.index, self.df['Close'], label='Precio Cierre', color='blue', linewidth=1.5, alpha=0.6)

            # Medias móviles
            ax.plot(self.df.index, self.df['EMA_short'], label='EMA Corta (5)', color='orange', linestyle='--', linewidth=1.2)
            ax.plot(self.df.index, self.df['EMA_long'], label='EMA Larga (20)', color='magenta', linestyle='--', linewidth=1.2)

            # Señales Buy/Sell
            ax.scatter(self.df.index, self.df['Buy'], marker='^', color='green', label='Compra', s=80, alpha=0.9)
            ax.scatter(self.df.index, self.df['Sell'], marker='v', color='red', label='Venta', s=80, alpha=0.9)

            # Configuración estética
            ax.set_title("Estrategia EMA — Precio, Cruces y Señales", fontsize=13, fontweight='bold')
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Precio")
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

            self.logger.log("Gráfico Precio + EMAs generado correctamente.")
        except Exception as e:
            self.logger.log(f"Error al generar gráfico Precio + EMAs: {str(e)}", level="ERROR")

    def plot_bollinger(self):
        """Gráfico de Bandas de Bollinger"""
        try:
            self.logger.log("Generando gráfico de Bandas de Bollinger...")
            fig, ax = plt.subplots(figsize=(14, 6))

            ax.plot(self.df.index, self.df['Close'], label='Precio Cierre', color='blue', linewidth=1.5)
            ax.plot(self.df.index, self.df['BOL_M'], label='Media', color='orange', linestyle='--')
            ax.plot(self.df.index, self.df['BOL_U'], label='Banda Superior', color='green', linestyle=':')
            ax.plot(self.df.index, self.df['BOL_D'], label='Banda Inferior', color='red', linestyle=':')

            ax.fill_between(self.df.index, self.df['BOL_D'], self.df['BOL_U'], color='gray', alpha=0.1)
            ax.set_title("Bandas de Bollinger — Volatilidad del Activo", fontsize=13, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

            self.logger.log("Gráfico de Bandas de Bollinger generado correctamente.")
        except Exception as e:
            self.logger.log(f"Error al generar gráfico Bollinger: {str(e)}", level="ERROR")

    def plot_macd(self):
        """Gráfico MACD + Señal + Histograma"""
        try:
            self.logger.log("Generando gráfico MACD...")
            fig, ax = plt.subplots(figsize=(12, 4))

            ax.plot(self.df.index, self.df['MACD'], label='MACD', color='purple', linewidth=1.5)
            ax.plot(self.df.index, self.df['Signal_Line'], label='Línea de Señal', color='orange', linestyle='--')
            ax.bar(self.df.index, self.df['MACD_Hist'], label='Histograma', color='gray', alpha=0.4)

            ax.set_title("MACD — Momentum del Activo", fontsize=13, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

            self.logger.log("Gráfico MACD generado correctamente.")
        except Exception as e:
            self.logger.log(f"Error al generar gráfico MACD: {str(e)}", level="ERROR")
# %%
# %% 
# EJECUCIÓN DEL OBJETIVO 1
# ============================================
# Genera las visualizaciones técnicas del activo
# utilizando la clase Visualizer definida en la celda anterior.

visual = Visualizer(df_signals, logger=log)

visual.plot_price_ema_signals()
visual.plot_bollinger()
visual.plot_macd()
# %%
# %%
# CELDA 10 — VISUALIZACIÓN DEL RENDIMIENTO (OBJETIVO 2)
# ========================================================
# Autor: Rafy (R_Profit)
# Descripción:
# Analiza y visualiza el desempeño del bot:
# curva de capital (Equity), retornos diarios y drawdown.
# ========================================================

import matplotlib.pyplot as plt
import numpy as np

class PerformanceAnalyzer:
    """
    Clase encargada del análisis del rendimiento y riesgo
    del portafolio resultante del backtest.
    """

    def __init__(self, df, logger=None, initial_capital=100):
        self.df = df.copy()
        self.logger = logger or SafeLogger()
        self.initial_capital = initial_capital
        self.results = {}

    def compute_returns(self):
        """Calcula los retornos porcentuales diarios."""
        try:
            self.logger.log("Calculando retornos porcentuales diarios...")
            self.df['Returns'] = self.df['Equity'].pct_change().fillna(0)
            self.results['mean_return'] = self.df['Returns'].mean()
            self.results['volatility'] = self.df['Returns'].std()
            self.results['total_return'] = (self.df['Equity'].iloc[-1] / self.initial_capital) - 1
            self.logger.log("Cálculo de retornos completado correctamente.")
        except Exception as e:
            self.logger.log(f"Error al calcular retornos: {str(e)}", level="ERROR")

    def compute_drawdown(self):
        """Calcula el drawdown (pérdida máxima desde un pico)."""
        try:
            self.logger.log("Calculando drawdown...")
            self.df['Rolling_Max'] = self.df['Equity'].cummax()
            self.df['Drawdown'] = (self.df['Equity'] - self.df['Rolling_Max']) / self.df['Rolling_Max']
            self.results['max_drawdown'] = self.df['Drawdown'].min()
            self.logger.log("Drawdown calculado correctamente.")
        except Exception as e:
            self.logger.log(f"Error al calcular drawdown: {str(e)}", level="ERROR")

    def plot_equity_curve(self):
        """Grafica la evolución del capital."""
        try:
            self.logger.log("Graficando curva de capital (Equity Curve)...")
            plt.figure(figsize=(12, 5))
            plt.plot(self.df.index, self.df['Equity'], label='Curva de Capital', color='blue', linewidth=2)
            plt.title("Evolución del Capital (Backtest)")
            plt.xlabel("Fecha")
            plt.ylabel("Capital")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()
            self.logger.log("Curva de capital graficada correctamente.")
        except Exception as e:
            self.logger.log(f"Error al graficar curva de capital: {str(e)}", level="ERROR")

    def plot_returns(self):
        """Grafica los retornos porcentuales diarios."""
        try:
            self.logger.log("Graficando retornos diarios...")
            plt.figure(figsize=(12, 4))
            plt.bar(self.df.index, self.df['Returns'] * 100, color='gray', alpha=0.6)
            plt.title("Retornos Diarios (%)")
            plt.xlabel("Fecha")
            plt.ylabel("Retorno (%)")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
            self.logger.log("Gráfico de retornos generado correctamente.")
        except Exception as e:
            self.logger.log(f"Error al graficar retornos: {str(e)}", level="ERROR")

    def plot_drawdown(self):
        """Grafica el drawdown histórico."""
        try:
            self.logger.log("Graficando drawdown...")
            plt.figure(figsize=(12, 4))
            plt.fill_between(self.df.index, self.df['Drawdown'] * 100, color='red', alpha=0.4)
            plt.title("Drawdown Histórico (%)")
            plt.xlabel("Fecha")
            plt.ylabel("Drawdown (%)")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
            self.logger.log("Gráfico de drawdown generado correctamente.")
        except Exception as e:
            self.logger.log(f"Error al graficar drawdown: {str(e)}", level="ERROR")

    def summary(self):
        """Muestra resumen estadístico de la estrategia."""
        try:
            self.logger.log("Mostrando resumen estadístico del rendimiento:")
            print("\n📈 === RESUMEN DE DESEMPEÑO ===")
            print(f"Retorno total: {self.results['total_return']*100:.2f}%")
            print(f"Retorno promedio diario: {self.results['mean_return']*100:.3f}%")
            print(f"Volatilidad diaria: {self.results['volatility']*100:.3f}%")
            print(f"Máximo Drawdown: {self.results['max_drawdown']*100:.2f}%")
            print("================================\n")
        except Exception as e:
            self.logger.log(f"Error al mostrar resumen: {str(e)}", level="ERROR")

    def run_full_analysis(self):
        """Ejecuta todo el pipeline de análisis y visualización."""
        self.logger.log("=== INICIO DEL ANÁLISIS DE RENDIMIENTO ===")
        self.compute_returns()
        self.compute_drawdown()
        self.plot_equity_curve()
        self.plot_returns()
        self.plot_drawdown()
        self.summary()
        self.logger.log("=== FIN DEL ANÁLISIS DE RENDIMIENTO ===")

# %%
# %%
# %% 
# EJECUCIÓN DEL ANÁLISIS DE RENDIMIENTO
# ===================================================
# Ejecuta el análisis completo del rendimiento y drawdown

analyzer = PerformanceAnalyzer(df_signals, logger=log, initial_capital=100)
analyzer.run_full_analysis()
# %%