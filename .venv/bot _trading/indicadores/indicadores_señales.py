# Señales indicadores SMA, EMA, Bandas de Bollinger, MACD y RSI
# ===============================================================
# Autor: Rafy (R_Profit)
# Descripción: Cálculo de indicadores, generación de señales, visualización y análisis de rendimiento.
# ===============================================================
# %% ============================================
# IMPORTACIONES Y CONFIGURACIÓN GLOBAL
# ===============================================
import yfinance as yfin # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import ta # type: ignore
import datetime
import os
import traceback
import matplotlib.pyplot as plt # type: ignore
import warnings
from typing import Optional

warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4f}'.format
# %%

# %% ============================================
# CLASE DE SEGURIDAD Y TRAZABILIDAD
# ===============================================
class SafeLogger:
    """Controlador de seguridad y trazabilidad."""
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

# %%
# %% ============================================
# DESCARGA Y VALIDACIÓN DE DATOS

class MarketData:
    """Descarga y valida datos financieros desde Yahoo Finance."""
    def __init__(self, symbol='GC=F', start='2025-01-01', end='2025-10-01', logger=None):
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
                raise ValueError("No se descargaron datos.")
            df = df.dropna().copy()
            self.data = df
            self.logger.log(f"Datos descargados correctamente: {len(df)} registros.")
        except Exception as e:
            self.logger.log(f"Error en la descarga: {str(e)}", level="ERROR")
        return self.data

    def get_data(self):
        if self.data.empty:
            self.logger.log("Intento de acceso a datos vacíos.", level="WARNING")
            raise ValueError("No hay datos cargados.")
        return self.data
# %%
# %% ============================================
# INDICADORES TÉCNICOS
# ===============================================
class Indicators:
    """Calcula SMA, EMA, Bandas de Bollinger, MACD y RSI."""
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
        self.logger.safe_run(self.sma)
        self.logger.safe_run(self.ema)
        self.logger.safe_run(self.bollinger)
        self.logger.safe_run(self.macd)
        self.logger.safe_run(self.rsi)
        self.logger.log("Todos los indicadores calculados correctamente.")
        return self.df

# %% ============================================
# CELDA 7 — SEÑALES
# ===============================================
class Signals:
    """Señales EMA+MACD y Bollinger+RSI."""
    def __init__(self, df: pd.DataFrame, logger=None):
        self.df = df.copy()
        self.logger = logger or SafeLogger()

    def apply_ema_macd_signals(self, short_window=5, long_window=20):
        self.logger.log("Generando señales EMA+MACD...")
        df = self.df
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
        df['Buy_EMA'] = np.where(df['Signals_EMA']==1, df['Close'], np.nan)
        df['Sell_EMA'] = np.where(df['Signals_EMA']==-1, df['Close'], np.nan)
        self.df = df
        self.logger.log("Señales EMA+MACD generadas.")
        return df

    def apply_bollinger_rsi_signals(self, bw_window=20, width_thresh=0.10):
        self.logger.log("Generando señales Bollinger+RSI...")
        df = self.df
        df['BandWidth'] = df['BOL_U'] - df['BOL_D']
        df['BW_MA'] = df['BandWidth'].rolling(window=bw_window, min_periods=1).mean()
        df['BW_diff'] = df['BandWidth'].diff()
        df['Widening'] = (df['BandWidth'] > df['BW_MA']*(1+width_thresh)) & (df['BW_diff']>0)
        df['Narrowing'] = (df['BandWidth'] < df['BW_MA']*(1-width_thresh)) & (df['BW_diff']<0)
        df['Signals_BOLL'] = 0
        df['Buy_BOLL'] = np.nan
        df['Sell_BOLL'] = np.nan
        df['Buy_BOLL_SQUEEZE'] = np.nan
        df['Sell_BOLL_SQUEEZE'] = np.nan
        for i in range(1,len(df)):
            price = df['Close'].iat[i]
            rsi = df['RSI'].iat[i]
            bol_u = df['BOL_U'].iat[i]
            bol_d = df['BOL_D'].iat[i]
            widening = df['Widening'].iat[i]
            narrowing = df['Narrowing'].iat[i]
            if widening and price>bol_u and rsi>50:
                df['Signals_BOLL'].iat[i]=1
                df['Buy_BOLL'].iat[i]=price
            elif widening and price<bol_d and rsi<50:
                df['Signals_BOLL'].iat[i]=-1
                df['Sell_BOLL'].iat[i]=price
            elif narrowing and price>=bol_u and rsi>70:
                df['Signals_BOLL'].iat[i]=-1
                df['Sell_BOLL_SQUEEZE'].iat[i]=price
                df['Sell_BOLL'].iat[i]=price
            elif narrowing and price<=bol_d and rsi<30:
                df['Signals_BOLL'].iat[i]=1
                df['Buy_BOLL_SQUEEZE'].iat[i]=price
                df['Buy_BOLL'].iat[i]=price
            elif price<=bol_d and rsi<30:
                df['Signals_BOLL'].iat[i]=1
                df['Buy_BOLL'].iat[i]=price
            elif price>=bol_u and rsi>70:
                df['Signals_BOLL'].iat[i]=-1
                df['Sell_BOLL'].iat[i]=price
        self.df = df
        self.logger.log("Señales Bollinger+RSI generadas.")
        return df

    def run_pipeline(self):
        self.apply_ema_macd_signals()
        self.apply_bollinger_rsi_signals()
        df = self.df
        initial_capital = 1000
        df['Equity'] = initial_capital * (1+ (df['Signals_EMA'].shift(1) * df['Close'].pct_change().fillna(0))).cumprod()
        self.df = df
        self.logger.log("Pipeline de señales completado.")
        return df

# %% ============================================
# CELDA 8 — VISUALIZACIÓN
# ===============================================
class Visualizer:
    """Visualización de EMA+MACD y Bollinger+RSI con señales y etiquetas."""
    def __init__(self, df: pd.DataFrame, logger=None):
        self.df = df.copy()
        self.logger = logger or SafeLogger()

    def plot_price_ema_signals(self):
        try:
            self.logger.log("Gráfico EMA+MACD con señales...")
            plt.close('all')
            fig, (ax1, ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True,gridspec_kw={'height_ratios':[3,1]})
            ax1.plot(self.df.index,self.df['Close'],color='blue',label='Precio')
            ax1.plot(self.df.index,self.df['EMA_short'],'--',color='orange',label='EMA Corta')
            ax1.plot(self.df.index,self.df['EMA_long'],'--',color='magenta',label='EMA Larga')
            ax1.scatter(self.df.index,self.df['Buy_EMA'],color='green',marker='^',s=80,label='Compra')
            ax1.scatter(self.df.index,self.df['Sell_EMA'],color='red',marker='v',s=80,label='Venta')
            for idx,price in self.df['Buy_EMA'].dropna().items():
                ax1.text(idx,price,f"{price:.2f}",color='green',fontsize=8)
            for idx,price in self.df['Sell_EMA'].dropna().items():
                ax1.text(idx,price,f"{price:.2f}",color='red',fontsize=8)
            ax1.legend();ax1.grid(True,linestyle='--',alpha=0.5);ax1.set_title("EMA+MACD con señales")
            ax2.plot(self.df.index,self.df['MACD'],color='purple',label='MACD')
            ax2.plot(self.df.index,self.df['Signal_Line'],'--',color='orange',label='Señal')
            ax2.bar(self.df.index,self.df['MACD_Hist'],color='gray',alpha=0.4,label='Hist')
            ax2.legend();ax2.grid(True,linestyle='--',alpha=0.5)
            plt.tight_layout();plt.show()
            self.logger.log("Gráfico EMA+MACD generado correctamente.")
        except Exception as e:
            self.logger.log(f"Error plot_price_ema_signals: {e}",level="ERROR")

    def plot_bollinger(self):
        try:
            self.logger.log("Gráfico Bollinger+RSI con BandWidth y señales...")
            plt.close('all')
            fig, (ax1,ax2,ax3)=plt.subplots(3,1,figsize=(14,10),sharex=True,gridspec_kw={'height_ratios':[3,1,0.6]})
            ax1.plot(self.df.index,self.df['Close'],label='Precio',color='blue',linewidth=1.2)
            ax1.plot(self.df.index,self.df.get('BOL_M',np.nan),'--',color='orange',label='Media')
            ax1.plot(self.df.index,self.df.get('BOL_U',np.nan),':',color='magenta',label='Banda Sup')
            ax1.plot(self.df.index,self.df.get('BOL_D',np.nan),':',color='cyan',label='Banda Inf')
            ax1.fill_between(self.df.index,self.df.get('BOL_D',0),self.df.get('BOL_U',0),color='gray',alpha=0.08)
            ax1.scatter(self.df.index,self.df.get('Buy_BOLL',np.nan),marker='^',color='green',s=90,label='Buy Widening')
            ax1.scatter(self.df.index,self.df.get('Sell_BOLL',np.nan),marker='v',color='red',s=90,label='Sell Widening')
            ax1.scatter(self.df.index,self.df.get('Buy_BOLL_SQUEEZE',np.nan),marker='P',color='lime',s=80,label='Buy Squeeze')
            ax1.scatter(self.df.index,self.df.get('Sell_BOLL_SQUEEZE',np.nan),marker='X',color='darkorange',s=80,label='Sell Squeeze')
            ax1.set_title("Bollinger+RSI con BandWidth y señales")
            ax1.legend(loc='upper left',fontsize=9);ax1.grid(True,linestyle='--',alpha=0.4)
            if 'RSI' in self.df.columns:
                ax2.plot(self.df.index,self.df['RSI'],label='RSI',color='purple')
                ax2.axhline(70,color='red',linestyle='--',alpha=0.6)
                ax2.axhline(30,color='green',linestyle='--',alpha=0.6)
                ax2.fill_between(self.df.index,70,30,color='gray',alpha=0.08)
                ax2.set_ylabel("RSI");ax2.legend(loc='upper left');ax2.grid(True,linestyle='--',alpha=0.4)
            if 'BandWidth' in self.df.columns:
                ax3.plot(self.df.index,self.df['BandWidth'],label='BandWidth')
                ax3.plot(self.df.index,self.df['BW_MA'],'--',label='BW_MA')
                ax3.fill_between(self.df.index,0,self.df['BandWidth'],where=self.df['Widening'],color='green',alpha=0.12,interpolate=True)
                ax3.fill_between(self.df.index,0,self.df['BandWidth'],where=self.df['Narrowing'],color='red',alpha=0.08,interpolate=True)
                ax3.set_ylabel("BandWidth");ax3.legend(loc='upper left');ax3.grid(True,linestyle='--',alpha=0.3)
            plt.tight_layout();plt.show()
            self.logger.log("Gráfico Bollinger+RSI generado correctamente.")
        except Exception as e:
            self.logger.log(f"Error plot_bollinger: {e}",level="ERROR")

# %% ============================================
# CELDA 10 — RENDIMIENTO
# ===============================================
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
        self.results['total_return'] = (self.df['Equity'].iloc[-1]/self.initial_capital)-1

    def compute_drawdown(self):
        self.df['Rolling_Max'] = self.df['Equity'].cummax()
        self.df['Drawdown'] = (self.df['Equity']-self.df['Rolling_Max'])/self.df['Rolling_Max']
        self.results['max_drawdown'] = self.df['Drawdown'].min()

    def run_full_analysis(self):
        self.compute_returns()
        self.compute_drawdown()
        plt.figure(figsize=(12,5))
        plt.plot(self.df.index,self.df['Equity'],color='blue',linewidth=2,label='Equity')
        plt.title("Evolución del Capital")
        plt.legend();plt.grid(True);plt.tight_layout();plt.show()
        print(f"\n📊 Resultado de desempeño:")
        print(f"Retorno total: {self.results['total_return']*100:.2f}%")
        print(f"Retorno promedio diario: {self.results['mean_return']*100:.3f}%")
        print(f"Volatilidad diaria: {self.results['volatility']*100:.3f}%")
        print(f"Máximo Drawdown: {self.results['max_drawdown']*100:.2f}%")
        
# %%

## %% ============================================
# PIPELINE COMPLETO Y EJECUCIÓN FINAL (DINÁMICO)
# ============================================

if __name__ == "__main__":
    log = SafeLogger()

    log.log("=== 🚀 INICIO PIPELINE COMPLETO DINÁMICO ===")

    # 1️⃣ DATOS – descarga dinámica
    symbol = "GC=F"  # 👈 Aquí puedes cambiar el activo (oro, plata, etc.)
    start_date = "2025-01-01"
    end_date = "2025-10-01"

    market = MarketData(symbol=symbol, start=start_date, end=end_date, logger=log)
    df_raw = market.load_data()

    if df_raw is None or df_raw.empty:
        log.log("❌ No se descargaron datos. Revisa el símbolo o las fechas.", level="ERROR")
        raise SystemExit("Ejecución detenida: no hay datos válidos.")

    # 2️⃣ INDICADORES – recalcula siempre tras descarga
    indicators = Indicators(df_raw, logger=log)
    df_indicators = indicators.compute_all()

    if df_indicators is None or df_indicators.empty:
        log.log("❌ Fallo en el cálculo de indicadores.", level="ERROR")
        raise SystemExit("Ejecución detenida: no se generaron indicadores.")

    # 3️⃣ SEÑALES – confirmaciones EMA+MACD y Bollinger+RSI
    signals = Signals(df_indicators, logger=log)
    df_signals = signals.run_pipeline()

    if df_signals is None or df_signals.empty:
        log.log("❌ Fallo en la generación de señales.", level="ERROR")
        raise SystemExit("Ejecución detenida: señales no disponibles.")

    # 4️⃣ VISUALIZACIÓN TÉCNICA
    log.log("🧭 Generando gráficos dinámicos con datos actualizados...")
    plt.close('all')  # Cierra gráficos antiguos (previene duplicados)

    visual = Visualizer(df_signals, logger=log)
    visual.plot_price_ema_signals()  # Gráfico EMA + MACD con etiquetas
    visual.plot_bollinger()          # Gráfico Bollinger + RSI dinámico

    # 5️⃣ ANÁLISIS DE RENDIMIENTO
    initial_capital = 1000  # 👈 Modifica aquí tu capital inicial
    analyzer = PerformanceAnalyzer(df_signals, logger=log, initial_capital=initial_capital)
    analyzer.run_full_analysis()

    log.log("✅ Pipeline ejecutado correctamente con datos actualizados.")
    print("\n🎯 Ejecución finalizada. Gráficos y desempeño calculados dinámicamente.\n")
# %% ============================================