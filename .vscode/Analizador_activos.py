#analista de precios y retornos de activos financieros
# ===========================================
# Módulo de Análisis Financiero GC=F
# ===========================================
# Autor: Rafael Medina Arias (R_Profit)
# Función: descarga, analiza y visualiza comportamiento de activos financieros
# Para integración dentro del bot de trading
# ===========================================

# %%
import yfinance as yf # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import mplfinance as mpf # type: ignore
import ta # type: ignore
from datetime import datetime

# %% seguridad_helpers
# Helpers mínimos: rate limiter, validación y fingerprint (no intrusivo)
import time
import threading
import hashlib
import re
from typing import Optional
import pandas as pd

class _RateLimiter:
    """Rate limiter muy ligero por instancia."""
    def __init__(self, capacity=5, refill_time=60):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_time = refill_time
        self.lock = threading.Lock()
        self.last_refill = time.time()

    def _refill(self):
        now = time.time()
        if now - self.last_refill > self.refill_time:
            with self.lock:
                self.tokens = self.capacity
                self.last_refill = now

    def allow(self) -> bool:
        self._refill()
        with self.lock:
            if self.tokens > 0:
                self.tokens -= 1
                return True
            return False

def _validate_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        raise ValueError("Symbol debe ser string")
    s = symbol.strip()
    if not re.fullmatch(r"[A-Za-z0-9\-\._=]+", s):
        raise ValueError("Symbol contiene caracteres inválidos")
    return s

def _validate_date_iso(date_str: str) -> str:
    from datetime import datetime
    try:
        datetime.fromisoformat(date_str)
        return date_str
    except Exception:
        raise ValueError("Fecha debe tener formato ISO YYYY-MM-DD")

def _dataframe_fingerprint(df: pd.DataFrame) -> Optional[str]:
    try:
        h = hashlib.sha256()
        h.update(df.head(5).to_csv(index=True).encode("utf-8"))
        h.update(",".join(map(str, df.columns)).encode("utf-8"))
        return h.hexdigest()[:16]
    except Exception:
        return None
    
# %% end seguridad_helpers

# %%
class AnalizadorFinanciero:
    """
    Para descargar, analizar y graficar datos de un activo financiero.
    Diseñada para integrarse fácilmente en un bot de trading.
    """

    def __init__(self, symbol='GC=F', start='2024-01-01', end='2025-10-01'):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.data = pd.DataFrame()
        self.retorno_anual = None
        self.volatilidad_anual = None
        self.resumen = None
     # Seguridad / estado mínimo (no intrusivo)
        self._rate_limiter = _RateLimiter(capacity=5, refill_time=60)
        self._last_fingerprint = None
    print()
# %%    
    # Validación simple de fechas
    def _fechas_validas(self):
        try:
            s = datetime.fromisoformat(self.start)
            e = datetime.fromisoformat(self.end)
            return s < e
        except Exception:
            return False
# %%
    def descargar_datos(self):
        """
        Procesamiento de datos usando yfinance.
        Normaliza MultiIndex y nombres de columnas, limpia datos.
        Añade validaciones de entrada, rate limit por instancia y fingerprint.
        """
        # Validar fechas
        if not self._fechas_validas():
            print(f"[Qubit!] Fechas inválidas: start={self.start}, end={self.end}. Asegura start < end y formato YYYY-MM-DD.")
            self.data = pd.DataFrame()
            return

        try:
            # validaciones ligeras de parámetros
            try:
                self.symbol = _validate_symbol(self.symbol)
                self.start = _validate_date_iso(self.start)
                self.end = _validate_date_iso(self.end)
            except ValueError as ve:
                print(f"[Qubit!] Parámetros inválidos: {ve}")
                self.data = pd.DataFrame()
                return

            if not self._rate_limiter.allow():
                print("[Qubit!] Rate limit alcanzado. Intenta más tarde.")
                self.data = pd.DataFrame()
                return

            df = yf.download(
                tickers=self.symbol,
                start=self.start,
                end=self.end,
                auto_adjust=True,
                progress=False
            )

            if df is None or df.empty:
                print(f"[Qubit!] No se descargaron datos para {self.symbol} en el rango {self.start} - {self.end}.")
                self.data = pd.DataFrame()
                return

            # Normalizar MultiIndex -> ('Adj Close','GC=F') -> columnas simples
            if isinstance(df.columns, pd.MultiIndex):
                # caso común en tu output: nivel 0 = Price, nivel 1 = Ticker
                if self.symbol in df.columns.get_level_values(1):
                    cols = [c for c in df.columns if isinstance(c, tuple) and c[1] == self.symbol]
                    if cols:
                        new_df = df.loc[:, cols].copy()
                        new_df.columns = [c[0] for c in cols]
                        df = new_df
                else:
                    # caso alternativo: (ticker, field)
                    if self.symbol in df.columns.get_level_values(0):
                        try:
                            df = df.xs(self.symbol, axis=1, level=0, drop_level=True)
                        except Exception:
                            df = df.copy()
                            df.columns = [c[1] if isinstance(c, tuple) and len(c) > 1 else c for c in df.columns]
                    else:
                        # aplanar multiindex a strings
                        df = df.copy()
                        df.columns = ['_'.join(map(str, c)).strip() if isinstance(c, tuple) else str(c) for c in df.columns]

            # normalizar 'Adj_Close' -> 'Adj Close'
            if 'Adj_Close' in df.columns and 'Adj Close' not in df.columns:
                df.rename(columns={'Adj_Close': 'Adj Close'}, inplace=True)

            # mapear nombres comunes
            col_map_candidates = {
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'adj close': 'Adj Close', 'adj_close': 'Adj Close',
                'volume': 'Volume'
            }
            normalized = {}
            for c in df.columns:
                cl = str(c).strip().lower()
                if cl in col_map_candidates:
                    normalized[c] = col_map_candidates[cl]
            if normalized:
                df = df.rename(columns=normalized)

            # si no aparecen Close ni Adj Close, devolver para diagnóstico
            if 'Close' not in df.columns and 'Adj Close' not in df.columns:
                print(f"[Qubit!] Diagnóstico columnas recibidas: {repr(df.columns)}")
                try:
                    df.head(10).to_csv('diagnostico_gc_columns.csv')
                    print("[Qubit!] Se guardó diagnostico_gc_columns.csv con las primeras filas.")
                except Exception:
                    pass
                self.data = df
                return

            df.dropna(how='all', inplace=True)
            if df.empty:
                print(f"[Qubit!] Tras limpieza el DataFrame quedó vacío para {self.symbol}.")
                self.data = pd.DataFrame()
                return

            # fingerprint para auditabilidad mínima
            try:
                fp = _dataframe_fingerprint(df)
                self._last_fingerprint = fp
            except Exception:
                self._last_fingerprint = None

            self.data = df

        except Exception as e:
            print(f"[Qubit!] No se pudo descargar {self.symbol}: {e}")
            self.data = pd.DataFrame()
            
# %%

# %%

    def _obtener_series_close(self):
        """
        Devuelve una pd.Series 1-D con los precios de cierre.
        Maneja casos donde 'Close' sea DataFrame de una sola columna.
        """
        if self.data.empty:
            return pd.Series(dtype=float)

        if 'Close' not in self.data.columns:
            # intentar con 'Adj Close' o 'Adj_Close' o variaciones
            for alt in ['Adj Close', 'Adj_Close', 'close']:
                if alt in self.data.columns:
                    series = self.data[alt]
                    return series.squeeze()
            # sin columna de cierre
            raise KeyError("No existe la columna 'Close' ni 'Adj Close' en los datos descargados.")

        close_col = self.data['Close']
        if isinstance(close_col, pd.DataFrame):
            return close_col.squeeze()
        return close_col
# %%
    
    def agregar_indicadores(self):
        """
        Agrega indicadores técnicos: RSI, EMA_20, EMA_50.
        Calcula RSI sobre una Series 1-D y reindexa las series resultantes en el DataFrame.
        """
        if self.data.empty:
            print("[INFO] No hay datos para agregar indicadores.")
            return

        try:
            close_series = self._obtener_series_close()
        except KeyError as ke:
            print(f"[Qubit!] {ke}")
            return

        close_series = close_series.dropna()
        if close_series.empty:
            print("[Qubit!] Serie de cierre vacía después de dropna().")
            return

        # RSI
        try:
            rsi = ta.momentum.RSIIndicator(close_series).rsi()
            self.data.loc[rsi.index, 'RSI'] = rsi
        except Exception as e:
            print(f"[Qubit!] Al calcular RSI: {e}")

        # EMAs
        try:
            ema20 = close_series.ewm(span=20, adjust=False).mean()
            ema50 = close_series.ewm(span=50, adjust=False).mean()
            self.data.loc[ema20.index, 'EMA_20'] = ema20
            self.data.loc[ema50.index, 'EMA_50'] = ema50
        except Exception as e:
            print(f"[Qubit!] Al calcular EMAs: {e}")

# %%

    def calcular_retorno_volatilidad(self):
        """
        Calcula retornos simples y logarítmicos, volatilidad anualizada y retorno anualizado.
        Prefiere 'Adj Close' si está disponible.
        """
        if self.data.empty:
            print("[INFO] No hay datos para calcular retornos.")
            return

        if 'Adj Close' in self.data.columns:
            precio = self.data['Adj Close']
        elif 'Close' in self.data.columns:
            precio = self.data['Close']
        else:
            print("[Qubit!] No existe 'Adj Close' ni 'Close' para calcular retornos.")
            return

        precio = precio.squeeze().dropna()
        if precio.empty:
            print("[Qubit!] Serie de precios vacía para cálculo de retornos.")
            return

        self.data.loc[precio.index, 'Return'] = precio.pct_change()
        self.data.loc[precio.index, 'Log_Return'] = np.log(precio / precio.shift(1))

        log_ret = self.data['Log_Return'].dropna()
        if log_ret.empty:
            print("[Qubit!] No hay log returns válidos para calcular estadísticas.")
            return

        self.volatilidad_anual = log_ret.std() * np.sqrt(252)
        self.retorno_anual = log_ret.mean() * 252
        self.resumen = self.data.describe()
    
# %%
        
    def graficar_precios(self):
        """
        Grafica la serie de precios de cierre (Adj Close preferido).
        """
        if self.data.empty:
            print("[INFO] No hay datos para graficar precios.")
            return

        if 'Adj Close' in self.data.columns:
            serie = self.data['Adj Close']
        elif 'Close' in self.data.columns:
            serie = self.data['Close']
        else:
            print("[INFO] No hay columna de cierre para graficar.")
            return

        plt.figure(figsize=(10, 4))
        plt.plot(serie.index, serie.values)
        plt.title(f'Precio de Cierre - {self.symbol}')
        plt.ylabel('Precio ($)')
        plt.xlabel('Fecha')
        plt.grid(True)
        plt.show()

# %%
    def graficar_candlestick(self):
        """
        Grafica velas japonesas con mplfinance; comprueba que existan Open, High, Low, Close.
        """
        if self.data.empty:
            print("[INFO] No hay datos para graficar velas.")
            return

        required = {'Open', 'High', 'Low', 'Close'}
        if not required.issubset(set(self.data.columns)):
            print("[Qubit!] DataFrame no contiene columnas mínimas para velas (Open,High,Low,Close).")
            return

        try:
            mpf.plot(self.data.tail(120), type='candle', style='charles',
                     title=f'Últimos 120 días - {self.symbol}', volume=True)
        except Exception as e:
            print(f"[Qubit!] al graficar candlestick: {e}")

# %%

    def ejecutar_analisis(self, graficar=True):
        """
        Pipeline completo: descarga, agrega indicadores, calcula estadísticas y grafica.
        Retorna (dataframe, retorno_anual, volatilidad_anual, resumen)
        """
        self.descargar_datos()
        if self.data.empty:
            print("[INFO] Análisis detenido: no hay datos.")
            return None

        self.agregar_indicadores()
        self.calcular_retorno_volatilidad()

        if graficar:
            self.graficar_precios()
            self.graficar_candlestick()

        if self.retorno_anual is not None and self.volatilidad_anual is not None:
            print(f"🔹 Retorno Anualizado: {self.retorno_anual:.2%}")
            print(f"🔹 Volatilidad Anualizada: {self.volatilidad_anual:.2%}")

        if self.resumen is not None:
            cols = [c for c in ['Adj Close', 'Return', 'Log_Return'] if c in self.resumen.columns]
            if cols:
                print("\nResumen estadístico:")
                print(self.resumen[cols])

        return self.data, self.retorno_anual, self.volatilidad_anual, self.resumen

# %%
# Test local (si se ejecuta como script). En el bot simplemente importa la clase y crea instancias.
if __name__ == "__main__":
    analista = AnalizadorFinanciero(symbol='GC=F', start='2024-01-01', end='2025-10-01')
    analista.ejecutar_analisis()
    print("columns:", list(analista.data.columns))
    print(analista.data.head(5))
    
# %%

#SUGERENCIA ESTRATEGICA
# señal simple de cruse de media moviles y RSI
#Así el módulo queda listo para integrarse como módulo analítico dentro del bot de trading que estás construyendo.
 
 #def generar_senal(self):
   # """Ejemplo de señal simple basada en cruce EMA y RSI."""
   # if self.data.empty:
   #     return None
   # df = self.data.dropna(subset=['EMA_20', 'EMA_50', 'RSI'])
   # if df.empty:
   #     return None
   # ultima = df.iloc[-1]
   # if ultima['EMA_20'] > ultima['EMA_50'] and ultima['RSI'] < 70:
   #     return "COMPRA"
   # elif ultima['EMA_20'] < ultima['EMA_50'] and ultima['RSI'] > 30:
   #     return "VENTA"
   # else:
#    return "ESPERAR"
# %%

# %%
#Sugerencia funcionalidad extra
 # para la estabilidad del bot y concistencia del RSI mejor instalar estas librerias especificas
 # pip install yfinance==0.2.50 ta==0.11.0 mplfinance==0.12.10b0
# %%
