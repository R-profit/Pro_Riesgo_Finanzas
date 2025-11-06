# %% ======================================================
# Módulo de Modelos de Pronóstico (ARIMA y Suavizamiento)
# ======================================================
# Autor: Rafy (R_Profit)
# Fecha: 2025-11-02
# Descripción:
# Este módulo define clases orientadas a objetos para el entrenamiento y
# predicción de series temporales con modelos ARIMA, AutoARIMA y
# Suavizamiento Exponencial. Preparado para integrarse al Bot de Trading.
# ======================================================
# %%
# ======================================================
# CAPA DE SEGURIDAD Y VALIDACIÓN DE ENTORNO
# ======================================================

import sys
import os
import platform
import traceback
from datetime import datetime

def registrar_error(error_msg: str):
    """Guarda errores críticos en un log estándar"""
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "errores_pronosticos.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")

try:
    # Versión mínima de Python requerida
    if sys.version_info < (3, 11):
        raise EnvironmentError("Versión de Python no compatible. Se requiere 3.11 o superior.")

    # Validar entorno virtual activo
    if sys.prefix == sys.base_prefix:
        raise EnvironmentError("No se detectó un entorno virtual activo (.venv311).")

    # Validar dependencias críticas
    required = ["numpy", "pandas", "statsmodels", "pmdarima", "yfinance", "matplotlib"]
    for pkg in required:
        __import__(pkg)

    print(f"[OK] Entorno validado correctamente - Python {platform.python_version()}")
    print(f"[INFO] Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("--------------------------------------------------------------")

except Exception as e:
    err_trace = traceback.format_exc()
    print(f"[ERROR CRÍTICO] Fallo en la capa de seguridad: {e}")
    registrar_error(err_trace)
    sys.exit(1)

# %%

# %%======================================================
# IMPORTS PRINCIPALES DEL MÓDULO
# ======================================================

import warnings
import numpy as np # type: ignore
import pandas as pd # type: ignore
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA # type: ignore
from pmdarima import auto_arima # type: ignore
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing # type: ignore

print ("todas la librerias se instalaron correctamente")
warnings.filterwarnings("ignore")
# %%


# %% ======================================================
# Clase base para control de seguridad y estructura común
# ======================================================
class ModeloBase:
    def __init__(self, datos: pd.DataFrame):
        # Aplanar columnas si yfinance devuelve MultiIndex
        if isinstance(datos.columns, pd.MultiIndex):
            datos.columns = ['_'.join([str(c) for c in col if c]) for col in datos.columns]

        # Si la columna correcta no se llama exactamente "Close"
        posibles = [c for c in datos.columns if 'Close' in c]
        if posibles:
            datos.rename(columns={posibles[0]: 'Close'}, inplace=True)

        if not isinstance(datos, pd.DataFrame):
            raise TypeError("Se esperaba un DataFrame con una columna 'Close', no una Serie.")
        if 'Close' not in datos.columns:
            raise ValueError(f"El DataFrame debe contener una columna 'Close'. Columnas encontradas: {list(datos.columns)}")
        if datos['Close'].isnull().any():
            print("⚠️ Hay datos nulos en la columna 'Close', se eliminarán automáticamente.")
            datos = datos.dropna(subset=['Close'])
        self.datos = datos.reset_index(drop=True)
        self.modelo = None
        self.entrenado = False
        
    def verificar_entrenamiento(self):
        if not self.entrenado:
            raise RuntimeError("El modelo debe ser entrenado antes de pronosticar.")    
# %%
# %%======================================================
# Clase para modelos ARIMA y AutoARIMA
# ======================================================
class ModeloARIMA(ModeloBase):
    def entrenar(self, order=(7, 1, 3)):
        """Entrena un modelo ARIMA clásico"""
        try:
            self.modelo = ARIMA(self.datos['Close'], order=order).fit()
            self.entrenado = True
            print(f"[INFO] Modelo ARIMA entrenado correctamente con orden {order}.")
        except Exception as e:
            print(f"[ERROR] Fallo en entrenamiento ARIMA: {e}")

    def entrenar_auto(self):
        """Entrena automáticamente el modelo ARIMA óptimo"""
        try:
            self.modelo = auto_arima(self.datos['Close'], seasonal=False, trace=False)
            self.entrenado = True
            print("[INFO] Modelo AutoARIMA entrenado correctamente.")
        except Exception as e:
            print(f"[ERROR] Fallo en entrenamiento AutoARIMA: {e}")

    def pronosticar(self, pasos=5):
        """Genera pronósticos a futuro"""
        self.verificar_entrenamiento()
        try:
            pred = self.modelo.forecast(pasos)
            return np.array(pred)
        except Exception as e:
            print(f"[ERROR] Fallo en pronóstico ARIMA: {e}")
            return None

# %%
# %%======================================================
# Clase para Suavizamiento Exponencial
# ======================================================
class ModeloSuavizamiento(ModeloBase):
    def entrenar(self):
        """Entrena diferentes variantes de suavizamiento exponencial"""
        try:
            self.ses = SimpleExpSmoothing(self.datos['Close']).fit()
            self.holt = ExponentialSmoothing(self.datos['Close'], trend='add').fit()
            self.hw_damped = ExponentialSmoothing(
                self.datos['Close'], trend='add', damped_trend=True
            ).fit()
            self.ets = ExponentialSmoothing(self.datos['Close'], trend='add').fit()
            self.entrenado = True
            print("[INFO] Modelos de suavizamiento entrenados correctamente.")
        except Exception as e:
            print(f"[ERROR] Fallo en entrenamiento de suavizamiento: {e}")

    def pronosticar(self, pasos=5, metodo='holt'):
        """Permite elegir el modelo de suavizamiento para pronóstico"""
        self.verificar_entrenamiento()
        try:
            modelos = {
                'ses': self.ses,
                'holt': self.holt,
                'hw_damped': self.hw_damped,
                'ets': self.ets
            }
            if metodo not in modelos:
                raise ValueError(f"Método '{metodo}' no válido. Opciones: {list(modelos.keys())}")
            pred = modelos[metodo].forecast(pasos)
            return np.array(pred)
        except Exception as e:
            print(f"[ERROR] Fallo en pronóstico {metodo}: {e}")
            return None

# %%
# %%======================================================
## ======================================================
# EJECUCIÓN PRINCIPAL CON CAPA DE SEGURIDAD GLOBAL
# ======================================================
if __name__ == "__main__":
    import os
    import logging
    from datetime import datetime
    import yfinance as yf
    import matplotlib.pyplot as plt

    # -----------------------------
    # CONFIGURACIÓN DE SEGURIDAD
    # -----------------------------
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "errores_pronosticos.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.ERROR,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("🛡️ Iniciando ejecución segura del módulo de pronósticos...\n")

    try:
        # Descarga de datos
        df = yf.download("GC=F", start="2024-01-01", end="2025-01-01")[['Close']]
        print("[INFO] Datos descargados correctamente.")

        # --- Entrenamiento ARIMA ---
        arima_model = ModeloARIMA(df)
        arima_model.entrenar(order=(5, 1, 2))
        pred_arima = arima_model.pronosticar(5)
        print("[INFO] Pronóstico ARIMA completado.")

        # --- Entrenamiento AutoARIMA ---
        auto_model = ModeloARIMA(df)
        auto_model.entrenar_auto()
        pred_auto = auto_model.pronosticar(5)
        print("[INFO] Pronóstico AutoARIMA completado.")

        # --- Entrenamiento Suavizamiento ---
        suav = ModeloSuavizamiento(df)
        suav.entrenar()
        pred_ses = suav.pronosticar(5, metodo='ses')
        print("[INFO] Pronóstico Suavizamiento completado.")

        # --- Bloque de visualización (usa el nuevo gráfico con zoom y anotaciones) ---
        from datetime import datetime
        import matplotlib.dates as mdates
        import numpy as np
        import pandas as pd

        window = 12
        pasos = 5
        fmt_val = "{:,.2f}"

        ultimos_datos = df['Close'][-window:]
        fechas_reales = ultimos_datos.index
        valores_reales = ultimos_datos.values
        fecha_ultimo = fechas_reales[-1]
        fechas_pronostico = pd.date_range(fecha_ultimo, periods=pasos+1, freq='B')[1:]

        plt.figure(figsize=(10, 5))
        plt.plot(fechas_reales, valores_reales, label='Datos reales', color='black', linewidth=2)

        def plot_pronostico(fechas, valores, label, marcador):
            if valores is not None:
                plt.plot(fechas, valores, label=label, linestyle='--', marker=marcador)
                for x, y in zip(fechas, valores):
                    plt.annotate(fmt_val.format(y),
                                 xy=(x, y),
                                 xytext=(0, 6),
                                 textcoords="offset points",
                                 ha='center', va='bottom',
                                 fontsize=8,
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        plot_pronostico(fechas_pronostico, pred_arima, 'ARIMA', 'o')
        plot_pronostico(fechas_pronostico, pred_auto, 'AutoARIMA', 's')
        plot_pronostico(fechas_pronostico, pred_ses, 'Suavizamiento', '^')

        plt.title('Zoom de Pronóstico - Oro (GC=F)')
        plt.xlabel('Fecha')
        plt.ylabel('Precio de Cierre (USD)')
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        plt.show()

        print("\n✅ Ejecución completada sin errores.\n")

    except Exception as e:
        error_msg = f"[ERROR CRÍTICO] {type(e).__name__}: {str(e)}"
        print(error_msg)
        logging.error(error_msg, exc_info=True)

    finally:
        print("🧩 Proceso finalizado. Logs disponibles en:", log_path)

# %% ======================================================
# %%
    # ======================================================
    # Gráfico zoom + anotaciones de pronóstico (mejor legibilidad)
    # ======================================================
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Parám. de visualización (adapta según tu metodología)
    window = 12     # cuántos días reales previos mostrar (ajusta a tu preferencia)
    pasos = 5       # horizonte de pronóstico (ya lo usas)
    fmt_val = "{:,.2f}"  # formato para anotaciones (dos decimales, separador de miles)

    # --- Preparar datos reales: últimos 'window' días ---
    ultimos_datos = df['Close'][-window:]  # últimos N datos
    fechas_reales = ultimos_datos.index
    valores_reales = ultimos_datos.values

    # --- Fechas para pronóstico (días hábiles) ---
    fecha_ultimo = fechas_reales[-1]
    fechas_pronostico = pd.date_range(fecha_ultimo, periods=pasos+1, freq='B')[1:]

    # --- Intentar obtener intervalos de confianza para cada modelo ---
    def obtener_confianza(modelo, pasos):
        """Devuelve (pred, lower, upper) si es posible, o (pred, None, None)."""
        try:
            # Para statsmodels ARIMAResults
            if hasattr(modelo, "get_forecast"):
                res = modelo.get_forecast(steps=pasos)
                pred = res.predicted_mean
                ci = res.conf_int(alpha=0.05)
                lower = ci.iloc[:, 0].values
                upper = ci.iloc[:, 1].values
                return np.array(pred), lower, upper
            # Para pmdarima AutoARIMA
            if hasattr(modelo, "predict") and "return_conf_int" in modelo.predict.__code__.co_varnames:
                pred, conf = modelo.predict(n_periods=pasos, return_conf_int=True, alpha=0.05)
                lower = conf[:, 0]
                upper = conf[:, 1]
                return np.array(pred), lower, upper
            # Si no soporta conf_int
            pred = modelo.forecast(pasos) if hasattr(modelo, "forecast") else None
            return np.array(pred), None, None
        except Exception:
            # Fallback: devuelve pred (si existe) y sin bandas
            try:
                pred = modelo.predict(n_periods=pasos) if hasattr(modelo, "predict") else modelo.forecast(pasos)
                return np.array(pred), None, None
            except Exception:
                return None, None, None

    pred_a, low_a, up_a = (None, None, None)
    pred_b, low_b, up_b = (None, None, None)
    pred_c, low_c, up_c = (None, None, None)

    if pred_arima is not None:
        # Si tu pred_arima es un array simple (por ejemplo: np.array), intentamos obtener de self.modelo
        try:
            pred_a, low_a, up_a = obtener_confianza(arima_model.modelo, pasos)
        except Exception:
            pred_a = pred_arima
    if pred_auto is not None:
        try:
            pred_b, low_b, up_b = obtener_confianza(auto_model.modelo, pasos)
        except Exception:
            pred_b = pred_auto
    if pred_ses is not None:
        try:
            # Para suavizamiento, modelos fitted tienen .forecast que a veces soporta conf_int no siempre.
            pred_c, low_c, up_c = obtener_confianza(suav.holt, pasos)
        except Exception:
            pred_c = pred_ses

    # Si no pudimos obtener pred desde modelos, usamos los arrays ya calculados
    if pred_a is None and pred_arima is not None:
        pred_a = np.array(pred_arima)
    if pred_b is None and pred_auto is not None:
        pred_b = np.array(pred_auto)
    if pred_c is None and pred_ses is not None:
        pred_c = np.array(pred_ses)

    # --- Graficar con zoom en la ventana (últimos window días + pronóstico) ---
    fig, ax = plt.subplots(figsize=(11, 5), dpi=100)

    ax.plot(fechas_reales, valores_reales, label='Datos reales', color='black', linewidth=2)

    # plot pronósticos y bandas de confianza si existen
    def plot_with_conf(ax, fechas, pred, low, up, label, marker):
        if pred is None:
            return
        ax.plot(fechas, pred, label=label, marker=marker, linestyle='--')
        if low is not None and up is not None:
            ax.fill_between(fechas, low, up, alpha=0.15)

        # anotar cada punto del pronóstico con su valor
        for x, y in zip(fechas, pred):
            ax.annotate(fmt_val.format(y), xy=(x, y), xytext=(0, 6),
                        textcoords="offset points", ha='center', va='bottom', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, linewidth=0.3))

    plot_with_conf(ax, fechas_pronostico, pred_a, low_a, up_a, 'ARIMA', 'o')
    plot_with_conf(ax, fechas_pronostico, pred_b, low_b, up_b, 'AutoARIMA', 's')
    plot_with_conf(ax, fechas_pronostico, pred_c, low_c, up_c, 'Suavizamiento', '^')

    # --- Formato eje X: mostrar ticks legibles (últimos window días + pronóstico) ---
    # Construimos rango mínimo y máximo para el eje x
    x_min = fechas_reales[0]
    x_max = fechas_pronostico[-1]
    ax.set_xlim([x_min - pd.Timedelta(days=0.5), x_max + pd.Timedelta(days=0.5)])

    # Locators y formatter para fechas
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.title('Zoom: Pronóstico sobre últimos datos - Oro (GC=F)')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de Cierre (USD)')
    plt.grid(True, linestyle=':')
    plt.legend(loc='upper left')
    plt.tight_layout()

    # Opcional: guardar la figura
    out_path = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(os.path.join(out_path, f"pronostico_GC_F_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"), dpi=200)

    plt.show()
# %%

