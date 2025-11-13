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

# ======================================================
# Configuración universal de codificación segura
# ======================================================
import sys, io

# Fuerza UTF-8 en la consola de Windows (para evitar errores con emojis o acentos)
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')



# %%
# ======================================================
# CAPA DE SEGURIDAD Y VALIDACIÓN DE ENTORNO
# ======================================================

import os
import traceback
from datetime import datetime

def registrar_error(error_msg: str):
    """Guarda errores críticos en un log estándar (sin datos sensibles)."""
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "errores_pronosticos.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")

def validar_entorno(strict: bool = False, verbose: bool = False) -> bool:
    """
    Valida versión de Python, venv activo y dependencias críticas.
    - strict=True: relanza excepción (para tests o CI).
    - verbose=True: imprime mensajes de estado NO sensibles (para uso manual).
    No llama sys.exit().
    """
    try:
        # Versión mínima
        if sys.version_info < (3, 11):
            raise EnvironmentError("Se requiere Python 3.11 o superior.")

        # Venv activo
        if sys.prefix == sys.base_prefix:
            raise EnvironmentError("No se detectó un entorno virtual activo.")

        # Dependencias (yfinance opcional)
        required = ["numpy", "pandas", "statsmodels", "pmdarima", "matplotlib"]
        for pkg in required:
            __import__(pkg)

        if verbose:
            print("[OK] Entorno validado. Dependencias críticas cargadas.")

        return True

    except Exception:
        err_trace = traceback.format_exc()
        registrar_error(err_trace)
        if verbose:
            # Mensaje genérico (sin detalles internos)
            print("[ERROR] Validación de entorno falló. Revisa logs/errores_pronosticos.log")
        if strict:
            raise
        return False

# %%

# %%======================================================
# IMPORTS PRINCIPALES DEL MÓDULO
# ======================================================

import warnings
import numpy as np # type: ignore
import pandas as pd # type: ignore

# import yfinance as yf opcional, ahora del provider

from statsmodels.tsa.arima.model import ARIMA # type: ignore
from pmdarima import auto_arima # type: ignore
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing # type: ignore

warnings.filterwarnings("ignore")
# %%
# %% ======================================================
# Helpers de preprocesado ligero
# ======================================================
from typing import Any, Optional, Dict, Tuple

def _winsorize_series(s: pd.Series, p: float = 0.01) -> pd.Series:
    """Cap suave de outliers (1% por defecto)."""
    if s.empty:
        return s
    lo, hi = np.nanpercentile(s.values, [100*p, 100*(1-p)])
    return s.clip(lower=lo, upper=hi)

def _z_critical(alpha: float = 0.05) -> float:
    """Aproximación rápida del cuantíl normal (95% -> 1.96)."""
    # Para no meter dependencias nuevas; suficiente para IC aproximados.
    return 1.96 if abs(alpha - 0.05) < 1e-6 else 1.645 if alpha <= 0.1 else 2.576
# %%

# %% ======================================================
# Clase base para control de seguridad y estructura común
# ======================================================
class ModeloBase:
    def __init__(self, datos: pd.DataFrame):
        if not isinstance(datos, pd.DataFrame):
            raise TypeError("Se esperaba un DataFrame con columna 'Close' o 'close'.")

        # Aplanar MultiIndex si aparece
        if isinstance(datos.columns, pd.MultiIndex):
            datos = datos.copy()
            datos.columns = ["_".join([str(c) for c in col if c]) for col in datos.columns]

        # Normaliza nombre de columna objetivo
        cols_l = {c.lower(): c for c in datos.columns}
        if "close" in cols_l and "Close" not in datos.columns:
            datos = datos.rename(columns={cols_l["close"]: "Close"})

        if "Close" not in datos.columns:
            raise ValueError(f"Falta columna 'Close'/'close'. Columnas: {list(datos.columns)}")

        # Índice temporal: intenta forzarlo a DatetimeIndex si no lo es
        if not isinstance(datos.index, pd.DatetimeIndex):
            datos = datos.copy()
            datos.index = pd.to_datetime(datos.index, errors="coerce")

        # Limpieza mínima
        datos = datos[["Close"]].dropna()
        if datos.empty:
            raise ValueError("No hay datos válidos en 'Close'.")

        # Conserva índice temporal (no resetea)
        self.datos = datos
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
        """Genera pronósticos a futuro (compatible statsmodels / pmdarima)."""
        self.verificar_entrenamiento()
        try:
            m = self.modelo
            # statsmodels ARIMAResults → forecast(steps=...)
            if hasattr(m, "forecast"):
                return np.array(m.forecast(steps=int(pasos)))
            # pmdarima ARIMA → predict(n_periods=...)
            if hasattr(m, "predict"):
                return np.array(m.predict(n_periods=int(pasos)))
            raise AttributeError("El modelo no expone forecast/predict compatibles.")
        except Exception as e:
            print(f"[ERROR] Fallo en pronóstico ARIMA/AutoARIMA: {e}")
            return None

# %%
# %%======================================================
# Clase para Suavizamiento Exponencial
# ======================================================
#añadimos winsoriza(robustes)
#prueba multiples variantes y elegimos por IAC
# estacionalidad opcional (si procede por tf)

class ModeloSuavizamiento(ModeloBase):
    """
    - Auto-selección del mejor método por AIC entre:
        * 'ses' (nivel)
        * 'holt' (tendencia aditiva)
        * 'hw_damped' (tendencia amortiguada)
        * 'ets_add' (nivel+tendencia+ estacionalidad aditiva, si se indica periodo)
        * 'ets_mul' (nivel+tendencia+ estacionalidad multiplicativa, si se indica periodo)
    - Winsorize previo para robustez.
    - Intervalos de confianza aproximados por residuales (asunción normal).
    """
    def __init__(self, datos: pd.DataFrame):
        super().__init__(datos)
        self.modelos: Dict[str, Any] = {}
        self.mejor: Optional[str] = None
        self._resid_std: Optional[float] = None

    def entrenar(
        self,
        seasonal_periods: Optional[int] = None,   # p.ej. 5 para D1 (semanal laboral) o 96 para M15 intradía
        auto_select: bool = True,
        damped_trend: bool = True,
        winsorize_p: float = 0.01,
    ):
        """
        Entrena variantes y selecciona la de menor AIC si auto_select=True.
        Si seasonal_periods es None, no prueba estacionalidad.
        """
        y = self.datos["Close"].astype("float64")
        if winsorize_p and 0.0 < winsorize_p < 0.5:
            y = _winsorize_series(y, p=winsorize_p)

        candidatos: Dict[str, Any] = {}

        # SES
        try:
            candidatos["ses"] = SimpleExpSmoothing(y).fit()
        except Exception:
            pass

        # Holt (tendencia aditiva)
        try:
            candidatos["holt"] = ExponentialSmoothing(y, trend="add").fit()
        except Exception:
            pass

        # Holt amortiguado
        if damped_trend:
            try:
                candidatos["hw_damped"] = ExponentialSmoothing(
                    y, trend="add", damped_trend=True
                ).fit()
            except Exception:
                pass

        # ETS con estacionalidad (si se especifica)
        if seasonal_periods and seasonal_periods > 1:
            # ETS aditivo
            try:
                candidatos["ets_add"] = ExponentialSmoothing(
                    y, trend="add", seasonal="add", seasonal_periods=seasonal_periods
                ).fit()
            except Exception:
                pass
            # ETS multiplicativo (siempre que y>0; winsorize ya ayudó)
            if (y > 0).all():
                try:
                    candidatos["ets_mul"] = ExponentialSmoothing(
                        y, trend="add", seasonal="mul", seasonal_periods=seasonal_periods
                    ).fit()
                except Exception:
                    pass

        if not candidatos:
            raise RuntimeError("No se pudo ajustar ningún modelo de suavizamiento.")

        # Auto-selección por AIC
        if auto_select:
            aics = {k: getattr(m, "aic", np.inf) for k, m in candidatos.items()}
            self.mejor = min(aics, key=aics.get)
        else:
            # por defecto si no auto-select: usa 'holt' si está, si no SES
            self.mejor = "holt" if "holt" in candidatos else "ses"

        self.modelos = candidatos

        # Desviación de residuales del modelo elegido (para IC aproximados)
        try:
            resid = pd.Series(self.modelos[self.mejor].resid).dropna()
            self._resid_std = float(resid.std(ddof=1)) if len(resid) > 5 else None
        except Exception:
            self._resid_std = None

        self.entrenado = True
        # No prints aquí (silencioso para producción)

    def pronosticar(self, pasos: int = 5, metodo: str = "auto") -> Optional[np.ndarray]:
        """Pronóstico puntual (vector)."""
        self.verificar_entrenamiento()
        nombre = self.mejor if metodo == "auto" else metodo
        mdl = self.modelos.get(nombre)
        if mdl is None:
            raise ValueError(f"Método '{nombre}' no entrenado. Disponibles: {list(self.modelos.keys())}")
        try:
            pred = mdl.forecast(int(pasos))
            return np.asarray(pred, dtype=float)
        except Exception:
            return None

    def pronosticar_con_intervalo(
        self, pasos: int = 5, metodo: str = "auto", alpha: float = 0.05
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Devuelve (pred, lower, upper). IC aproximados con residuales ~ N(0, sigma^2).
        Para ETS en statsmodels no hay confint nativo; este es un proxy conservador.
        """
        yhat = self.pronosticar(pasos=pasos, metodo=metodo)
        if yhat is None:
            return None, None, None
        if not self._resid_std:
            return yhat, None, None
        z = _z_critical(alpha=alpha)
        err = self._resid_std
        lower = yhat - z * err
        upper = yhat + z * err
        return yhat, lower, upper


# %%
# %%======================================================
# EJECUCIÓN PRINCIPAL CON CAPA DE SEGURIDAD GLOBAL
# ======================================================
# MODO BOT (por defecto): no ejecuta demo, lo usa el runner con MT5
#python -m bot_runner.main once --config .\configs\bot.yaml --strategy bot_runner.strategies.pronostico_arima:ArimaForecastStrategy --params "{\"symbol\":\"XAUUSD\",\"timeframe\":\"M15\",\"steps\":12}"

# ACTIVAR DEMO por variable de entorno (PowerShell)
#$env:PRONOSTICOS_DEMO="1"; python .\src\pro_riesgo_finanzas\pronosticos.py

# ACTIVAR DEMO por CLI (sin tocar variables)
#python .\src\pro_riesgo_finanzas\pronosticos.py --demo


# Modo BOT por defecto + Switch de DEMO
# ======================================================
if __name__ == "__main__":
    """
    Por defecto, este módulo NO ejecuta nada (modo BOT).
    Activa la DEMO sólo si:
      - pones PRONOSTICOS_DEMO=1 en el entorno, o
      - llamas: python pronosticos.py --demo
    """
    import os, sys

    # --- Switch por ENV (tiene prioridad) ---
    DEMO_ENV = os.getenv("PRONOSTICOS_DEMO", "").strip() == "1"

    # --- Switch por CLI (sólo si ejecutas directamente este .py) ---
    DEMO_ARG = ("--demo" in sys.argv)

    if DEMO_ENV or DEMO_ARG:
        # ------------- DEMO OPCIONAL (segura) -------------
        import logging, traceback
        from datetime import datetime
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        # 1) Validación (silenciosa por defecto)
        validar_entorno(strict=False, verbose=True)

        # 2) Logging a archivo
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            filename=os.path.join("logs", "errores_pronosticos.log"),
            level=logging.ERROR,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 3) DEMO SIEMPRE CON YF (no MT5 aquí para no tocar broker)
        try:
            from bot_runner.providers.yf_provider import YFProvider
            provider = YFProvider()
            symbol = "GC=F"
            timeframe = "D1"
            df = provider.get_df(symbol, timeframe, start=datetime(2024, 1, 1), end=datetime(2025, 1, 1))
            if df is None or df.empty:
                print("[WARN] Provider devolvió DataFrame vacío.")
                raise SystemExit(0)

            datos = df.rename(columns={"close": "Close"})
            if "Close" not in datos.columns:
                raise ValueError("El DataFrame no tiene columna 'Close'/'close'.")

            # 4) Entrenamiento + pronósticos (igual que tu flujo)
            arima_model = ModeloARIMA(datos); arima_model.entrenar_auto()
            pred_auto  = arima_model.pronosticar(5)

            suav = ModeloSuavizamiento(datos)
            suav.entrenar(seasonal_periods=5, auto_select=True, damped_trend=True, winsorize_p=0.01)
            pred_suav = suav.pronosticar(5, metodo="auto")

            # 5) Visualización corta
            window, pasos = 12, 5
            ult = datos["Close"][-window:]
            fechas_reales = ult.index
            fechas_pronostico = pd.date_range(fechas_reales[-1], periods=pasos+1, freq="B")[1:]

            plt.figure(figsize=(10, 5))
            plt.plot(fechas_reales, ult.values, label="Reales", color="black", linewidth=2)
            if pred_auto is not None:
                plt.plot(fechas_pronostico, pred_auto, label="AutoARIMA", linestyle="--", marker="o")
            if pred_suav is not None:  # <--- antes era pred_ses
                plt.plot(fechas_pronostico, pred_suav, label="Suavizamiento (auto)", linestyle="--", marker="^")
            plt.title(f"DEMO Pronósticos — {symbol} ({timeframe})")
            plt.legend(); plt.grid(True, linestyle=":")
            os.makedirs("outputs", exist_ok=True)
            plt.savefig(os.path.join("outputs", f"demo_pronosticos_{symbol}_{timeframe}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"), dpi=150)
            plt.show()

            # 6) Export simple para RN
            try:
                arr_auto = np.asarray(pred_auto, dtype=float).ravel() if pred_auto is not None else None
                arr_suav = np.asarray(pred_suav,  dtype=float).ravel() if pred_suav  is not None else None
                if arr_auto is None or arr_suav is None:
                    raise ValueError("Predicción inválida para export.")

                df_pred = pd.DataFrame({
                    "Fecha": pd.to_datetime(fechas_pronostico),
                    "AutoARIMA": arr_auto,
                    "Suavizamiento": arr_suav
                })
                os.makedirs("datos", exist_ok=True)
                ruta_csv = os.path.join("datos", "predicciones_pronosticos_demo.csv")
                df_pred.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
                print(f"[OK] Exportado: {ruta_csv}")
            except Exception:
                registrar_error(traceback.format_exc())
                print("[ERROR] Export CSV DEMO — revisa logs.")
        except SystemExit:
            pass
        except Exception:
            registrar_error(traceback.format_exc())
            print("[ERROR] DEMO falló; revisa logs/errores_pronosticos.log")
    else:
        # ------------- MODO BOT (predeterminado) -------------
        # No hacer nada: el runner cargará estas clases y les pasará datos MT5.
        # Esto evita ejecuciones accidentales y mantiene la seguridad en producción.
        pass

# %% ======================================================
#