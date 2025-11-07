# ======================================================
# Módulo de Red Neuronal para Pronósticos Financieros
# Archivo: red_neuronal.py
# Autor: Rafy (R_Profit) 
# Fecha: 2025-11-05
# Descripción:
#   Implementa una red neuronal MLP (feedforward) para
#   pronosticar precios usando datos exportados desde
#   pronosticos.py (CSV en /datos).
#
#   - OOP: Clase ModeloRedNeuronal (entrenar / pronosticar / evaluar)
#   - Pipeline: StandardScaler + MLPRegressor (early stopping)
#   - Seguridad: validaciones, rutas seguras, logs, UTF-8 en Windows
#   - Gráficas: guarda PNG en /reportes y muestra en pantalla
#
# Dependencias (entorno .venv311):
#   numpy, pandas, scikit-learn, matplotlib (opcional seaborn)
# ======================================================
# -*- coding: utf-8 -*-

from __future__ import annotations

# %%======================================================
# Configuración universal de codificación segura
# ======================================================
import sys
import io

# Fuerza UTF-8 en la consola de Windows (evita errores con acentos/emoji)
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
#%%
# %%======================================================
# IMPORTS PRINCIPALES / ESTÁNDAR
# ======================================================
import os
import subprocess
import traceback
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # type: ignore

# ML / Métricas
from sklearn.neural_network import MLPRegressor  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error  # type: ignore

print("[OK] Entorno validado correctamente -", sys.version)
print(f"[INFO] Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("--------------------------------------------------------------")
# %%
# %%======================================================
# Rutas seguras + Logger
# ======================================================
DATA_DIR = "datos"
LOGS_DIR = "logs"
REPORTS_DIR = "reportes"
for _d in (DATA_DIR, LOGS_DIR, REPORTS_DIR):
    os.makedirs(_d, exist_ok=True)

CSV_BASE = os.path.join(DATA_DIR, "pronosticos_base.csv")
CSV_PRED_CLASICOS = os.path.join(DATA_DIR, "predicciones_pronosticos.csv")

def registrar_error(traza: str) -> None:
    """Registra la traza de error en /logs con marca de tiempo."""
    try:
        with open(os.path.join(LOGS_DIR, "errores_red_neuronal.log"), "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} | {traza}\n")
    except Exception:
        pass  # el logging no debe romper el flujo

# %%
# %%======================================================
# Capa de seguridad e integración (opcional) con pronosticos.py
# ======================================================
try:
    # Si no existe el CSV de predicciones clásicas, intentamos generarlo.
    if not os.path.exists(CSV_PRED_CLASICOS):
        print(f"[WARN] No se encontró '{CSV_PRED_CLASICOS}'.")
        if os.path.isfile("pronosticos.py"):
            print("[INFO] Ejecutando pronosticos.py automáticamente...")
            resultado = subprocess.run(
                [sys.executable, "pronosticos.py"],
                capture_output=True,
                text=True
            )
            print("[INFO] Salida de pronosticos.py:\n", resultado.stdout)
            print(f"[INFO] Return code pronosticos.py: {resultado.returncode}")
            if resultado.returncode != 0:
                print("[ERROR] pronosticos.py no se ejecutó correctamente. Continuaré SOLO con la red neuronal.")
                if resultado.stderr:
                    registrar_error("STDERR pronosticos.py:\n" + resultado.stderr)
            else:
                print("[INFO] pronosticos.py ejecutado correctamente. CSVs generados en /datos.")
        else:
            print("[WARN] 'pronosticos.py' no está en el directorio actual. Saltando comparativa.")
    else:
        print(f"[INFO] Archivo '{CSV_PRED_CLASICOS}' detectado correctamente.")
except Exception as e:
    err = traceback.format_exc()
    print(f"[ERROR] Fallo durante la preparación: {e}")
    registrar_error(err)
    # No detenemos la ejecución: la RN puede correr sin comparativa.

# %%
# %%======================================================
# CLASE PRINCIPAL DE RED NEURONAL (OOP)
# ======================================================
class ModeloRedNeuronal:
    """
    ======================================================
    Modelo de Red Neuronal (MLP) para series que contienen
    la columna 'Close'. Mantiene una API simple:
      - entrenar()
      - pronosticar(pasos)
      - evaluar()

    Seguridad:
      - Valida existencia de 'Close'
      - Valida longitud mínima (lags)
      - Pronóstico sólo si el modelo está entrenado
    ======================================================
    """

    def __init__(self, datos: pd.DataFrame, lags: int = 7) -> None:
        if "Close" not in datos.columns:
            raise ValueError("El DataFrame debe contener una columna 'Close'")
        self.datos = datos.copy()
        self.lags = int(lags)
        self.nn: Optional[Pipeline] = None
        self.entrenado: bool = False

    def __repr__(self) -> str:
        return f"ModeloRedNeuronal(lags={self.lags}, entrenado={self.entrenado})"

    # --------------------------------------------------
    # Preparación de datos: X (ventanas), y (siguiente valor)
    # --------------------------------------------------
    def preparar_datos(self) -> Tuple[np.ndarray, np.ndarray]:
        serie = self.datos["Close"].astype(float).to_numpy()
        if len(serie) <= self.lags:
            raise ValueError("Datos insuficientes para los lags indicados.")
        X, y = [], []
        for i in range(self.lags, len(serie)):
            X.append(serie[i - self.lags : i])
            y.append(serie[i])
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float)

    # --------------------------------------------------
    # Entrenamiento (Pipeline: StandardScaler + MLPRegressor)
    # --------------------------------------------------
    def entrenar(self) -> None:
        X, y = self.preparar_datos()
        self.nn = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                max_iter=5000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=42,
                verbose=False
            ))
        ])
        self.nn.fit(X, y)
        self.entrenado = True
        print("[INFO] Red neuronal entrenada correctamente (scaler + early stopping).")

    # --------------------------------------------------
    # Evaluación simple sobre holdout temporal (último 15%)
    # --------------------------------------------------
    def evaluar(self) -> dict:
        if not self.entrenado or self.nn is None:
            raise RuntimeError("Entrena la red antes de evaluar.")
        X, y = self.preparar_datos()
        split = int(len(X) * 0.85)
        X_val, y_val = X[split:], y[split:]
        if len(X_val) == 0:
            print("[WARN] Conjunto de validación vacío; agrega más datos o ajusta lags.")
            return {"mae": float("nan"), "mape": float("nan")}
        y_hat = self.nn.predict(X_val)
        mae = float(mean_absolute_error(y_val, y_hat))
        mape = float(mean_absolute_percentage_error(y_val, y_hat))
        print(f"[METRIC] MAE={mae:.6f} | MAPE={mape:.6f}")
        return {"mae": mae, "mape": mape}

    # --------------------------------------------------
    # Pronóstico multi-paso autoregresivo
    # --------------------------------------------------
    def pronosticar(self, pasos: int = 5) -> np.ndarray:
        if not self.entrenado or self.nn is None:
            raise RuntimeError("Debe entrenarse la red antes de pronosticar.")
        ventana = self.datos["Close"].astype(float).to_numpy()[-self.lags :].tolist()
        predicciones: List[float] = []
        for _ in range(int(pasos)):
            X = np.array(ventana[-self.lags :], dtype=float).reshape(1, -1)
            pred = float(self.nn.predict(X)[0])
            predicciones.append(pred)
            ventana.append(pred)
        return np.asarray(predicciones, dtype=float)

# %%
# %%======================================================
# Funciones utilitarias de carga
# ======================================================
def cargar_dataset_base(path: str = CSV_BASE) -> pd.DataFrame:
    """Lee el CSV base con índice de fecha y columna 'Close'."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el dataset exportado: {path}")
    df = pd.read_csv(path, index_col=0)
    # Si el índice viene como string, intenta parsearlo a fecha
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    if "Close" not in df.columns:
        raise ValueError("El dataset base debe contener la columna 'Close'.")
    return df

# %%
# %%======================================================
# EJECUCIÓN PRINCIPAL — encapsulada (SIN cambiar tu lógica)
# ======================================================
def main() -> int:
    try:
        # --- Cargar dataset preparado por pronosticos.py ---
        df = cargar_dataset_base(CSV_BASE)

        # --- Entrenamiento + (opcional) evaluación ---
        modelo_nn = ModeloRedNeuronal(df, lags=7)
        modelo_nn.entrenar()
        try:
            _ = modelo_nn.evaluar()
        except Exception:
            registrar_error(traceback.format_exc())

        # --- Pronóstico ---
        pred_nn = modelo_nn.pronosticar(pasos=5)

        # ======================================================
        # Gráfica NN (guarda primero, muestra después)
        # ======================================================
        ultimos_datos = df["Close"].tail(30)
        fechas_reales = pd.to_datetime(ultimos_datos.index)
        valores_reales = ultimos_datos.values
        fechas_pronostico = pd.date_range(fechas_reales[-1], periods=len(pred_nn) + 1, freq="B")[1:]

        plt.figure(figsize=(10, 5))
        plt.plot(fechas_reales, valores_reales, label="Datos reales", color="black", linewidth=2)
        plt.plot(fechas_pronostico, pred_nn, label="Red Neuronal", color="blue", marker="D", linestyle="--")
        for i, val in enumerate(pred_nn):
            plt.text(fechas_pronostico[i], val, f"{val:.2f}", fontsize=8, ha="left", color="blue")

        plt.title("Pronóstico con Red Neuronal - Basado en pronosticos.py")
        plt.xlabel("Fecha")
        plt.ylabel("Precio de Cierre (USD)")
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.tight_layout()

        ruta_img_nn = os.path.join(REPORTS_DIR, f"nn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(ruta_img_nn, dpi=300)
        plt.show()
        print(f"[INFO] Gráfica NN guardada en: {ruta_img_nn}")

        # ======================================================
        # Gráfica comparativa final - (si están las predicciones clásicas)
        # ======================================================
        if os.path.exists(CSV_PRED_CLASICOS):
            try:
                # Intentar usar seaborn (si no está, no bloquea)
                try:
                    import seaborn as sns  # type: ignore
                    plt.style.use('seaborn-v0_8-darkgrid')
                    sns.set_context("talk")
                except Exception:
                    plt.style.use('default')

                print("[INFO] Generando gráfica comparativa de todos los modelos...")

                df_modelos = pd.read_csv(CSV_PRED_CLASICOS)
                if "Fecha" not in df_modelos.columns:
                    raise ValueError(f"'{CSV_PRED_CLASICOS}' no contiene columna 'Fecha'.")

                df_modelos["Fecha"] = pd.to_datetime(df_modelos["Fecha"])

                # Crear DataFrame de predicciones de la red neuronal
                df_red = pd.DataFrame({
                    "Fecha": pd.date_range(df_modelos["Fecha"].iloc[-1], periods=len(pred_nn) + 1, freq="B")[1:],
                    "Red_Neuronal": pred_nn
                })

                # Combinar
                df_completo = df_modelos.merge(df_red, on="Fecha", how="outer")

                # Últimos datos reales (para contexto visual)
                ultimos_datos = df["Close"].tail(30)
                fechas_reales = pd.to_datetime(ultimos_datos.index)
                valores_reales = ultimos_datos.values

                plt.figure(figsize=(12, 6))
                plt.plot(fechas_reales, valores_reales, color="black", linewidth=2.5, label="Datos Reales")

                # Estéticos por serie
                colores = {
                    "ARIMA": "#1f77b4",
                    "AutoARIMA": "#2ca02c",
                    "Suavizamiento": "#ff7f0e",
                    "Red_Neuronal": "#d62728",
                }
                marcadores = {
                    "ARIMA": "o",
                    "AutoARIMA": "s",
                    "Suavizamiento": "^",
                    "Red_Neuronal": "D",
                }

                for col in ["ARIMA", "AutoARIMA", "Suavizamiento", "Red_Neuronal"]:
                    if col in df_completo.columns:
                        fechas = pd.to_datetime(df_completo["Fecha"])
                        plt.plot(
                            fechas, df_completo[col],
                            label=col,
                            color=colores.get(col, None),
                            marker=marcadores.get(col, None),
                            linestyle="--",
                            linewidth=1.8,
                            markersize=7
                        )
                        # Mostrar valores solo para la RN (evita saturación)
                        if col == "Red_Neuronal":
                            for i, val in enumerate(df_completo[col].dropna()):
                                try:
                                    plt.text(fechas[i], val, f"{float(val):.2f}", fontsize=8, ha="left", color=colores["Red_Neuronal"])
                                except Exception:
                                    continue

                plt.title("Comparativa Final: ARIMA, AutoARIMA, Suavizamiento y Red Neuronal", fontsize=14, weight="bold")
                plt.xlabel("Fecha")
                plt.ylabel("Precio de Cierre (USD)")
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True, linestyle=":")
                plt.tight_layout()

                # Rango automático robusto (ignora NaN)
                cols_exist = [c for c in ["ARIMA", "AutoARIMA", "Suavizamiento", "Red_Neuronal"] if c in df_completo.columns]
                if cols_exist:
                    ymin = float(df_completo[cols_exist].min(numeric_only=True).min())
                    ymax = float(df_completo[cols_exist].max(numeric_only=True).max())
                    if np.isfinite(ymin) and np.isfinite(ymax):
                        plt.ylim(ymin * 0.995, ymax * 1.005)

                ruta_img_cmp = os.path.join(REPORTS_DIR, f"comparativa_todos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.savefig(ruta_img_cmp, dpi=300)
                plt.show()
                print(f"[INFO] Gráfica comparativa final guardada en: {ruta_img_cmp}")

            except Exception:
                print(f"[WARN] No se pudo generar la comparativa.")
                registrar_error(traceback.format_exc())
        else:
            print(f"[WARN] No hay comparativa: falta {CSV_PRED_CLASICOS}")

        return 0

    except Exception as e:
        traza = traceback.format_exc()
        print(f"[ERROR] Fallo durante la ejecución de red_neuronal.py: {e}")
        registrar_error(traza)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

# %%======================================================
# 