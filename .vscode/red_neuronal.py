# %% ======================================================
# Módulo de Red Neuronal para Pronósticos Financieros
# ======================================================
# Autor: Rafy (R_Profit)
# Fecha: 2025-11-04
# Descripción:
# Este módulo implementa una red neuronal tipo feedforward
# (MLPRegressor) para pronosticar precios usando datos
# exportados desde pronosticos.py
# ======================================================
# -*- coding: utf-8 -*-
# ======================================================
# Configuración universal de codificación segura
# ======================================================
import sys, io

# Fuerza UTF-8 en la consola de Windows (para evitar errores con emojis o acentos)
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

## %% ======================================================
# Capa de seguridad e integración automática con pronósticos
# ======================================================

import os
import subprocess
import sys
import traceback
from datetime import datetime

print("[OK] Entorno validado correctamente -", sys.version)
print(f"[INFO] Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("--------------------------------------------------------------")

try:
    # Verificar si existe el archivo de predicciones
    if not os.path.exists("predicciones_pronosticos.csv"):
        print("[WARN] No se encontró 'predicciones_pronosticos.csv'. Ejecutando pronosticos.py automáticamente...")
        resultado = subprocess.run(
            [sys.executable, "pronosticos.py"],
            capture_output=True,
            text=True
        )
        print("[INFO] Salida de pronosticos.py:\n", resultado.stdout)

        if resultado.returncode != 0:
            print("[ERROR] pronosticos.py no se ejecutó correctamente.")
            print(resultado.stderr)
            raise RuntimeError("Error en la ejecución de pronosticos.py")
        else:
            print("[INFO] pronosticos.py ejecutado correctamente. Archivo CSV generado.")
    else:
        print("[INFO] Archivo 'predicciones_pronosticos.csv' detectado correctamente.")

except Exception as e:
    print(f"[ERROR] Fallo durante la ejecución de red_neuronal.py: {e}")
    with open("logs/errores_red_neuronal.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} - {str(e)}\n")
        traceback.print_exc(file=f)
    sys.exit(1)

# %%

# %% ======================================================
# IMPORTS PRINCIPALES
# ======================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
from sklearn.neural_network import MLPRegressor # type: ignore

# ======================================================
# CLASE PRINCIPAL DE RED NEURONAL
# ======================================================
class ModeloRedNeuronal:
    def __init__(self, datos: pd.DataFrame, lags: int = 7):
        self.datos = datos
        self.lags = lags
        self.nn = None
        self.entrenado = False

    def preparar_datos(self):
        X, y = [], []
        serie = self.datos['Close'].values
        for i in range(self.lags, len(serie)):
            X.append(serie[i-self.lags:i])
            y.append(serie[i])
        return np.array(X), np.array(y)

    def entrenar(self):
        X, y = self.preparar_datos()
        self.nn = MLPRegressor(hidden_layer_sizes=(20,), activation='relu',
                               max_iter=2000, random_state=42)
        self.nn.fit(X, y)
        self.entrenado = True
        print("[INFO] Red neuronal entrenada correctamente.")

    def pronosticar(self, pasos=5):
        if not self.entrenado:
            raise RuntimeError("Debe entrenarse la red antes de pronosticar.")
        ultimos = self.datos['Close'].values[-self.lags:].tolist()
        predicciones = []
        for _ in range(pasos):
            X = np.array(ultimos[-self.lags:]).reshape(1, -1)
            pred = self.nn.predict(X)[0]
            predicciones.append(pred)
            ultimos.append(pred)
        return np.array(predicciones)
    
# %%

# %%======================================================
# EJECUCIÓN PRINCIPAL
# ======================================================
if __name__ == "__main__":
    try:
        ruta_csv = os.path.join("datos", "pronosticos_base.csv")
        if not os.path.exists(ruta_csv):
            raise FileNotFoundError(f"No se encontró el dataset exportado: {ruta_csv}")

        df = pd.read_csv(ruta_csv, index_col=0)

        modelo_nn = ModeloRedNeuronal(df, lags=7)
        modelo_nn.entrenar()
        pred_nn = modelo_nn.pronosticar(5)

        # --- Gráfica ---
        ultimos_datos = df['Close'][-30:]
        fechas_reales = pd.to_datetime(ultimos_datos.index)
        valores_reales = ultimos_datos.values
        fechas_pronostico = pd.date_range(fechas_reales[-1], periods=6, freq='B')[1:]

        plt.figure(figsize=(10, 5))
        plt.plot(fechas_reales, valores_reales, label='Datos reales', color='black', linewidth=2)
        plt.plot(fechas_pronostico, pred_nn, label='Red Neuronal', color='blue', marker='D', linestyle='--')
        for i, val in enumerate(pred_nn):
            plt.text(fechas_pronostico[i], val, f"{val:.2f}", fontsize=8, ha='left', color='blue')

        plt.title("Pronóstico con Red Neuronal - Basado en Pronósticos.py")
        plt.xlabel("Fecha")
        plt.ylabel("Precio de Cierre (USD)")
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        plt.show()
        
        # ======================================================
        # Gráfica comparativa final - Todos los modelos
        # ======================================================
        import seaborn as sns # type: ignore

        print("[INFO] Generando gráfica comparativa de todos los modelos...")

        # Cargar predicciones de modelos clásicos
        ruta_pred_modelos = os.path.join("datos", "predicciones_pronosticos.csv")
        if not os.path.exists(ruta_pred_modelos):
            raise FileNotFoundError("No se encontró 'predicciones_pronosticos.csv'. Ejecuta primero pronosticos.py")

        df_modelos = pd.read_csv(ruta_pred_modelos)
        df_modelos['Fecha'] = pd.to_datetime(df_modelos['Fecha'])

        # Crear DataFrame de predicciones de la red neuronal
        df_red = pd.DataFrame({
            'Fecha': pd.date_range(df_modelos['Fecha'].iloc[-1], periods=len(pred_nn)+1, freq='B')[1:],
            'Red_Neuronal': pred_nn
        })

        # Combinar ambos conjuntos
        df_completo = df_modelos.merge(df_red, on='Fecha', how='outer')

        # Últimos datos reales (para contexto visual)
        ultimos_datos = df['Close'][-30:]
        fechas_reales = pd.to_datetime(ultimos_datos.index)
        valores_reales = ultimos_datos.values

        # --- Configuración visual ---
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_context("talk")

        plt.figure(figsize=(12, 6))
        plt.plot(fechas_reales, valores_reales, color='black', linewidth=2.5, label='Datos Reales')

        # --- Colores y marcadores diferenciados ---
        colores = {
            'ARIMA': '#1f77b4',
            'AutoARIMA': '#2ca02c',
            'Suavizamiento': '#ff7f0e',
            'Red_Neuronal': '#d62728'
        }
        marcadores = {
            'ARIMA': 'o',
            'AutoARIMA': 's',
            'Suavizamiento': '^',
            'Red_Neuronal': 'D'
        }

        # --- Graficar todas las curvas ---
        for col in ['ARIMA', 'AutoARIMA', 'Suavizamiento', 'Red_Neuronal']:
            if col in df_completo.columns:
                fechas = pd.to_datetime(df_completo['Fecha'])
                plt.plot(fechas, df_completo[col],
                         label=col,
                         color=colores[col],
                         marker=marcadores[col],
                         linestyle='--',
                         linewidth=1.8,
                         markersize=7)
                # Mostrar valores sobre cada punto
                for i, val in enumerate(df_completo[col]):
                    plt.text(fechas[i], val, f"{val:.2f}",
                             fontsize=8, ha='left', color=colores[col])

        # --- Ajustes visuales ---
        plt.title("Comparativa Final: ARIMA, AutoARIMA, Suavizamiento y Red Neuronal",
                  fontsize=14, weight='bold')
        plt.xlabel("Fecha")
        plt.ylabel("Precio de Cierre (USD)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, linestyle=':')
        plt.tight_layout()

        # --- Ajuste automático del rango ---
        min_val = df_completo[['ARIMA', 'AutoARIMA', 'Suavizamiento', 'Red_Neuronal']].min().min()
        max_val = df_completo[['ARIMA', 'AutoARIMA', 'Suavizamiento', 'Red_Neuronal']].max().max()
        plt.ylim(min_val * 0.995, max_val * 1.005)

        # Mostrar y guardar
        plt.show()

        os.makedirs("reportes", exist_ok=True)
        ruta_img = os.path.join("reportes", f"comparativa_todos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(ruta_img, dpi=300)
        print(f"[INFO] Gráfica comparativa final guardada en: {ruta_img}")


    except Exception as e:
        err_trace = traceback.format_exc()
        print(f"[ERROR] Fallo durante la ejecución de red_neuronal.py: {e}")
        registrar_error(err_trace)


