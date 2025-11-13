# %% ======================================================
# Utilidades de visualización para pronósticos
# ======================================================
#apta para bot, CLI, web(puede guardar y devolver bytes)
#importa , matplolib dentro del modulo viz no en tu core de modelos

from __future__ import annotations
from typing import Dict, Optional, Tuple
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_pronosticos_zoom(
    serie_close: pd.Series,
    predicciones: Dict[str, np.ndarray],
    fechas_pred: pd.DatetimeIndex,
    title: str = "Pronósticos",
    save_path: Optional[str] = None,
    return_bytes: bool = False,
) -> Optional[bytes]:
    """
    Dibuja 'serie_close' (última ventana) y varias curvas de pronóstico.
    - predicciones: {"ARIMA": array, "AutoARIMA": array, "Suavizamiento": array, ...}
    - fechas_pred: DatetimeIndex para los puntos pronosticados
    - save_path: si se define, guarda PNG
    - return_bytes: si True, retorna PNG en bytes (para web)
    """
    if serie_close.empty:
        raise ValueError("serie_close está vacía.")
    if len(fechas_pred) == 0:
        raise ValueError("fechas_pred está vacía.")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    # Datos reales (última ventana)
    ax.plot(serie_close.index, serie_close.values, label="Datos reales",
            color="black", linewidth=2)

    # Pronósticos
    for nombre, yhat in predicciones.items():
        if yhat is None:
            continue
        yhat = np.asarray(yhat, dtype=float).ravel()
        if yhat.size != len(fechas_pred):
            continue
        ax.plot(fechas_pred, yhat, label=nombre, linestyle="--", marker="o")

    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio de Cierre")
    ax.grid(True, linestyle=":")
    ax.legend(loc="best")
    fig.tight_layout()

    # Guardar a disco si se pide
    if save_path:
        fig.savefig(save_path, dpi=160)

    # Retornar bytes (para app/web)
    if return_bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        return buf.getvalue()

    plt.show()
    return None

