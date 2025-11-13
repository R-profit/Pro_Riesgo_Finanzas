# %% Utilidades de E/S seguras y logging
from __future__ import annotations
import os
import logging
from typing import Optional

# %% Rutas y carpetas
def ensure_dir(path: str) -> str:
    """Crea el directorio si no existe. Devuelve la ruta normalizada."""
    norm = os.path.normpath(path)
    os.makedirs(norm, exist_ok=True)
    return norm

# %% Logging centralizado
def setup_logger(name: str = "bot", log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Crea logger con salida a consola y archivo rotativo ligero.
    - No falla si la ruta no existe (la crea).
    - Nivel por defecto INFO; usa DEBUG para mayor detalle.
    """
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # ya configurado

    logger.setLevel(level)

    # Consola
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # Archivo
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
