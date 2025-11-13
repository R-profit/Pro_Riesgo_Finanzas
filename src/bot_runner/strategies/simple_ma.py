# %% Estrategia: Cruce de Medias Móviles Simple
from __future__ import annotations
import pandas as pd

class SimpleMAStrategy:
    """
    Cruce de medias móviles (rápida vs lenta).
    - Señal = 1 cuando MA_fast > MA_slow, -1 cuando MA_fast < MA_slow, 0 en otro caso.
    - Position = última señal arrastrada (ffill) para backtest básico.
    """

    def __init__(self, fast: int = 10, slow: int = 30):
        if fast >= slow:
            raise ValueError("'fast' debe ser menor que 'slow'.")
        self.fast = int(fast)
        self.slow = int(slow)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Close" not in df:
            raise KeyError("El DataFrame debe contener la columna 'Close'.")

        out = df.copy()
        out["ma_fast"] = out["Close"].rolling(self.fast, min_periods=self.fast).mean()
        out["ma_slow"] = out["Close"].rolling(self.slow, min_periods=self.slow).mean()

        out["signal"] = 0
        valid = out["ma_fast"].notna() & out["ma_slow"].notna()
        out.loc[valid & (out["ma_fast"] > out["ma_slow"]), "signal"] = 1
        out.loc[valid & (out["ma_fast"] < out["ma_slow"]), "signal"] = -1

        out["position"] = out["signal"].replace(to_replace=0, method="ffill").fillna(0)
        return out

