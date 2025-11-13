#----------------------------  -----------
# YFProvider: Proveedor de datos usando yfinance
#------------------------------------------------------------

# %%
#adapter para el flujo de datos usando yfinance; mismom formato q MT5
# indice UTC;columnas uniformes)(para backtest/offline)

from __future__ import annotations
from typing import Optional
from datetime import datetime, timedelta

def _to_yf_interval(tf: str) -> str:
    return {
        "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
        "H1": "60m", "H4": "240m", "D1": "1d",
    }[tf]

class YFProvider:
    def __init__(self):
        try:
            import yfinance as yf  # lazy import
        except Exception as e:
            raise ImportError("Instala 'yfinance' para usar YFProvider.") from e
        self._yf = yf

    def get_df(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        count: Optional[int] = None,
    ):
        import pandas as pd
        interval = _to_yf_interval(timeframe)
        if start is None and end is None and count:
            # ventana relativa (aprox): para intradía pide un rango suficiente
            end = datetime.utcnow()
            # Heurística: count velas * factor
            factor = 2 if timeframe in ("D1",) else 6
            start = end - timedelta(minutes=count * factor)  # simple aprox

        df = self._yf.download(
            tickers=symbol,
            start=start, end=end,
            interval=interval,
            auto_adjust=False, progress=False, threads=False,
        )
        if df is None or df.empty:
            # Estructura vacía coherente
            return pd.DataFrame(
                columns=["open","high","low","close","tick_volume","spread","real_volume"],
                index=pd.DatetimeIndex([], name="time", tz="UTC")
            )

        # Normaliza a nuestra convención
        df = df.rename(columns={
            "Open":"open","High":"high","Low":"low","Close":"close","Volume":"tick_volume"
        })
        df.index.name = "time"
        # Asegura tz UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        for c in ("open","high","low","close"):
            if c in df.columns: df[c] = df[c].astype("float64")
        if "tick_volume" in df.columns:
            df["tick_volume"] = df["tick_volume"].astype("int64")
        # spread/real_volume no existen en yfinance → rellena si quieres:
        if "spread" not in df.columns: df["spread"] = 0
        if "real_volume" not in df.columns: df["real_volume"] = 0

        return df[["open","high","low","close","tick_volume","spread","real_volume"]]

