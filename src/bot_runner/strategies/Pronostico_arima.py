from __future__ import annotations
from typing import Iterable, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Tus clases reales del módulo pronósticos (no tocamos su lógica)
from pro_riesgo_finanzas.pronosticos import ModeloARIMA
from bot_runner.io_utils import ensure_dir
from bot_runner.config import BotConfig

def _tf_to_seconds(tf: str) -> int:
    return {
        "M1": 60, "M5": 5*60, "M15": 15*60, "M30": 30*60,
        "H1": 60*60, "H4": 4*60*60, "D1": 24*60*60,
    }[tf]

def _winsorize_close(df: pd.DataFrame, p: float = 0.01) -> pd.DataFrame:
    """Cap suave de outliers (1% por defecto). No cambia tendencia."""
    d = df.copy()
    if "Close" in d.columns:
        lo, hi = np.nanpercentile(d["Close"].values, [100*p, 100*(1-p)])
        d["Close"] = np.clip(d["Close"].values, lo, hi)
    return d

def _parse_order(order_param: Optional[str | Tuple[int,int,int]]) -> Optional[Tuple[int,int,int]]:
    """
    Acepta:
      - None -> usa AutoARIMA
      - tuple (p,d,q)
      - string "5,1,2" -> (5,1,2)
    """
    if order_param is None:
        return None
    if isinstance(order_param, tuple) and len(order_param) == 3:
        return int(order_param[0]), int(order_param[1]), int(order_param[2])
    if isinstance(order_param, str):
        parts = [int(x.strip()) for x in order_param.split(",") if x.strip() != ""]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
    raise ValueError(f"Parámetro 'order' inválido: {order_param!r}")

class ArimaForecastStrategy:
    """
    Strategy simple, óptima para primer bot:
      - Ventana y horizonte configurables (count, steps).
      - AutoARIMA por defecto; ARIMA fijo opcional con 'order'.
      - Limpieza robusta: normaliza 'close'->'Close', dropna y winsorize suave.
      - Fechas UTC y salida estándar en CSV.
    """
    name = "arima_auto"

    def __init__(self):
        self.symbol: str = ""
        self.timeframe: str = ""
        self.provider = None
        self.steps: int = 12
        self.count: int = 2000
        self.order: Optional[Tuple[int,int,int]] = None  # None => AutoARIMA
        self.output_filename: Optional[str] = None
        self.config: Optional[BotConfig] = None

    def prepare(
        self,
        symbol: str,
        timeframe: str,
        provider,
        steps: int = 12,
        count: int = 2000,
        order: Optional[str] = None,               # "5,1,2" ó None (auto)
        output_filename: Optional[str] = None,
        config: BotConfig = None,
        **_,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.provider = provider
        self.steps = int(steps)
        self.count = int(count)
        self.order = _parse_order(order) if order is not None else None
        self.output_filename = output_filename
        self.config = config

    def run(self, **_) -> Iterable[dict]:
        # 1) Datos (MT5 por defecto, YF si ejecutas con --no-mt5)
        df = self.provider.get_df(self.symbol, self.timeframe, count=max(self.count, 100))
        if df is None or df.empty:
            return []

        # 2) Normalización para tus modelos
        datos = df.rename(columns={"close": "Close"})  # si ya es 'Close', no cambia
        if "Close" not in datos.columns:
            return []

        datos = datos[["Close"]].dropna()
        if datos.empty:
            return []

        # 3) Limpieza robusta (cap outliers suave)
        datos = _winsorize_close(datos, p=0.01)

        # 4) Entrenamiento (AutoARIMA por defecto; ARIMA fijo opcional)
        model = ModeloARIMA(datos)
        if self.order is None:
            model.entrenar_auto()
        else:
            model.entrenar(order=self.order)

        pred = model.pronosticar(self.steps)
        if pred is None:
            return []

        # 5) Salida T+h con fechas UTC coherentes
        out = []
        last_ts = df.index[-1].to_pydatetime()  # índice del provider es UTC
        step_s = _tf_to_seconds(self.timeframe)
        for h, yhat in enumerate(np.asarray(pred, dtype=float).ravel(), 1):
            out.append({
                "ts": datetime.utcnow().isoformat() + "Z",
                "date": (last_ts + timedelta(seconds=step_s*h)).isoformat() + "Z",
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "t_plus": h,
                "yhat": float(yhat),
                "strategy": self.name if self.order is None else f"arima_fixed_{self.order}",
            })
        return out

    def post(self, rows: list[dict], **_):
        if not rows:
            return
        export_dir = "outputs"
        if self.config and getattr(self.config, "execution", None):
            export_dir = str(self.config.execution.export_dir)
        ensure_dir(export_dir)
        fname = self.output_filename or f"{self.name}_{self.symbol}_{self.timeframe}.csv"
        pd.DataFrame(rows).to_csv(f"{export_dir}/{fname}", index=False, encoding="utf-8-sig")

