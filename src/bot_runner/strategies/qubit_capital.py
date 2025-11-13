# %% ======================================================
# Estrategia compuesta "QubitCapital"
# ======================================================
# Combina:
#  - Pronósticos (ARIMA/AutoARIMA/ETS) → sesgo direccional (forecast filter)
#  - Indicadores/Señales (EMA+MACD, Bollinger+RSI) → timing de entradas
#  - Comparador Activo vs Portafolio → filtro de régimen (outperformance)
#
# Contrato: compatible con loader (prepare/run/post).
# Produce DataFrame con columnas estándar que el runner sabe exportar.
# ======================================================

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Iterable

import numpy as np            # type: ignore
import pandas as pd           # type: ignore

# Módulos existentes (tu base)
from pro_riesgo_finanzas.pronosticos import ModeloARIMA, ModeloSuavizamiento
from pro_riesgo_finanzas.signals_ta import Indicators, Signals
from pro_riesgo_finanzas.compara_port import ComparatorService
from bot_runner.backtests.simple_bt import run_backtest, BTParams

# ======================================================
# Parámetros con defaults seguros
# ======================================================
@dataclass
class QubitParams:
    steps_forecast: int = 5          # horizonte pronóstico
    ema_short: int = 12
    ema_long: int = 26
    rsi_window: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    bw_window: int = 20
    bw_thresh: float = 0.10

    # comparador (si no se pasa portafolio, se omite el filtro)
    portfolio_tickers: Optional[list[str]] = None
    portfolio_weights: Optional[list[float]] = None

    # backtest costs
    slippage_bps: float = 0.0
    fee_bps: float = 0.0

    # gating thresholds
    min_sharpe_edge: float = 0.10    # activo debe superar al portafolio por al menos 0.10 en Sharpe
    use_forecast_gate: bool = True   # exigir señal congruente con pronóstico
    use_portfolio_gate: bool = True  # exigir outperformance activo>portafolio si se configuró

# ======================================================
# Helper: forecast sign (dirección)
# ======================================================
def _forecast_bias(df_close: pd.DataFrame, steps: int) -> int:
    """
    Devuelve sesgo direccional {-1, 0, 1} a partir de AutoARIMA y ETS-Holt
    usando el último punto pronosticado vs último real.
    """
    datos = df_close.copy()
    if "Close" not in datos.columns:
        raise ValueError("Se requiere columna 'Close'.")

    # AutoARIMA
    ar = ModeloARIMA(datos)
    ar.entrenar_auto()
    f_ar = ar.pronosticar(steps)
    v_ar = float(np.nanmean(f_ar[-2:])) if f_ar is not None and len(f_ar) > 0 else np.nan

    # ETS (selección automática interna por AIC en tu versión mejorada)
    ets = ModeloSuavizamiento(datos)
    ets.entrenar(seasonal_periods=None, auto_select=True, damped_trend=True)  # sin estacionalidad por defecto
    f_ets = ets.pronosticar(steps=steps, metodo="auto")
    v_ets = float(np.nanmean(f_ets[-2:])) if f_ets is not None and len(f_ets) > 0 else np.nan

    last = float(datos["Close"].iloc[-1])

    votes = []
    if not np.isnan(v_ar):
        votes.append(np.sign(v_ar - last))
    if not np.isnan(v_ets):
        votes.append(np.sign(v_ets - last))

    if not votes:
        return 0
    s = int(np.sign(np.nansum(votes)))
    return s  # -1 down, 0 neutral, 1 up

# ======================================================
# Clase Estrategia (contrato con loader)
# ======================================================
class QubitCapitalComposite:
    name = "QubitCapitalComposite"

    def __init__(self, **kwargs):
        # Carga parámetros con defaults + override de kwargs
        base = QubitParams()
        for k, v in kwargs.items():
            if hasattr(base, k):
                setattr(base, k, v)
        self.p: QubitParams = base
        self._artifacts: Dict[str, Any] = {}

    # --------------------------------------------------
    # prepare: no-IO; setea estado
    # --------------------------------------------------
    def prepare(self, **kwargs) -> None:
        self._artifacts.clear()

    # --------------------------------------------------
    # run: recibe DataFrame OHLCV del runner (yfinance)
    # --------------------------------------------------
    def run(self, **kwargs) -> Iterable[dict]:
        """
        Espera kwargs['data'] = DataFrame con columnas OHLCV (Close requerido).
        Debe devolver un iterable de dicts con el resultado por símbolo,
        pero como el runner nos pasa 1 símbolo por llamada, devolvemos un item.
        """
        df: pd.DataFrame = kwargs.get("data")
        if df is None or df.empty or "Close" not in df.columns:
            raise ValueError("Data inválida o sin 'Close'.")

        # --------- (1) Indicadores y Señales técnicas ----------
        indi = Indicators(df)
        df_ind = indi.compute_all()

        sig = Signals(df_ind)
        df_sig = sig.run_pipeline()  # genera Signals_EMA, Signals_BOLL, Equity (indicativa)

        # Normalizamos nombres rápidos para que el runner pueda graficar algo si quiere
        out = df_sig.copy()
        out.rename(columns={
            "EMA_short": "ma_fast",
            "EMA_long": "ma_slow"
        }, inplace=True)

        # Señal base técnica (EMA+MACD) prioritaria; si es 0, intenta con BOLL
        base_signal = out["Signals_EMA"].fillna(0).astype(int)
        alt_signal = out["Signals_BOLL"].fillna(0).astype(int)
        combo = base_signal.where(base_signal != 0, alt_signal)  # usa BOLL si EMA está neutra

        # --------- (2) Filtro por Forecast (pronósticos) ----------
        if self.p.use_forecast_gate:
            df_close = out[["Close"]].dropna().copy()
            bias = _forecast_bias(df_close, self.p.steps_forecast)  # {-1,0,1}
        else:
            bias = 0

        if bias != 0:
            # exige congruencia señal↔pronóstico, si no, anula
            combo = combo.where(np.sign(combo) == bias, 0)

        # --------- (3) Filtro por Comparador (régimen) ----------
        if self.p.use_portfolio_gate and self.p.portfolio_tickers and self.p.portfolio_weights:
            comp = ComparatorService()
            # mapeo “portafolio” con datos descargados por el runner: usamos el mismo df del símbolo
            # El comparador espera dict[ticker]=df; para el portafolio descargaremos con yfinance en runner normalmente.
            # Aquí aplicamos un “proxy” usando solo Close del activo, y exigimos que Sharpe del activo > Sharpe del portafolio + margen.
            # NOTA: Para máxima fidelidad, ideal: pasar dfs reales del portafolio desde el runner. Aquí lo dejamos “ligero”.
            # → Si no hay dfs del portafolio en kwargs, omitimos gate y dejamos combo tal cual.
            port_map: Optional[Dict[str, pd.DataFrame]] = kwargs.get("portfolio_dfs")
            if port_map:
                try:
                    df_b100, kpis = comp.compare(out, port_map, self.p.portfolio_weights)
                    edge = (kpis["asset"]["sharpe"] - kpis["portfolio"]["sharpe"])
                    if not np.isnan(edge) and edge < self.p.min_sharpe_edge:
                        combo = combo * 0  # no operar en régimen desfavorable
                except Exception:
                    # si el comparador falla por fechas, seguimos sin gate
                    pass

        # Señal final {-1,0,1}
        out["signal"] = combo.clip(-1, 1).astype(int)

        # --------- (4) Backtest interno (sin lookahead) ----------
        bt = run_backtest(
            out,
            price_col="Close",
            signal_col="signal",
            params=BTParams(
                slippage_bps=self.p.slippage_bps,
                fee_bps=self.p.fee_bps,
                initial_capital=10000
            )
        )
        out_bt = bt["df"]
        metrics = bt["metrics"]

        # Guarda artefactos (por si el runner/estrategia.post quiere usarlos)
        self._artifacts["metrics"] = metrics

        # Devolver en el contrato esperado por el runner (un item)
        yield {
            "df": out_bt,
            "metrics": metrics
        }

    # --------------------------------------------------
    # post: aquí podrías exportar extra (logs/plots/JSON)
    # --------------------------------------------------
    def post(self, **kwargs) -> None:
        # Manténlo opcional/silencioso para producción
        pass

