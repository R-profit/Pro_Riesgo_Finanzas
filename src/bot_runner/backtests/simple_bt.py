# %% ======================================================
# Backtester simple (vectorizado, sin lookahead)
# ======================================================
# Autor: R_Profit (QubitCapital)
# Propósito: convertir una columna de señales {-1,0,1} en PnL y métricas.
# Seguridad: sin prints en producción; usa funciones puras.
# ======================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

# %% ------------------------------------------------------
# Config/Params del backtest (riesgo y costos)
# ------------------------------------------------------
@dataclass(frozen=True)
class BTParams:
    slippage_bps: float = 0.0      # coste por operación en basis points (0.0001 = 1 bps)
    fee_bps: float = 0.0           # comisión ida y vuelta en bps aplicadas en cambio de posición
    initial_capital: float = 10000
    trading_days: int = 252

# %% ------------------------------------------------------
# Métricas clave (CAGR, Sharpe, MaxDD, WinRate)
# ------------------------------------------------------
def _metrics(equity: pd.Series, rets: pd.Series, trading_days: int) -> Dict[str, float]:
    eq = equity.dropna()
    r = rets.dropna()
    if eq.empty or r.empty:
        return {"CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "WinRate": np.nan}

    years = max((eq.index[-1] - eq.index[0]).days / 365.0, 1e-9)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)

    mean_d = float(r.mean())
    vol_d = float(r.std(ddof=0))
    sharpe = (mean_d * trading_days) / (vol_d * np.sqrt(trading_days)) if vol_d > 0 else np.nan

    roll_max = eq.cummax()
    dd = (eq / roll_max) - 1.0
    maxdd = float(dd.min())

    winrate = float((r > 0).mean())
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": maxdd, "WinRate": winrate}

# %% ------------------------------------------------------
# Backtest principal (sin piramidación; 100% long/flat/short)
# ------------------------------------------------------
def run_backtest(
    df: pd.DataFrame,
    *,
    price_col: str = "Close",
    signal_col: str = "signal",
    params: Optional[BTParams] = None
) -> Dict[str, object]:
    """
    Espera df con:
      - price_col: precios de cierre (float)
      - signal_col: {-1,0,1} sin lookahead (ya confirmado)
    Devuelve: dict con dataframe + métricas.
    """
    if params is None:
        params = BTParams()

    data = df[[price_col, signal_col]].copy().dropna()
    if data.empty:
        return {"df": df, "metrics": {}}

    # Retornos de precio
    px = data[price_col].astype(float)
    r = px.pct_change().fillna(0.0)

    # Posición = señal (se aplica al día siguiente → shift(1))
    pos = data[signal_col].astype(float).clip(-1, 1).shift(1).fillna(0.0)

    # Costes por cambio de posición (solo cuando cambia la señal)
    pos_change = pos.diff().abs().fillna(0.0)
    roundtrip_cost = params.slippage_bps + params.fee_bps
    cost_series = pos_change * roundtrip_cost * 1e-4  # pasar de bps a proporción

    # Retorno neto
    strat_r = (pos * r) - cost_series
    equity = (1.0 + strat_r).cumprod() * params.initial_capital

    out = df.copy()
    out["signal_used"] = pos
    out["ret_price"] = r
    out["ret_strategy"] = strat_r
    out["equity"] = equity

    m = _metrics(equity, strat_r, params.trading_days)
    return {"df": out, "metrics": m}

