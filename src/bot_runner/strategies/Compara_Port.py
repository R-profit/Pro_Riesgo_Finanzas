# %% ======================================================
# Strategy: ComparaPortStrategy
# Compara un activo vs un portafolio (pesos) y devuelve KPIs + recomendación.
# Integra el módulo: pro_riesgo_finanzas.comparador (provider-agnóstico MT5/YF).
# Seguridad: logs discretos; sin PII; validaciones básicas.
# ======================================================

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

# Nota: import local a la strategy, no a nivel global del runner:
from pro_riesgo_finanzas.comparador import comparar_y_exportar

# ------------------------------------------------------
# Parámetros (con defaults razonables)
# ------------------------------------------------------
@dataclass
class ComparaParams:
    asset: str                       # p.ej. "XAUUSD" (MT5) o "GC=F" (YF)
    portfolio: List[str]             # p.ej. ["XAGUSD","WTICOUSD"]
    weights: List[float]             # deben sumar 1.0
    timeframe: str = "D1"            # "M15","H1","D1", etc.
    # Opción A: lookback rolling (prioritaria si se define)
    lookback_days: Optional[int] = 120
    # Opción B: rango fijo (si ambos se dan, ignora lookback)
    start: Optional[str] = None      # "YYYY-MM-DD"
    end: Optional[str] = None        # "YYYY-MM-DD"
    # Export opcional (PNG/PDF usando el módulo comparador)
    export: bool = False
    export_basename: Optional[str] = None
    # Umbrales de recomendación
    min_sharpe_edge: float = 0.10    # ventaja mínima de Sharpe vs portafolio
    min_ret_edge: float = 0.00       # ventaja mínima de retorno anualizado

    # Campos internos calculados
    _start_iso: str = field(init=False, repr=False, default="")
    _end_iso: str = field(init=False, repr=False, default="")

    def resolve_dates(self) -> None:
        """Resuelve start/end: rango explícito o ventana lookback."""
        if self.start and self.end:
            self._start_iso = self.start
            self._end_iso = self.end
            return
        # lookback (UTC, fecha solamente)
        end_dt = datetime.now(timezone.utc).date()
        start_dt = end_dt - timedelta(days=int(self.lookback_days or 120))
        self._start_iso = start_dt.isoformat()
        self._end_iso = end_dt.isoformat()

# ------------------------------------------------------
# Strategy
# ------------------------------------------------------
class ComparaPortStrategy:
    """
    Strategy minimalista:
      - Usa comparar_y_exportar(...) que ya maneja ProviderAdapter (MT5 o DEMO YF).
      - Devuelve KPIs + recomendación simple (long_bias o neutral) para el bot.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None) -> None:
        # context puede traer logger, provider pre-inicializado, etc. (opcional)
        self.context = context or {}

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # --- Parseo de parámetros seguros
        p = ComparaParams(**params)
        p.resolve_dates()

        # --- Ejecuta comparador (maneja MT5 o YF DEMO por dentro)
        out = comparar_y_exportar(
            asset_ticker=p.asset,
            portfolio_tickers=p.portfolio,
            portfolio_weights=p.weights,
            start=p._start_iso,
            end=p._end_iso,
            timeframe=p.timeframe,
            export_basename=(p.export_basename if p.export else None),
        )

        k = out["payload"]["kpis"]   # {'asset': {...}, 'portfolio': {...}}
        aset, port = k["asset"], k["portfolio"]

        # --- Reglas de recomendación simples (puedes afinarlas):
        # long_bias si el activo supera al portafolio en Sharpe y retorno por un margen
        sharpe_edge = (aset.get("sharpe") or float("nan")) - (port.get("sharpe") or float("nan"))
        ret_edge    = (aset.get("ret_annual") or float("nan")) - (port.get("ret_annual") or float("nan"))

        long_bias = False
        if _is_finite(sharpe_edge) and _is_finite(ret_edge):
            if sharpe_edge >= p.min_sharpe_edge and ret_edge >= p.min_ret_edge:
                long_bias = True

        # --- Respuesta estándar para el runner
        result: Dict[str, Any] = {
            "meta": out["payload"]["meta"],
            "kpis": k,
            "sample": out["payload"]["sample"],  # últimas filas base100 (útil para UI/log)
            "signal": {
                "bias": "long" if long_bias else "neutral",
                "explanation": _explain_bias(long_bias, sharpe_edge, ret_edge),
                "thresholds": {
                    "min_sharpe_edge": p.min_sharpe_edge,
                    "min_ret_edge": p.min_ret_edge,
                },
                "edges": {
                    "sharpe_edge": sharpe_edge,
                    "ret_edge": ret_edge,
                },
            },
        }
        # Nota: No devolvemos figuras/PNG/PDF en el payload para no inflar el mensaje.
        # Si p.export=True, ya quedaron guardados por el módulo comparador.
        return result


# ------------------------------------------------------
# Helpers internos (sin dependencias externas)
# ------------------------------------------------------
def _is_finite(x: Any) -> bool:
    try:
        from math import isfinite
        return isfinite(float(x))
    except Exception:
        return False

def _explain_bias(long_bias: bool, sharpe_edge: float, ret_edge: float) -> str:
    if long_bias:
        return (f"Activo con ventaja en Sharpe (+{sharpe_edge:.2f}) y retorno anual (+{ret_edge:.2%}) "
                f"frente al portafolio; sesgo LONG.")
    return (f"Sin ventaja suficiente: ΔSharpe={sharpe_edge:.2f}, ΔRet={ret_edge:.2%}. "
            f"Sesgo NEUTRAL.")
# %%


# %% ======================================================
# Smoke test local (opcional, fuera del runner)
#   - Usa YF DEMO si no hay provider del bot.
#   - NO toca tu broker ni credenciales.
# ======================================================
if __name__ == "__main__":
    demo = ComparaPortStrategy()
    demo_out = demo.run({
        "asset": "GC=F",
        "portfolio": ["SI=F", "CL=F"],
        "weights": [0.5, 0.5],
        "timeframe": "D1",
        "lookback_days": 300,
        "export": True,
        "export_basename": "DEMO_GC_vs_PORTFOLIO",
        "min_sharpe_edge": 0.10,
        "min_ret_edge": 0.00,
    })
    from pprint import pprint
    pprint(demo_out["signal"])
    pprint(demo_out["kpis"])

