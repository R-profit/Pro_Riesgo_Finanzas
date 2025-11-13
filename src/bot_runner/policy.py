# %% ======================================================
# Execution Policy — minimalista, segura y modular
# - Lee la señal de la strategy (bias: long/neutral)
# - Aplica reglas de entrada/salida con riesgo fijo
# - Respeta modo DEMO/LIVE (ENV LIVE_TRADING=1)
# ======================================================

from __future__ import annotations
import os, math, time
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Interfaces muy simples para desacoplar broker:
class BrokerAdapter:
    """Interfaz mínima esperada por la policy."""
    def get_price(self, symbol: str) -> float: ...
    def position_size(self, symbol: str) -> float: ...
    def open_long(self, symbol: str, volume: float, sl: Optional[float], tp: Optional[float]) -> str: ...
    def close_all(self, symbol: str) -> None: ...

@dataclass
class PolicyConfig:
    risk_per_trade: float = 0.005     # 0.5% del capital
    capital_base: float = 10_000.0
    sl_pct: float = 0.006             # 0.6% stop loss
    tp_pct: float = 0.012             # 1.2% take profit
    cooldown_s: int = 180             # ventana anti “overtrading” (segundos)
    max_pos_per_symbol: int = 1       # no más de 1 posición por símbolo
    live_trading_env: str = "LIVE_TRADING"  # ENV=1 habilita ejecuciones reales

class ExecutionPolicy:
    """
    Política de ejecución basada en sesgo de la strategy:
      - bias='long'  -> intenta abrir long si no hay posición y si no está en cooldown.
      - bias='neutral' -> cierra posiciones si las hay.
    SL/TP porcentuales (simples) y tamaño por riesgo fijo.
    """

    def __init__(self, broker: BrokerAdapter, cfg: Optional[PolicyConfig] = None):
        self.broker = broker
        self.cfg = cfg or PolicyConfig()
        self._last_action_ts: Dict[str, float] = {}

    def _can_act(self, symbol: str) -> bool:
        now = time.time()
        last = self._last_action_ts.get(symbol, 0.0)
        return (now - last) >= self.cfg.cooldown_s

    def _mark_act(self, symbol: str) -> None:
        self._last_action_ts[symbol] = time.time()

    def _is_live(self) -> bool:
        return os.getenv(self.cfg.live_trading_env, "").strip() == "1"

    def _size_from_risk(self, price: float) -> float:
        risk_amt = self.cfg.capital_base * self.cfg.risk_per_trade
        sl_dist = price * self.cfg.sl_pct
        if sl_dist <= 0:
            return 0.0
        units = risk_amt / sl_dist
        # Ajuste simple a lotes/contratos enteros, si aplica:
        return max(0.01, round(units, 2))

    def _compute_sl_tp(self, price: float) -> (float, float):
        sl = price * (1.0 - self.cfg.sl_pct)
        tp = price * (1.0 + self.cfg.tp_pct)
        return sl, tp

    def decide(self, symbol: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica la política:
          - Si bias='long' y sin posición -> abrir long con SL/TP.
          - Si bias='neutral' y hay posición -> cerrar.
        Devuelve un dict con la acción realizada o recomendada.
        """
        bias = (signal or {}).get("bias", "neutral")
        info = {"action": "hold", "live": self._is_live(), "details": {}}

        # 1) Leer estado mínimo
        price = float(self.broker.get_price(symbol))
        current = float(self.broker.position_size(symbol))

        # 2) Reglas
        if bias == "long":
            if current > 0.0:
                info["action"] = "hold_long"
                info["details"] = {"reason": "already_long", "pos": current, "price": price}
                return info

            if not self._can_act(symbol):
                info["action"] = "cooldown"
                info["details"] = {"reason": "cooldown", "price": price}
                return info

            vol = self._size_from_risk(price)
            sl, tp = self._compute_sl_tp(price)

            if self._is_live():
                order_id = self.broker.open_long(symbol, vol, sl, tp)
                self._mark_act(symbol)
                info["action"] = "open_long"
                info["details"] = {"order_id": order_id, "vol": vol, "sl": sl, "tp": tp, "price": price}
            else:
                # DEMO: no ejecuta, solo reporta
                self._mark_act(symbol)
                info["action"] = "simulate_open_long"
                info["details"] = {"vol": vol, "sl": sl, "tp": tp, "price": price}

            return info

        # bias neutral -> cerrar si hay
        if bias == "neutral" and current > 0.0:
            if not self._can_act(symbol):
                info["action"] = "cooldown"
                info["details"] = {"reason": "cooldown_close", "pos": current, "price": price}
                return info

            if self._is_live():
                self.broker.close_all(symbol)
                self._mark_act(symbol)
                info["action"] = "close_all"
                info["details"] = {"pos": current, "price": price}
            else:
                self._mark_act(symbol)
                info["action"] = "simulate_close_all"
                info["details"] = {"pos": current, "price": price}

            return info

        # nada que hacer
        info["details"] = {"reason": "no_change", "pos": current, "price": price}
        return info
# %%

