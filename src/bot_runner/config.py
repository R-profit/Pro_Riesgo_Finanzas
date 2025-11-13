# %% Carga y validación de configuración
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import yaml
import os

# %% Modelos
@dataclass
class StrategyConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionConfig:
    plot: bool = True
    export_dir: str = "outputs"

@dataclass
class RiskConfig:
    cash: float = 10000.0

@dataclass
class BotConfig:
    symbols: List[str]
    period: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    strategy: StrategyConfig = field(default_factory=lambda: StrategyConfig(name="simple_ma"))
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

# %% Helpers
def _validate_time(cfg: BotConfig) -> None:
    """Regla: 'period' XOR ('start' y 'end')."""
    has_period = bool(cfg.period)
    has_range = bool(cfg.start) or bool(cfg.end)
    if has_period and has_range:
        raise ValueError("Usa 'period' O 'start'+'end', no ambos.")
    if not has_period and not (cfg.start and cfg.end):
        raise ValueError("Debes definir 'period' o ('start' y 'end').")

def _validate_symbols(symbols: List[str]) -> None:
    if not symbols or not isinstance(symbols, list):
        raise ValueError("'symbols' debe ser una lista no vacía.")
    bad = [s for s in symbols if not isinstance(s, str) or not s.strip()]
    if bad:
        raise ValueError(f"Símbolos inválidos: {bad}")

# %% Cargador público
def load_config(path: str) -> BotConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config no encontrada: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    _validate_symbols(raw.get("symbols"))

    cfg = BotConfig(
        symbols=raw["symbols"],
        period=raw.get("period"),
        start=raw.get("start"),
        end=raw.get("end"),
        strategy=StrategyConfig(**(raw.get("strategy") or {})),
        execution=ExecutionConfig(**(raw.get("execution") or {})),
        risk=RiskConfig(**(raw.get("risk") or {})),
    )
    _validate_time(cfg)
    return cfg

