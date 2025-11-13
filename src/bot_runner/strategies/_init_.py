# %% [registry+adapter] Alias + compat con loader seguro (prepare/run/post)
from __future__ import annotations
from typing import Any, Iterable, Optional, Type
import importlib
import logging
import inspect
import pandas as pd

# Tu loader seguro, ya con whitelist y validaciones
from .loader import load_strategy as _load_safe_instance

log = logging.getLogger("bot.strategies.registry")

# --------------------------------------------------------------------
# REGISTRO de alias → "paquete.modulo:Clase"
# Asegúrate que estos paths/clases existen:
#   - bot_runner/strategies/pronostico_arima.py -> class ArimaForecastStrategy
#   - bot_runner/strategies/signals_ta.py       -> class SignalsTaStrategy
#   - bot_runner/strategies/compara_port.py     -> class ComparePortfolioStrategy
REGISTRY: dict[str, str] = {
    "pronostico_arima": "bot_runner.strategies.pronostico_arima:ArimaForecastStrategy",
    "signals_ta":       "bot_runner.strategies.signals_ta:SignalsTaStrategy",
    "compara_port":     "bot_runner.strategies.compara_port:ComparePortfolioStrategy",
}
# --------------------------------------------------------------------


def _is_qualified(path: str) -> bool:
    return ":" in path and len(path.split(":", 1)[0]) > 0


def _import_class(path: str) -> Type:
    """Carga 'paquete.modulo:Clase' y devuelve la clase."""
    mod_path, cls_name = path.split(":", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def _accepts_kwargs(cls: Type) -> bool:
    """Heurística: ¿el constructor admite kwargs?"""
    try:
        sig = inspect.signature(cls)
        params = list(sig.parameters.values())
        # **kwargs o al menos algún parámetro con default (para ser flexible)
        return any(p.kind == p.VAR_KEYWORD for p in params) or any(
            p.default is not p.empty for p in params[1:]
        )
    except Exception:
        return True  # permisivo si no podemos inspeccionar


class _AdapterGenerate:
    """
    Envuelve una instancia con contrato prepare/run/post para exponer .generate(df)
    de modo que el BotRunner no cambie.
    """
    def __init__(self, inst: Any, params: Optional[dict] = None):
        self._inst = inst
        self._params = params or {}

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        # 1) prepare(**params) si existe
        if hasattr(self._inst, "prepare"):
            self._inst.prepare(**self._params)

        # 2) run(...) puede devolver DataFrame o Iterable[dict]
        out = self._inst.run(data=data) if hasattr(self._inst, "run") else None

        # 3) post() si existe (no falla el flujo si lanza)
        if hasattr(self._inst, "post"):
            try:
                self._inst.post()
            except Exception:
                pass

        # Normaliza salida a DataFrame
        if isinstance(out, pd.DataFrame):
            return out
        if isinstance(out, Iterable):
            rows = list(out)
            if not rows:
                return pd.DataFrame(index=data.index)
            if isinstance(rows[0], dict):
                return pd.DataFrame(rows)

        # Fallback: atributos comunes
        for attr in ("df", "data", "output"):
            cand = getattr(self._inst, attr, None)
            if isinstance(cand, pd.DataFrame):
                return cand

        raise TypeError("La estrategia (prepare/run/post) no devolvió salida utilizable.")


def get_strategy(name_or_path: str):
    """
    Devuelve un *callable estilo clase* que acepta **kwargs y retorna
    una instancia con .generate(data) — compatible con tu BotRunner.
    Soporta:
      - Alias del REGISTRY
      - Ruta 'pkg.mod:Class'
      - Instancia creada por tu loader seguro (contrato prepare/run/post)
    """
    target = REGISTRY.get(name_or_path, name_or_path)

    # Caso A) Ruta calificada: intentar cargar como CLASE con .generate
    if _is_qualified(target):
        try:
            cls = _import_class(target)
            if callable(getattr(cls, "generate", None)):
                # devolvemos un "constructor" compatible con Strategy(**params)
                def _factory(**params):
                    return cls(**params) if _accepts_kwargs(cls) else cls()
                return _factory
        except Exception:
            # si falla como clase, probamos con el loader seguro
            pass

        # Caso B) Usar loader seguro (instancia) + adaptador .generate
        def _factory(**params):
            inst = _load_safe_instance(target)   # instancia validada
            return _AdapterGenerate(inst, params)
        return _factory

    # Caso C) Alias → resolver a su ruta y recursar
    if target in REGISTRY and REGISTRY[target] != target:
        return get_strategy(REGISTRY[target])

    raise ImportError(f"Estrategia no reconocida: {name_or_path}")

