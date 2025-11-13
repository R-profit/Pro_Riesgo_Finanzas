# Base y loader de estrategias con restricción + allowlist por entorno
from __future__ import annotations
from typing import Protocol, Iterable
import importlib
import os

# Prefijo permitido por defecto (todo debe vivir aquí salvo que se whitelistee)
_DEFAULT_ALLOWED_PREFIX = "bot_runner.strategies."

def _allowed_prefixes() -> list[str]:
    """
    Lee ALLOWED_STRATEGY_PREFIXES de entorno (CSV) y añade el prefijo por defecto.
    Ejemplo de uso:
      ALLOWED_STRATEGY_PREFIXES="bot_runner.strategies.,empresa_fx.strategies."
    """
    raw = os.getenv("ALLOWED_STRATEGY_PREFIXES", "")
    extra = [p.strip() for p in raw.split(",") if p.strip()]
    # garantizamos que el prefijo por defecto siempre esté
    return sorted(set([_DEFAULT_ALLOWED_PREFIX, *extra]))

def _is_allowed(module_name: str) -> bool:
    return any(module_name.startswith(p) for p in _allowed_prefixes())

class Strategy(Protocol):
    """
    Contrato mínimo de una estrategia pluggable:
    - name: str (identificador humano)
    - prepare(**kwargs): validar/cargar dependencias y parámetros
    - run(**kwargs) -> Iterable[dict]: produce registros (para CSV/pipe)
    - post(**kwargs): persistencia/cierre (ej. escritura atómica de CSV)
    """
    name: str
    def prepare(self, **kwargs) -> None: ...
    def run(self, **kwargs) -> Iterable[dict]: ...
    def post(self, **kwargs) -> None: ...

def load_strategy(qualified: str) -> Strategy:
    """
    Carga dinámica segura de una estrategia.
    Formato requerido: 'paquete.modulo:Clase'
    Restricción: el paquete debe iniciar con uno de los prefijos permitidos.
    Control por entorno: ALLOWED_STRATEGY_PREFIXES (CSV).
    """
    if ":" not in qualified:
        raise ImportError("Ruta inválida: usa 'paquete.modulo:Clase'.")

    module_name, cls_name = qualified.split(":", 1)

    # Seguridad: restringimos donde se puede importar
    if not _is_allowed(module_name):
        allowed = ", ".join(_allowed_prefixes())
        raise ImportError(f"Módulo no permitido: {module_name}. Permitidos: {allowed}")

    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, cls_name)
        inst = cls()
    except Exception as e:
        # Mensaje limpio, sin filtrar rutas internas del sistema
        raise ImportError(f"No pude cargar la estrategia '{qualified}': {e}") from e

    # Validación del contrato
    for m in ("prepare", "run", "post"):
        attr = getattr(inst, m, None)
        if not callable(attr):
            raise TypeError(f"Estrategia inválida '{qualified}': método '{m}' no es callable")
    if not isinstance(getattr(inst, "name", None), str):
        raise TypeError(f"Estrategia inválida '{qualified}': atributo 'name' (str) requerido")

    return inst


