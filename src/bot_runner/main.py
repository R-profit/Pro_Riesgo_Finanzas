from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from dotenv import load_dotenv

# Asegura que 'src/' esté en sys.path para importar tu paquete local
# (esto evita problemas al ejecutar desde la raíz del repo en Windows).
ROOT = Path(__file__).resolve().parents[2]  # .../Pro_Riesgo_Finanzas/src/bot_runner/main.py -> raíz
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Carga .env ANTES de importar runner/otros módulos (para que lean variables)
load_dotenv()

import yaml
from bot_runner.runner import run_once, run_loop

def build_parser():
    p = argparse.ArgumentParser("bot-runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    # once
    once = sub.add_parser("once", help="Ejecuta una vez")
    once.add_argument("--config", required=True, help="Ruta al YAML de configuración")
    once.add_argument("--strategy", required=True, help="paquete.modulo:Clase")
    once.add_argument("--params", default="{}", help="JSON con params de estrategia")
    once.add_argument("--no-mt5", action="store_true", help="No inicializar MT5 (p.ej. backtest offline)")

    # loop
    loop = sub.add_parser("loop", help="Ejecuta en bucle")
    loop.add_argument("--config", required=True)
    loop.add_argument("--strategy", required=True)
    loop.add_argument("--every", type=int, default=300, help="Segundos entre ciclos")
    loop.add_argument("--params", default="{}")
    loop.add_argument("--no-mt5", action="store_true")

    return p

def _precheck_yaml(config_path: str):
    """
    Mantiene tu comportamiento previo:
    - Abre el YAML (sin cambiar nombres).
    - Valida claves mínimas (symbols/strategy opcionales).
    - No altera la ejecución: sólo sirve para detectar errores temprano.
    """
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Validaciones suaves (no rompemos tu estructura)
    if "symbols" in cfg and (not isinstance(cfg["symbols"], list) or not cfg["symbols"]):
        raise ValueError("En YAML: 'symbols' debe ser una lista no vacía si se define.")
    # Si definiste strategy en YAML, no lo forzamos; CLI --strategy manda la que vas a ejecutar ahora.
    return cfg  # por si quieres inspeccionarlo en el futuro

def main():
    p = build_parser()
    args = p.parse_args()

    # Mantener tu paso: cargar YAML con yaml.safe_load antes de correr
    _ = _precheck_yaml(args.config)

    # Params pasan intactos a la strategy (no renombro variables)
    params = json.loads(args.params) if args.params else {}

    if args.cmd == "once":
        code = run_once(
            config_path=args.config,
            strategy_path=args.strategy,
            strategy_params=params,
            use_mt5=not args.no_mt5,
        )
        raise SystemExit(code)

    if args.cmd == "loop":
        code = run_loop(
            config_path=args.config,
            strategy_path=args.strategy,
            every_seconds=args.every,
            strategy_params=params,
            use_mt5=not args.no_mt5,
        )
        raise SystemExit(code)

if __name__ == "__main__":
    main()

