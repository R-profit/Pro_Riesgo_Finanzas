# %% CLI del bot (seguro para línea de comandos)
from __future__ import annotations
import argparse
from bot_runner.config import load_config
from bot_runner.runner import BotRunner

def main():
    # %% Args
    ap = argparse.ArgumentParser(description="Bot de trading — señales (Simple MA).")
    ap.add_argument("-c", "--config", default="configs/bot.yml", help="Ruta al YAML de configuración.")
    args = ap.parse_args()

    # %% Carga y run
    cfg = load_config(args.config)
    BotRunner(cfg).run()

if __name__ == "__main__":
    main()
