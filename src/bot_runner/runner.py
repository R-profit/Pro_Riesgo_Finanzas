# %% Runner principal del bot
from __future__ import annotations
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from typing import Tuple, Optional

from .config import BotConfig
from .io_utils import ensure_dir, setup_logger
from .strategies import get_strategy

# >>> NUEVO: importa la policy
from .policies.execution_policy import ThresholdPolicy  # <- añade esto

class BotRunner:
    """Coordina: carga datos -> aplica estrategia -> exporta artefactos."""

    def __init__(self, cfg: BotConfig, log_name: str = "bot"):
        self.cfg = cfg
        self.logger = setup_logger(log_name)
        ensure_dir(self.cfg.execution.export_dir)

    # ... (tu _download, _export_csv, _export_plot se quedan igual) ...

    def _export_trades(self, symbol: str, buys: pd.DataFrame, sells: pd.DataFrame) -> Optional[str]:
        """Exporta CSV con decisiones finales de la policy (si las hay)."""
        if (buys is None or buys.empty) and (sells is None or sells.empty):
            return None
        out = os.path.join(self.cfg.execution.export_dir, f"{symbol}_trades.csv")
        # Unimos en un solo CSV con etiqueta de lado
        frames = []
        if buys is not None and not buys.empty:
            b = buys.copy();  b["side"] = "BUY";  frames.append(b)
        if sells is not None and not sells.empty:
            s = sells.copy(); s["side"] = "SELL"; frames.append(s)
        pd.concat(frames).to_csv(out, index=True)
        self.logger.info("TRADES exportado: %s", out)
        return out

    def run_symbol(self, symbol: str) -> Tuple[str, Optional[str], Optional[str]]:
        data = self._download(symbol)

        StrategyFactory = get_strategy(self.cfg.strategy.name)
        strat = StrategyFactory(self.cfg.strategy.params or {})

        self.logger.info("Estrategia: %s | params=%s", self.cfg.strategy.name, self.cfg.strategy.params)
        out = strat.generate(data)   # <- debe devolver DataFrame con al menos 'Close' (y ojalá 'signal', 'confidence')

        csv_path = self._export_csv(symbol, out)
        png_path = self._export_plot(symbol, out)

        # >>> NUEVO: aplica policy de umbrales
        policy = ThresholdPolicy(  # valores por defecto; puedes cambiarlos vía params si quieres
            buy_threshold = float((self.cfg.strategy.params or {}).get("buy_threshold", 0.60)),
            sell_threshold= float((self.cfg.strategy.params or {}).get("sell_threshold",0.60)),
        )
        buys, sells = policy.decide(out)
        trades_csv = self._export_trades(symbol, buys, sells)

        return csv_path, png_path, trades_csv

    def run(self) -> None:
        self.logger.info("=== BotRunner start ===")
        self.logger.info("Símbolos: %s", self.cfg.symbols)
        self.logger.info("Export dir: %s | plot: %s", self.cfg.execution.export_dir, self.cfg.execution.plot)

        for sym in self.cfg.symbols:
            try:
                csv, png, trades = self.run_symbol(sym)
                extra = ""
                if png:    extra += f" | PNG={png}"
                if trades: extra += f" | TRADES={trades}"
                self.logger.info("OK %s -> CSV=%s%s", sym, csv, extra)
            except Exception as e:
                self.logger.error("ERROR %s -> %s", sym, e)

        self.logger.info("=== BotRunner end ===")


#Reseña rápida: después de que la estrategia devuelve su DataFrame, 
#el runner aplica la política y te deja un *_trades.csv con las decisiones finales (BUY/SELL) filtradas por umbral.
