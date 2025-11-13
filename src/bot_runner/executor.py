# %% ======================================================
# Executor del Bot — orquestador con política de ejecución
# ======================================================
# Rol:
# - Carga configuración (bot.yml o dict ya parseado)
# - Aplica ExecutionPolicy (concurrencia, timeouts, reintentos, backoff)
# - Ejecuta estrategias (pronóstico, señales, comparador) de forma segura
# - Integra con BotRunner y centraliza logs/artefactos
# ======================================================

from __future__ import annotations

# %% [stdlib]
import os
import time
import json
import signal
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

# %% [local deps]
from .config import BotConfig, load_config  # load_config(cfg_path) -> BotConfig
from .io_utils import ensure_dir, setup_logger
from .runner import BotRunner

# ======================================================
# POLÍTICA DE EJECUCIÓN
# ======================================================

@dataclass(frozen=True)
class ExecutionPolicy:
    """Parámetros de control de ejecución (seguros por defecto)."""
    max_concurrency: int = 2          # hilos simultáneos
    timeout_s: int = 90               # timeout por símbolo-estrategia
    retry_max: int = 1                # reintentos máximos
    retry_backoff_s: float = 2.0      # backoff exponencial base
    demo_env_var: str = "BOT_DEMO"    # si == "1" activa modo demo
    dry_run: bool = False             # no exporta/ni toca disco si True
    log_name: str = "executor"        # nombre del logger dedicado

# ======================================================
# RESULTADOS
# ======================================================

@dataclass
class TaskSpec:
    symbol: str
    strategy_name: str

@dataclass
class TaskResult:
    symbol: str
    strategy_name: str
    ok: bool
    csv_path: Optional[str] = None
    png_path: Optional[str] = None
    attempts: int = 0
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

# ======================================================
# EXECUTOR
# ======================================================

class BotExecutor:
    """Aplica ExecutionPolicy y coordina BotRunner por símbolo/estrategia."""

    def __init__(self, cfg: BotConfig, policy: Optional[ExecutionPolicy] = None):
        self.cfg = cfg
        self.policy = policy or ExecutionPolicy()
        self.logger = setup_logger(self.policy.log_name)
        ensure_dir(self.cfg.execution.export_dir)

        # Modo demo por ENV
        self.demo_mode = os.getenv(self.policy.demo_env_var, "").strip() == "1"
        if self.demo_mode:
            self.logger.warning("Modo DEMO activado por ENV=%s", self.policy.demo_env_var)

        # Modo dry-run (no escribir archivos)
        if self.policy.dry_run:
            self.logger.warning("Dry-run activo: no se exportarán archivos.")

        # Instalamos manejadores de señal para shutdown limpio
        try:
            signal.signal(signal.SIGINT, self._handle_sig)
            signal.signal(signal.SIGTERM, self._handle_sig)
        except Exception:
            # En Windows o entornos restringidos puede fallar; lo ignoramos.
            pass
        self._shutdown = False

    # --------------------------------------------------
    # Señales del SO → parada ordenada
    # --------------------------------------------------
    def _handle_sig(self, *_):
        self.logger.warning("Señal de parada recibida. Cerrando tareas…")
        self._shutdown = True

    # --------------------------------------------------
    # Worker: ejecuta una estrategia sobre un símbolo con
    # reintentos, timeout y backoff exponencial.
    # --------------------------------------------------
    def _run_once(self, symbol: str, strategy: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Devuelve (ok, csv_path, png_path)."""
        # Clon superficial de cfg para forzar una estrategia puntual
        local_cfg = self.cfg.copy_with(strategy_name=strategy)

        # Dry-run: desactiva exportes visibles
        if self.policy.dry_run:
            local_cfg.execution.plot = False
            local_cfg.execution.export_dir = os.path.join(self.cfg.execution.export_dir, "_dryrun")

        runner = BotRunner(local_cfg, log_name=f"runner.{strategy}")
        csv_path, png_path = runner.run_symbol(symbol)
        return True, csv_path, png_path

    def _task_with_retries(self, spec: TaskSpec) -> TaskResult:
        attempts = 0
        last_err = None
        csv_path: Optional[str] = None
        png_path: Optional[str] = None

        while attempts <= self.policy.retry_max and not self._shutdown:
            attempts += 1
            try:
                self.logger.info("▶️ %s | %s | intento %d",
                                 spec.strategy_name, spec.symbol, attempts)
                ok, csv_path, png_path = self._run_once(spec.symbol, spec.strategy_name)
                return TaskResult(
                    symbol=spec.symbol,
                    strategy_name=spec.strategy_name,
                    ok=ok,
                    csv_path=csv_path,
                    png_path=png_path,
                    attempts=attempts,
                    meta={"demo": self.demo_mode, "dry_run": self.policy.dry_run},
                )
            except Exception as e:
                last_err = str(e)
                self.logger.error("❌ %s | %s | intento %d falló: %s",
                                  spec.strategy_name, spec.symbol, attempts, e)
                # backoff
                if attempts <= self.policy.retry_max:
                    sleep_for = self.policy.retry_backoff_s * (2 ** (attempts - 1))
                    time.sleep(min(sleep_for, 30.0))

        return TaskResult(
            symbol=spec.symbol,
            strategy_name=spec.strategy_name,
            ok=False,
            error=last_err,
            attempts=attempts,
            meta={"demo": self.demo_mode, "dry_run": self.policy.dry_run},
        )

    # --------------------------------------------------
    # Planificación y ejecución concurrente
    # --------------------------------------------------
    def plan_tasks(self) -> List[TaskSpec]:
        """Construye la lista de tareas: símbolo × estrategia."""
        tasks: List[TaskSpec] = []
        # Si en cfg hay UNA estrategia → la usamos; si no, puedes
        # pasar una lista por ENV (estrategias separadas por coma).
        strat_from_env = os.getenv("EXEC_STRATEGIES", "").strip()
        if strat_from_env:
            strategies = [s.strip() for s in strat_from_env.split(",") if s.strip()]
        else:
            strategies = [self.cfg.strategy.name]

        for s in strategies:
            for sym in self.cfg.symbols:
                tasks.append(TaskSpec(symbol=sym, strategy_name=s))
        return tasks

    def run(self) -> List[TaskResult]:
        tasks = self.plan_tasks()
        self.logger.info("=== Executor start | tasks=%d | concurrency=%d ===",
                         len(tasks), self.policy.max_concurrency)

        results: List[TaskResult] = []
        if not tasks:
            self.logger.warning("No hay tareas planificadas.")
            return results

        with ThreadPoolExecutor(max_workers=max(1, self.policy.max_concurrency)) as pool:
            future_map = {
                pool.submit(self._task_with_timeout, t): t for t in tasks
            }
            for fut in as_completed(future_map):
                spec = future_map[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    self.logger.exception("Fallo no controlado en tarea %s/%s", spec.strategy_name, spec.symbol)
                    res = TaskResult(
                        symbol=spec.symbol,
                        strategy_name=spec.strategy_name,
                        ok=False,
                        error=str(e),
                        attempts=1,
                    )
                results.append(res)
                # Log corto de estado
                if res.ok:
                    extra = f" CSV={res.csv_path}" + (f" PNG={res.png_path}" if res.png_path else "")
                    self.logger.info("✅ %s | %s completado.%s",
                                     res.strategy_name, res.symbol, extra)
                else:
                    self.logger.error("⛔ %s | %s ERROR: %s",
                                      res.strategy_name, res.symbol, res.error)

        self.logger.info("=== Executor end | ok=%d/%d ===",
                         sum(r.ok for r in results), len(results))
        return results

    # ---- envoltorio con timeout por tarea ----
    def _task_with_timeout(self, spec: TaskSpec) -> TaskResult:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._task_with_retries, spec)
            try:
                return future.result(timeout=self.policy.timeout_s)
            except TimeoutError:
                self.logger.error("⏱️ Timeout %ss en %s | %s",
                                  self.policy.timeout_s, spec.strategy_name, spec.symbol)
                return TaskResult(
                    symbol=spec.symbol,
                    strategy_name=spec.strategy_name,
                    ok=False,
                    error=f"timeout {self.policy.timeout_s}s",
                    attempts=1,
                )

# ======================================================
# CLI / ENTRYPOINT
# ======================================================

def main(argv: Optional[List[str]] = None) -> int:
    """
    Uso:
      python -m bot_runner.executor --config ./configs/bot.yml
    Opcionales:
      --dry-run            # no exporta archivos
      --concurrency 3      # hilos
      --timeout 120        # por tarea
      --retries 2
      --strategies "pronostico_arima,signals_ta,compara_port"
    """
    import argparse

    p = argparse.ArgumentParser("Bot Executor")
    p.add_argument("--config", required=True, help="Ruta al bot.yml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--retries", type=int, default=None)
    p.add_argument("--strategies", type=str, default=None,
                   help="Lista separada por coma (override).")
    args = p.parse_args(argv)

    cfg: BotConfig = load_config(args.config)

    policy = ExecutionPolicy(
        max_concurrency=args.concurrency if args.concurrency else ExecutionPolicy.max_concurrency,
        timeout_s=args.timeout if args.timeout else ExecutionPolicy.timeout_s,
        retry_max=args.retries if args.retries else ExecutionPolicy.retry_max,
        dry_run=args.dry_run,
    )

    # Override de estrategias por CLI → via ENV (para mantener un solo punto)
    if args.strategies:
        os.environ["EXEC_STRATEGIES"] = args.strategies

    exec_ = BotExecutor(cfg, policy=policy)
    results = exec_.run()

    # Salida mínima machine-readable (logs ya llevan el resto)
    print(json.dumps({
        "total": len(results),
        "ok": sum(r.ok for r in results),
        "errors": [
            {"symbol": r.symbol, "strategy": r.strategy_name, "error": r.error}
            for r in results if not r.ok
        ]
    }, ensure_ascii=False))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())


