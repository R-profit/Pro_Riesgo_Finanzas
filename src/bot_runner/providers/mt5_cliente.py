#Este codigo conecta MT5 con (pepperstone) en modo dual,
# (usa login por variables de entorno si estan; sino usa la sesion abierta de MT5).
#Lee velas OHCL, las normaliza a UTC y te la da como listad e diccionarios o pandas dataframe con indice "time" (tz-aware UTC)
# valida simbolos (regex + Allowlist opcional) y aplica reintentos con backoff exponencial si 
#la lectura falla temporalmente

# Cliente MT5 (Pepperstone) con modo dual, UTC, validación de símbolo y reintentos
from __future__ import annotations
import os, re, time
from dataclasses import dataclass
from typing import Optional, Literal, List, Iterable
from datetime import datetime
from pathlib import Path

try:
    import MetaTrader5 as MT5  # pip install MetaTrader5
except Exception as e:
    raise ImportError("Falta dependencia MetaTrader5. Instala 'MetaTrader5'.") from e

# -------------------- Timeframes soportados --------------------
TF = Literal["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
TF_MAP = {
    "M1":  MT5.TIMEFRAME_M1,
    "M5":  MT5.TIMEFRAME_M5,
    "M15": MT5.TIMEFRAME_M15,
    "M30": MT5.TIMEFRAME_M30,
    "H1":  MT5.TIMEFRAME_H1,
    "H4":  MT5.TIMEFRAME_H4,
    "D1":  MT5.TIMEFRAME_D1,
}

# -------------------- Configuración --------------------
@dataclass
class MT5Settings:
    login: Optional[int] = None            # MT5_LOGIN
    password: Optional[str] = None         # MT5_PASSWORD
    server: Optional[str] = None           # MT5_SERVER (ej. 'Pepperstone-Edge10')
    path_terminal: Optional[Path] = None   # MT5_TERMINAL_PATH (opcional)

# Regex conservadora para símbolos (letras/números y separadores comunes)
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9._\-#/]+$")

# -------------------- Cliente --------------------
class MT5Client:
    """
    Cliente MT5 con 'modo dual':
      - Si existen MT5_LOGIN/MT5_PASSWORD/MT5_SERVER -> login por código.
      - Si no existen -> usa la sesión abierta de la terminal MT5.
    Seguridad: validación de símbolo + reintentos con backoff en lecturas.
    """

    def __init__(
        self,
        settings: Optional[MT5Settings] = None,
        allowed_symbols: Optional[Iterable[str]] = None,
        max_retries: int = 3,
        backoff_base_sec: float = 0.75,
    ):
        self.settings = settings or MT5Settings(
            login=_env_int("MT5_LOGIN"),
            password=os.getenv("MT5_PASSWORD"),
            server=os.getenv("MT5_SERVER"),
            path_terminal=_env_path("MT5_TERMINAL_PATH"),
        )
        self._initialized = False
        self._authorized = False
        self.allowed_symbols = set(allowed_symbols) if allowed_symbols else None
        self.max_retries = max(1, int(max_retries))
        self.backoff_base_sec = float(backoff_base_sec)

    # ---------- Ciclo de vida ----------
    def initialize(self) -> None:
        """Inicializa MT5 y, si hay credenciales, hace login por código."""
        if self._initialized:
            return

        kwargs = {}
        if self.settings.path_terminal:
            kwargs["path"] = str(self.settings.path_terminal)

        if not MT5.initialize(**kwargs):
            raise RuntimeError(f"MT5.initialize() falló: {MT5.last_error()}")

        self._initialized = True

        # Modo dual: login por código solo si están las 3 variables
        if self.settings.login and self.settings.password and self.settings.server:
            if not MT5.login(
                login=int(self.settings.login),
                password=self.settings.password,
                server=self.settings.server,
            ):
                raise RuntimeError(f"MT5.login() falló: {MT5.last_error()}")
            self._authorized = True
        # Si no hay credenciales, asumimos sesión abierta y seguimos.

    def shutdown(self) -> None:
        """Cierra la conexión con MT5 (idempotente)."""
        if self._initialized:
            MT5.shutdown()
        self._initialized = False
        self._authorized = False

    # ---------- Datos: lista de dicts (UTC) ----------
    def copy_rates_range(
        self,
        symbol: str,
        timeframe: TF,
        start: datetime,
        end: datetime,
    ) -> List[dict]:
        """Obtiene velas entre fechas (UTC) como lista de dicts normalizados."""
        self._ensure_ready()
        self._validate_symbol(symbol)
        tf = _tf_to_mt5(timeframe)

        def _call():
            return MT5.copy_rates_range(symbol, tf, start, end)

        rates = self._with_retries(_call, op_name="copy_rates_range")
        if rates is None:
            raise RuntimeError(f"copy_rates_range retornó None. {MT5.last_error()}")
        return _normalize_rates_utc(rates)

    def copy_rates_from_pos(
        self,
        symbol: str,
        timeframe: TF,
        start_pos: int,
        count: int,
    ) -> List[dict]:
        """Obtiene N velas desde 'start_pos' (UTC) como lista de dicts normalizados."""
        self._ensure_ready()
        self._validate_symbol(symbol)
        tf = _tf_to_mt5(timeframe)

        def _call():
            return MT5.copy_rates_from_pos(symbol, tf, start_pos, count)

        rates = self._with_retries(_call, op_name="copy_rates_from_pos")
        if rates is None:
            raise RuntimeError(f"copy_rates_from_pos retornó None. {MT5.last_error()}")
        return _normalize_rates_utc(rates)

    # ---------- Helpers DataFrame (UTC) ----------
    @staticmethod
    def to_dataframe(rates: List[dict]):
        """
        Convierte la lista de dicts a pandas.DataFrame con índice 'time' en UTC (tz-aware).
        """
        try:
            import pandas as pd
        except Exception as e:
            raise ImportError("Falta dependencia 'pandas'. Instala 'pandas'.") from e

        if not rates:
            cols = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
            return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="time", tz="UTC"))

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], utc=True)  # UTC tz-aware
        df = df.set_index("time").sort_index()
        for c in ("open", "high", "low", "close"):
            df[c] = df[c].astype("float64")
        for c in ("tick_volume", "spread", "real_volume"):
            df[c] = df[c].astype("int64")
        return df

    def copy_rates_range_df(
        self,
        symbol: str,
        timeframe: TF,
        start: datetime,
        end: datetime,
    ):
        """Igual a copy_rates_range pero devuelve DataFrame indexado en UTC."""
        rates = self.copy_rates_range(symbol, timeframe, start, end)
        return self.to_dataframe(rates)

    def copy_rates_from_pos_df(
        self,
        symbol: str,
        timeframe: TF,
        start_pos: int,
        count: int,
    ):
        """Igual a copy_rates_from_pos pero devuelve DataFrame indexado en UTC."""
        rates = self.copy_rates_from_pos(symbol, timeframe, start_pos, count)
        return self.to_dataframe(rates)

    # ---------- Helpers internos ----------
    def _ensure_ready(self) -> None:
        if not self._initialized:
            raise RuntimeError("MT5 no inicializado. Llama a initialize().")

    def _validate_symbol(self, symbol: str) -> None:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Símbolo vacío o inválido.")
        if not _SYMBOL_RE.match(symbol):
            raise ValueError(f"Símbolo con caracteres no permitidos: {symbol!r}")
        if self.allowed_symbols is not None and symbol not in self.allowed_symbols:
            raise ValueError(f"Símbolo no permitido por allowlist: {symbol!r}")

    def _with_retries(self, func, op_name: str):
        """
        Ejecuta 'func' con reintentos y backoff exponencial simple.
        No reintenta si devuelve un objeto válido (aunque sea vacío).
        """
        attempt = 0
        last_exc = None
        while attempt < self.max_retries:
            try:
                result = func()
                if result is None:
                    raise RuntimeError(f"{op_name} devolvió None: {MT5.last_error()}")
                return result
            except Exception as e:
                last_exc = e
                attempt += 1
                if attempt >= self.max_retries:
                    break
                delay = self.backoff_base_sec * (2 ** (attempt - 1))
                time.sleep(delay)
        raise RuntimeError(f"{op_name} falló tras {self.max_retries} intentos: {last_exc}") from last_exc

# -------------------- Funciones privadas --------------------
def _normalize_rates_utc(rates) -> List[dict]:
    """
    Normaliza a tipos Python y convierte la marca de tiempo a UTC.
    """
    out: List[dict] = []
    for r in rates:
        out.append({
            "time": datetime.utcfromtimestamp(int(r["time"])),  # UTC
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "tick_volume": int(r["tick_volume"]),
            "spread": int(r["spread"]),
            "real_volume": int(r.get("real_volume", 0)),
        })
    return out

def _tf_to_mt5(tf: TF):
    if tf not in TF_MAP:
        raise ValueError(f"Timeframe no soportado: {tf}")
    return TF_MAP[tf]

def _env_int(name: str) -> Optional[int]:
    v = os.getenv(name)
    if v is None or v == "":
        return None
    return int(v)

def _env_path(name: str) -> Optional[Path]:
    v = os.getenv(name)
    if v:
        return Path(v)
    return None
