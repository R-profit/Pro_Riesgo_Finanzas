# %% 
#security]: Validación de inputs, rate-limit, y fingerprint sin PII (trazabilidad segura).
#data]: ProviderAdapter que prioriza MT5Provider del bot. Si no está disponible, usa YF DEMO; normaliza OHLC y fechas a UTC naive (coherente con tu estándar).
#analytics]:

#construye el portafolio ponderado alineando fechas.
#calcula Base 100 y KPIs (retorno y volatilidad anualizados, Sharpe, max drawdown).
#a anualización usa _infer_freq_per_year(timeframe) (M1…D1) → métrica consistente por timeframe.
#[viz]: Gráfico comparativo Base 100 listo para insertar en PDF o UI (sin plt.show() aquí).
#[export]: PDF con portada de KPIs + figura comparativa, y PNG de la figura.
#[app]: comparar_y_exportar(...) orquesta todo y devuelve un payload JSON-friendly para que el runner lo use (y opcionalmente exporta archivos).
#[run]: Smoke test DEMO independiente (NO toca MT5). Perfecto para verificar que todo corre antes de integrarlo.
# limites, dias bursatiles ,tasa libree. rutas etc
# %% 
from dataclasses import dataclass
from datetime import timedelta

@dataclass(frozen=True)
class AppConfig:
    RATE_LIMIT_CALLS: int = 8
    RATE_LIMIT_PERIOD: timedelta = timedelta(seconds=60)
    MAX_DOWNLOAD_YEARS: int = 15
    # Este valor queda como “valor por defecto” sólo para D1;
    # la anualización real se ajusta por timeframe en _infer_freq_per_year.
    TRADING_DAYS: int = 252
    FIG_RATIO = (12, 5)
    RISK_FREE: float = 0.02  # Tasa libre anual (ajústala)
    EXPORT_DIR: str = "exports"

CONFIG = AppConfig()
# %%


# %% [security] Validación, rate-limit y fingerprint (sin PII)
import hashlib, json, platform, re, threading, time, os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, Tuple

def build_fingerprint() -> Dict[str, str]:
    """Fingerprint técnico sin PII para trazabilidad."""
    meta = {
        "py": platform.python_version(),
        "os": platform.system(),
        "rel": platform.release(),
        "arch": platform.machine(),
        "host_hash": hashlib.sha256(platform.node().encode()).hexdigest()[:12],
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    meta["fp"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    return meta

@dataclass
class RateWindow:
    max_calls: int
    period_s: float

class RateLimiter:
    """Rate limiter por ventana deslizante, seguro para threads."""
    def __init__(self):
        self._locks: Dict[str, threading.Lock] = {}
        self._buckets: Dict[str, Deque[float]] = {}
        self._windows: Dict[str, RateWindow] = {}

    def register(self, key: str, max_calls: int, period_s: float):
        self._locks[key] = threading.Lock()
        self._buckets[key] = deque()
        self._windows[key] = RateWindow(max_calls, period_s)

    def limit(self, key: str):
        def deco(fn):
            def wrapper(*args, **kwargs):
                lock = self._locks[key]; bucket = self._buckets[key]; window = self._windows[key]
                now = time.time()
                with lock:
                    while bucket and now - bucket[0] >= window.period_s:
                        bucket.popleft()
                    if len(bucket) >= window.max_calls:
                        sleep_for = window.period_s - (now - bucket[0])
                        if sleep_for > 0:
                            time.sleep(min(sleep_for, window.period_s))
                        now = time.time()
                        while bucket and now - bucket[0] >= window.period_s:
                            bucket.popleft()
                    bucket.append(time.time())
                return fn(*args, **kwargs)
            return wrapper
        return deco

RATE_LIMITER = RateLimiter()

_TICKER_RE = re.compile(r"^[A-Za-z0-9\.\^\-\_/=]{1,30}$")

class ValidationError(ValueError):
    """Errores de validación de entradas."""

def validate_ticker(sym: str) -> str:
    """Valida ticker: no vacío y caracteres permitidos."""
    if not isinstance(sym, str) or not sym.strip():
        raise ValidationError("Ticker vacío.")
    if not _TICKER_RE.match(sym.strip()):
        raise ValidationError(f"Ticker inválido: {sym}")
    return sym.strip()

def validate_dates(start: str, end: str, *, max_years: int) -> Tuple[str, str]:
    """Valida fechas ISO y rango máximo en años."""
    try:
        s = datetime.fromisoformat(start); e = datetime.fromisoformat(end)
    except Exception as ex:
        raise ValidationError(f"Formato de fecha inválido (YYYY-MM-DD). Detalle: {ex}")
    if s >= e:
        raise ValidationError("start debe ser anterior a end.")
    if (e - s).days > max_years * 365:
        raise ValidationError(f"Rango excesivo (> {max_years} años).")
    return start, end

def ensure_dir(path: str):
    """Crea carpeta si no existe (para exportaciones)."""
    os.makedirs(path, exist_ok=True)
# %%


# %% [data] Capa de datos (provider-agnóstico: MT5/YF; DEMO fallback YF)
from typing import List, Dict, Optional
import pandas as pd  # type: ignore

# Mapa de timeframes del bot a intervalos YF solo para DEMO fallback
_TF_YF = {
    "M1":  "1m",
    "M5":  "5m",
    "M15": "15m",
    "M30": "30m",
    "H1":  "60m",
    "H4":  "240m",
    "D1":  "1d",
}

# Opcional: rate limit para el fallback YF
RATE_LIMITER.register(
    key="yfinance_download",
    max_calls=CONFIG.RATE_LIMIT_CALLS,
    period_s=CONFIG.RATE_LIMIT_PERIOD.total_seconds(),
)

class ProviderAdapter:
    """
    Adaptador mínimo:
      - Si existe bot_runner.providers.<mt5|yf> usa su interfaz get_df(symbol, timeframe, ...)
      - Si no, usa yfinance como DEMO (sin credenciales broker).
    Retorno esperado: DataFrame con columnas al menos ['open','high','low','close'] (lowercase).
    """
    def __init__(self, prefer_mt5: bool = True):
        self.prefer_mt5 = prefer_mt5
        self._bot_provider = self._load_bot_provider()

    def _load_bot_provider(self):
        try:
            if self.prefer_mt5:
                from bot_runner.providers.mt5_provider import MT5Provider  # type: ignore
                return MT5Provider()
            else:
                from bot_runner.providers.yf_provider import YFProvider  # type: ignore
                return YFProvider()
        except Exception:
            return None  # no hay provider del bot disponible

    def get_df(self, symbol: str, timeframe: str, *, start: Optional[str] = None,
               end: Optional[str] = None, count: Optional[int] = None) -> pd.DataFrame:
        # 0) Validaciones de entrada básicas
        symbol = validate_ticker(symbol)
        if start and end:
            validate_dates(start, end, max_years=CONFIG.MAX_DOWNLOAD_YEARS)

        # 1) Provider del bot disponible
        if self._bot_provider is not None:
            df = self._bot_provider.get_df(symbol, timeframe, start=start, end=end, count=count)
            return self._normalize(df)

        # 2) DEMO fallback con yfinance
        try:
            import yfinance as yf  # type: ignore
        except Exception:
            raise RuntimeError("No hay provider del bot ni yfinance disponible para DEMO.")

        interval = _TF_YF.get(timeframe.upper(), "1d")

        @RATE_LIMITER.limit("yfinance_download")
        def _download():
            return yf.download(
                tickers=symbol, start=start, end=end,
                interval=interval, progress=False, auto_adjust=False, threads=False
            )

        df = _download()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        # normaliza nombres
        colmap = {
            "Open": "open", "High": "high", "Low": "low", "Close": "close",
            "Adj Close": "adj_close", "Volume": "volume"
        }
        df = df.rename(columns={c: colmap.get(str(c), str(c).lower()) for c in df.columns})
        return self._normalize(df)

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        d = df.copy()
        d.columns = [str(c).lower() for c in d.columns]
        # requerimos ohlc mínimo
        for c in ("open", "high", "low", "close"):
            if c not in d.columns:
                # si viene 'price' la mapeamos a close
                if c == "close" and "price" in d.columns:
                    d["close"] = d["price"]
                else:
                    # si falta algo esencial, devolvemos vacío
                    return pd.DataFrame()
        if not isinstance(d.index, pd.DatetimeIndex):
            d.index = pd.to_datetime(d.index, errors="coerce")
        # normaliza a UTC naive (como acordamos)
        if d.index.tz is not None:
            d = d.tz_convert("UTC").tz_localize(None)
        return d.sort_index().dropna(how="any")
# %%


# %% [analytics] Comparación Activo vs Portafolio (KPIs + Base100)
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np  # type: ignore

@dataclass(frozen=True)
class PortfolioSpec:
    tickers: List[str]
    weights: List[float]   # deben sumar 1.0

def _infer_freq_per_year(timeframe: str) -> int:
    """Frecuencia por año según timeframe (anualización)."""
    tf = timeframe.upper()
    if tf == "D1":  return 252
    if tf == "H4":  return 6*5*52
    if tf == "H1":  return 24*5*52
    if tf == "M30": return 2*24*5*52
    if tf == "M15": return 4*24*5*52
    if tf == "M5":  return 12*24*5*52
    if tf == "M1":  return 60*24*5*52
    return 252

class ComparatorService:
    """
    Lógica de negocio:
    - Elige columna de cierre (adj_close > close si existe)
    - Construye serie del portafolio con pesos
    - Devuelve DataFrame Base 100 + KPIs del activo y del portafolio
    """

    @staticmethod
    def _choose_close(df: pd.DataFrame) -> pd.Series:
        cols = {c.lower(): c for c in df.columns}
        if "adj_close" in cols:
            return df[cols["adj_close"]].astype(float)
        if "close" in cols:
            return df[cols["close"]].astype(float)
        raise KeyError("No se encuentran columnas 'adj_close' ni 'close'.")

    @staticmethod
    def _base100(series: pd.Series) -> pd.Series:
        s = series.dropna()
        if s.empty or s.iloc[0] == 0:
            raise ValueError("Serie inválida para Base 100.")
        return (s / s.iloc[0]) * 100.0

    @staticmethod
    def _kpis_from_prices(px: pd.Series, fpy: int, risk_free_ann: float) -> Dict[str, float]:
        px = px.dropna()
        rets = px.pct_change().dropna()
        if rets.empty:
            return {"ret_annual": np.nan, "vol_annual": np.nan, "sharpe": np.nan, "max_dd": np.nan}

        mu = rets.mean() * fpy
        sigma = rets.std(ddof=0) * np.sqrt(fpy)
        sharpe = (mu - risk_free_ann) / sigma if sigma > 0 else np.nan

        cum = (1 + rets).cumprod()
        dd = (cum / cum.cummax()) - 1.0
        max_dd = dd.min()

        return {"ret_annual": float(mu), "vol_annual": float(sigma), "sharpe": float(sharpe), "max_dd": float(max_dd)}

    @staticmethod
    def build_portfolio_close(price_map: Dict[str, pd.Series], weights: List[float]) -> pd.Series:
        """Compone portafolio usando pesos; alinea por intersección de fechas."""
        if not price_map:
            raise ValueError("price_map vacío.")
        if len(price_map) != len(weights):
            raise ValueError("tickers y weights con longitudes distintas.")
        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("Los pesos del portafolio deben sumar 1.0")

        df = pd.DataFrame(price_map).dropna(how="any").sort_index()
        if df.empty:
            raise ValueError("No hay intersección de fechas entre los activos del portafolio.")

        norm = df.divide(df.iloc[0], axis=1)              # normaliza cada serie a su inicio
        port_rel = (norm * np.array(weights)).sum(axis=1) # serie base 1.0 relativa
        return port_rel

    def compare(self, asset_df: pd.DataFrame, portfolio_dfs: Dict[str, pd.DataFrame],
                port_weights: List[float], timeframe: str = "D1") -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:

        asset_close = self._choose_close(asset_df)
        port_prices = {k: self._choose_close(v) for k, v in portfolio_dfs.items()}
        port_rel = self.build_portfolio_close(port_prices, port_weights)  # base 1.0

        # Alinear activo y portafolio por intersección de fechas
        df = pd.concat({"asset": asset_close, "port_rel": port_rel}, axis=1, join="inner").dropna()
        if df.empty:
            raise ValueError("Sin intersección de fechas entre activo y portafolio.")

        # Base 100
        asset_b100 = self._base100(df["asset"])
        port_b100  = (df["port_rel"] / df["port_rel"].iloc[0]) * 100.0
        df_b100    = pd.DataFrame({"Asset_100": asset_b100, "Portfolio_100": port_b100}).dropna()

        fpy = _infer_freq_per_year(timeframe)
        k_asset = self._kpis_from_prices(df["asset"], fpy=fpy, risk_free_ann=CONFIG.RISK_FREE)
        k_port  = self._kpis_from_prices(df["port_rel"] * 100.0, fpy=fpy, risk_free_ann=CONFIG.RISK_FREE)

        return df_b100, {"asset": k_asset, "portfolio": k_port}
# %%


# %% [viz] Gráficos (sin plt.show; listos para app o bot)
import matplotlib.pyplot as plt  # type: ignore

def plot_comparison_base100(df_b100: pd.DataFrame, title: str = "Comparación Base 100"):
    required = {"Asset_100", "Portfolio_100"}
    assert required.issubset(df_b100.columns), "Columnas requeridas no están presentes."
    fig = plt.figure(figsize=CONFIG.FIG_RATIO)
    ax = fig.add_subplot(111)
    ax.plot(df_b100.index, df_b100["Asset_100"], label="Activo (Base 100)")
    ax.plot(df_b100.index, df_b100["Portfolio_100"], label="Portafolio (Base 100)")
    ax.set_title(title)
    ax.set_xlabel("Fecha"); ax.set_ylabel("Índice")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig
# %%


# %% [export] Exportación PDF/PNG (portada + figura)
from typing import Dict
from matplotlib.backends.backend_pdf import PdfPages  # type: ignore

def _figure_cover(meta: Dict, kpis: Dict[str, Dict[str, float]]):
    """Portada PDF con metadatos y KPIs del activo/portafolio."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Reporte de Comparación Activo vs Portafolio",
                 fontsize=18, fontweight="bold", y=0.95)

    def _fmt_pct(x: float) -> str: return f"{x:.2%}" if x == x else "n/a"
    def _fmt_shp(x: float) -> str: return f"{x:.2f}" if x == x else "n/a"
    def _fmt_w(ws) -> str:
        try: return "[" + ", ".join(f"{w:.4f}" for w in ws) + "]"
        except Exception: return str(ws)

    _meta_rows = [
        ["Activo",            f"{meta['asset']}"],
        ["Portafolio",        ", ".join(meta['portfolio']['tickers'])],
        ["Pesos",             _fmt_w(meta['portfolio']['weights'])],
        ["Rango",             f"{meta['start']} \u2192 {meta['end']}"],
        ["Timeframe",         f"{meta.get('timeframe','D1')}"],
        ["Fingerprint",       f"{meta['fingerprint']}"],
    ]
    _asset_rows = [
        ["Retorno anualizado", _fmt_pct(kpis['asset']['ret_annual'])],
        ["Vol anualizada",     _fmt_pct(kpis['asset']['vol_annual'])],
        ["Sharpe",             _fmt_shp(kpis['asset']['sharpe'])],
        ["Max Drawdown",       _fmt_pct(kpis['asset']['max_dd'])],
    ]
    _port_rows = [
        ["Retorno anualizado", _fmt_pct(kpis['portfolio']['ret_annual'])],
        ["Vol anualizada",     _fmt_pct(kpis['portfolio']['vol_annual'])],
        ["Sharpe",             _fmt_shp(kpis['portfolio']['sharpe'])],
        ["Max Drawdown",       _fmt_pct(kpis['portfolio']['max_dd'])],
    ]

    # Layout simple: 3 bloques
    ax_meta = fig.add_axes([0.06, 0.53, 0.88, 0.34])
    ax_a    = fig.add_axes([0.06, 0.12, 0.40, 0.34])
    ax_p    = fig.add_axes([0.54, 0.12, 0.40, 0.34])
    for _ax in (ax_meta, ax_a, ax_p):
        _ax.axis("off")

    def _draw_table(_ax, _title: str, _rows: list, _labels=("Métrica", "Valor")):
        _ax.text(0.0, 1.05, _title, transform=_ax.transAxes,
                 ha="left", va="bottom", fontsize=13, fontweight="bold")
        _tbl = _ax.table(cellText=_rows, colLabels=_labels, loc="center",
                         cellLoc="left", colLoc="left", bbox=[0.0, 0.0, 1.0, 1.0])
        _tbl.auto_set_font_size(False); _tbl.set_fontsize(11); _tbl.scale(1.0, 1.25)
        for (r, c), cell in _tbl.get_celld().items():
            cell.set_linewidth(0.8); cell.set_edgecolor("black")
            if r == 0: cell.set_facecolor("#f0f0f0"); cell.set_text_props(fontweight="bold")
            elif r % 2 == 0: cell.set_facecolor("#fafafa")
        return _tbl

    _draw_table(ax_meta, "Metadatos", _meta_rows, ("Campo", "Valor"))
    _draw_table(ax_a,   "KPIs Activo", _asset_rows)
    _draw_table(ax_p,   "KPIs Portafolio", _port_rows)
    return fig

def export_png(fig, path_png: str):
    """Guarda la figura de comparación en PNG."""
    ensure_dir(os.path.dirname(path_png) or ".")
    fig.savefig(path_png, dpi=150, bbox_inches="tight")

def export_pdf(meta: Dict, kpis: Dict[str, Dict[str, float]], fig_comp, path_pdf: str):
    """Genera PDF: portada + figura de comparación (NO cierra fig_comp)."""
    ensure_dir(os.path.dirname(path_pdf) or ".")
    with PdfPages(path_pdf) as pdf:
        cover = _figure_cover(meta, kpis)
        pdf.savefig(cover); plt.close(cover)   # cerramos sólo la portada
        pdf.savefig(fig_comp)                  # NO cerrar la figura comparativa
# %%


# %% [app] Orquestador (listo para bot/VSCode; devuelve payload JSON-friendly)
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("comparador")

def comparar_y_exportar(
    asset_ticker: str,
    portfolio_tickers: List[str],
    portfolio_weights: List[float],
    start: str,
    end: str,
    timeframe: str = "D1",
    export_basename: str = None,
):
    """Pipeline listo para bot: usa ProviderAdapter (MT5 o YF DEMO)."""
    fp = build_fingerprint()
    log.info(f"Fingerprint env: {fp['fp']} ({fp['py']} | {fp['os']})")

    provider = ProviderAdapter(prefer_mt5=True)  # usa MT5 si está disponible
    comp = ComparatorService()

    # Descarga (provider-agnóstico)
    asset_df = provider.get_df(asset_ticker, timeframe, start=start, end=end)
    port_map = {t: provider.get_df(t, timeframe, start=start, end=end) for t in portfolio_tickers}

    # Comparación
    df_b100, kpis = comp.compare(asset_df, port_map, portfolio_weights, timeframe=timeframe)

    # Figura
    fig = plot_comparison_base100(df_b100, title=f"{asset_ticker} vs Portafolio (Base 100) — {timeframe}")

    payload = {
        "meta": {
            "fingerprint": fp["fp"],
            "start": start,
            "end": end,
            "timeframe": timeframe,
            "asset": asset_ticker,
            "portfolio": {"tickers": portfolio_tickers, "weights": portfolio_weights},
        },
        "kpis": kpis,
        "sample": df_b100.tail(5).to_dict(orient="index"),
    }

    if export_basename:
        ensure_dir(CONFIG.EXPORT_DIR)
        png_path = os.path.join(CONFIG.EXPORT_DIR, f"{export_basename}.png")
        pdf_path = os.path.join(CONFIG.EXPORT_DIR, f"{export_basename}.pdf")
        export_png(fig, png_path)
        export_pdf(payload["meta"], kpis, fig, pdf_path)
        log.info(f"PNG → {png_path}")
        log.info(f"PDF → {pdf_path}")

    return {"payload": payload, "df_base100": df_b100, "figure": fig}
# %%


# %% [run] Smoke test DEMO (no usa broker; útil para validar rápido)
def main() -> int:
    asset = "GC=F"  # oro (YF DEMO)
    ptf  = ["SI=F", "CL=F"]  # plata y crudo (YF DEMO)
    wgt  = [0.5, 0.5]
    start, end = "2024-01-01", "2025-10-01"
    timeframe = "D1"  # prueba "M15"/"H1" si tu provider real lo soporta

    out = comparar_y_exportar(
        asset_ticker=asset,
        portfolio_tickers=ptf,
        portfolio_weights=wgt,
        start=start,
        end=end,
        timeframe=timeframe,
        export_basename=f"{asset}_vs_PORT_{timeframe}_{start}_{end}"
    )
    print("KPIs:", out["payload"]["kpis"])
    print("Tail df_base100:", list(out["payload"]["sample"].items())[:2])

    import matplotlib.pyplot as plt  # type: ignore
    plt.show()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
