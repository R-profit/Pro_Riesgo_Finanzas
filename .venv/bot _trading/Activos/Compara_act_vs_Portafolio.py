# %% [config] Configuración parametros generales
# limites, dias bursatiles ,tasa libree. rutas etc

from dataclasses import dataclass
from datetime import timedelta

@dataclass(frozen=True)
class AppConfig:
    RATE_LIMIT_CALLS: int = 8
    RATE_LIMIT_PERIOD: timedelta = timedelta(seconds=60)
    MAX_DOWNLOAD_YEARS: int = 15
    TRADING_DAYS: int = 252
    FIG_RATIO = (12, 5)
    YF_INTERVAL: str = "1d"
    RISK_FREE: float = 0.02  # Tasa libre anual (ajústala)
    EXPORT_DIR: str = "exports"

CONFIG = AppConfig()
# %%

# %% [security] Helpers: validación, rate-limit y fingerprint

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

# %%# %% [data] Capa de datos (yfinance)
# descarga normaliza datos columnas/multilindex

from typing import List
import pandas as pd # type: ignore
import yfinance as yf # type: ignore

# Registra el rate limit para yfinance
RATE_LIMITER.register(
    key="yfinance_download",
    max_calls=CONFIG.RATE_LIMIT_CALLS,
    period_s=CONFIG.RATE_LIMIT_PERIOD.total_seconds(),
)

class DataLoader:
    """Descarga datos de mercado con validación, rate-limit y normalización."""

    @RATE_LIMITER.limit("yfinance_download")
    def download(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        ticker = validate_ticker(ticker)
        validate_dates(start, end, max_years=CONFIG.MAX_DOWNLOAD_YEARS)

        df = yf.download(
            tickers=ticker,
            start=start, end=end,
            interval=CONFIG.YF_INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False
        )
        if df is None or df.empty:
            raise RuntimeError(f"Sin datos para {ticker} entre {start} y {end}.")

        # Aplana MultiIndex si viene por múltiples tickers/fields
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, axis=1, level=1, drop_level=True)
            except Exception:
                df = df.copy()
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        # Normaliza nombres de columnas
        colmap = {"adj_close": "Adj Close", "adj close": "Adj Close",
                  "open": "Open", "high": "High", "low": "Low",
                  "close": "Close", "volume": "Volume"}
        df = df.rename(columns={c: colmap.get(str(c).lower(), c) for c in df.columns})
        df = df.sort_index().dropna(how="all")
        return df

    def download_many(self, tickers: List[str], start: str, end: str) -> dict:
        """Descarga varios tickers y devuelve dict[ticker]=DataFrame."""
        return {t: self.download(t, start, end) for t in tickers}
    
# %%

# %%# %% [analytics] Comparación Activo vs Portafolio (KPIs + Base100)
# construye portafollio ponderado, alinea fechas, calcula base 100 y KPIS(retorno, vol, sharpe, max drawd
# om)  

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np # type: ignore

@dataclass(frozen=True)
class PortfolioSpec:
    tickers: List[str]
    weights: List[float]   # deben sumar 1.0

class ComparatorService:
    """
    Lógica de negocio:
    - Elige columna de cierre (Adj Close > Close)
    - Construye serie del portafolio con pesos
    - Devuelve DataFrame Base 100 + KPIs del activo y del portafolio
    """

    @staticmethod
    def _choose_close(df: pd.DataFrame) -> pd.Series:
        if "Adj Close" in df.columns:
            return df["Adj Close"].astype(float)
        if "Close" in df.columns:
            return df["Close"].astype(float)
        raise KeyError("No se encuentran columnas 'Adj Close' ni 'Close'.")

    @staticmethod
    def _base100(series: pd.Series) -> pd.Series:
        s = series.dropna()
        if s.empty or s.iloc[0] == 0:
            raise ValueError("Serie inválida para Base 100.")
        return (s / s.iloc[0]) * 100.0

    @staticmethod
    def _kpis_from_prices(px: pd.Series) -> Dict[str, float]:
        px = px.dropna()
        rets = px.pct_change().dropna()
        if rets.empty:
            return {"ret_annual": np.nan, "vol_annual": np.nan, "sharpe": np.nan, "max_dd": np.nan}

        mu = rets.mean() * CONFIG.TRADING_DAYS
        sigma = rets.std(ddof=0) * np.sqrt(CONFIG.TRADING_DAYS)
        sharpe = (mu - CONFIG.RISK_FREE) / sigma if sigma > 0 else np.nan

        cum = (1 + rets).cumprod()
        rolling_max = cum.cummax()
        dd = (cum / rolling_max) - 1.0
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

        norm = df.divide(df.iloc[0], axis=1)         # normaliza cada serie a su inicio
        port_rel = (norm * np.array(weights)).sum(axis=1)  # serie base 1.0 relativa
        return port_rel

    def compare(self, asset_df: pd.DataFrame, portfolio_dfs: Dict[str, pd.DataFrame],
                port_weights: List[float]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:

        asset_close = self._choose_close(asset_df)
        port_prices = {k: self._choose_close(v) for k, v in portfolio_dfs.items()}
        port_rel = self.build_portfolio_close(port_prices, port_weights)  # base 1.0

        # Alinear activo y portafolio por intersección de fechas
        df = pd.concat({"asset": asset_close, "port_rel": port_rel}, axis=1, join="inner").dropna()
        if df.empty:
            raise ValueError("Sin intersección de fechas entre activo y portafolio.")

        # Base 100
        asset_b100 = self._base100(df["asset"])
        port_b100 = (df["port_rel"] / df["port_rel"].iloc[0]) * 100.0

        df_b100 = pd.DataFrame({"Asset_100": asset_b100, "Portfolio_100": port_b100}).dropna()

        # KPIs en “precios” (evita sesgo del reescalado)
        k_asset = self._kpis_from_prices(df["asset"])
        k_port = self._kpis_from_prices(df["port_rel"] * 100.0)  # escala a “precio” ficticio

        return df_b100, {"asset": k_asset, "portfolio": k_port}
    
# %%
# %% [viz] Gráficos
# crea figura de comparacion base 100(sin plt.show)

import matplotlib.pyplot as plt # type: ignore

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

# %% [export] Exportación PDF/PNG
# Exporta png del grafico y pdf con portada(metadatos + kpis) + grafico

import os
from typing import Dict
import matplotlib.pyplot as plt # type: ignore
from matplotlib.backends.backend_pdf import PdfPages # type: ignore

def _figure_cover(meta: Dict, kpis: Dict[str, Dict[str, float]]):
    """Portada PDF con metadatos y KPIs del activo/portafolio."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Reporte de Comparación Activo vs Portafolio",
                 fontsize=18, fontweight="bold", y=0.95)
    lines = [
        f"Activo: {meta['asset']}",
        f"Portafolio: {', '.join(meta['portfolio']['tickers'])}",
        f"Pesos: {meta['portfolio']['weights']}",
        f"Rango: {meta['start']} → {meta['end']}",
        f"Fingerprint: {meta['fingerprint']}",
        "",
        "KPIs Activo:",
        f"  Retorno anualizado: {kpis['asset']['ret_annual']:.2%}",
        f"  Vol anualizada:     {kpis['asset']['vol_annual']:.2%}",
        f"  Sharpe:             {kpis['asset']['sharpe']:.2f}",
        f"  Max Drawdown:       {kpis['asset']['max_dd']:.2%}",
        "",
        "KPIs Portafolio:",
        f"  Retorno anualizado: {kpis['portfolio']['ret_annual']:.2%}",
        f"  Vol anualizada:     {kpis['portfolio']['vol_annual']:.2%}",
        f"  Sharpe:             {kpis['portfolio']['sharpe']:.2f}",
        f"  Max Drawdown:       {kpis['portfolio']['max_dd']:.2%}",
    ]
        
    def _fmt_pct(x: float) -> str: return f"{x:.2%}"
    def _fmt_shp(x: float) -> str: return f"{x:.2f}"
    def _fmt_w(ws) -> str:
        try: return "[" + ", ".join(f"{w:.4f}" for w in ws) + "]"
        except Exception: return str(ws)

    _meta_rows = [
        ["Activo",            f"{meta['asset']}"],
        ["Portafolio",        ", ".join(meta['portfolio']['tickers'])],
        ["Pesos",             _fmt_w(meta['portfolio']['weights'])],
        ["Rango",             f"{meta['start']} \u2192 {meta['end']}"],
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

    # Layout: Metadatos arriba (ancho completo), dos tablas abajo (lado a lado)
    ax_meta = fig.add_axes([0.06, 0.53, 0.88, 0.34])  # [left, bottom, width, height]
    ax_a    = fig.add_axes([0.06, 0.12, 0.40, 0.34])
    ax_p    = fig.add_axes([0.54, 0.12, 0.40, 0.34])
    for _ax in (ax_meta, ax_a, ax_p):
        _ax.axis("off")

    def _draw_table(_ax, _title: str, _rows: list, _labels=("Métrica", "Valor")):
        # Título del bloque
        _ax.text(0.0, 1.05, _title, transform=_ax.transAxes,
                 ha="left", va="bottom", fontsize=13, fontweight="bold")
        # Tabla con celdas y bordes
        _tbl = _ax.table(
            cellText=_rows,
            colLabels=_labels,
            loc="center",
            cellLoc="left",
            colLoc="left",
            bbox=[0.0, 0.0, 1.0, 1.0]
        )
        _tbl.auto_set_font_size(False)
        _tbl.set_fontsize(11)
        _tbl.scale(1.0, 1.25)
        # Estilo: bordes + header sombreado + zebra rows
        for (r, c), cell in _tbl.get_celld().items():
            cell.set_linewidth(0.8)
            cell.set_edgecolor("black")
            if r == 0:
                cell.set_facecolor("#f0f0f0")
                cell.set_text_props(fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#fafafa")
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
# %% [app] Orquestador para bot/VSCode
# funcion para descargar, comparar, graficar y exportar si se pide, devuelve payloas jason-frendly+ df_base 100+ figura
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
    export_basename: str = None,
):
    """Pipeline completo y listo para integrarse en un bot."""
    fp = build_fingerprint()
    log.info(f"Fingerprint env: {fp['fp']} ({fp['py']} | {fp['os']})")

    loader = DataLoader()
    comp = ComparatorService()

    # Descarga
    asset_df = loader.download(asset_ticker, start, end)
    port_map = loader.download_many(portfolio_tickers, start, end)

    # Comparación
    df_b100, kpis = comp.compare(asset_df, port_map, portfolio_weights)

    # Figura
    fig = plot_comparison_base100(df_b100, title=f"{asset_ticker} vs Portafolio (Base 100)")

    # Payload serializable
    payload = {
        "meta": {
            "fingerprint": fp["fp"],
            "start": start,
            "end": end,
            "asset": asset_ticker,
            "portfolio": {"tickers": portfolio_tickers, "weights": portfolio_weights},
        },
        "kpis": kpis,
        "sample": df_b100.tail(5).to_dict(orient="index"),
    }

    # Exportaciones opcionales
    if export_basename:
        ensure_dir(CONFIG.EXPORT_DIR)
        png_path = os.path.join(CONFIG.EXPORT_DIR, f"{export_basename}.png")
        pdf_path = os.path.join(CONFIG.EXPORT_DIR, f"{export_basename}.pdf")
        export_png(fig, png_path)                 # guarda PNG
        export_pdf(payload["meta"], kpis, fig, pdf_path)  # guarda PDF (y cierra fig)
        log.info(f"PNG → {png_path}")
        log.info(f"PDF → {pdf_path}")

    return {"payload": payload, "df_base100": df_b100, "figure": fig}

# %%
# %% [run] Ejemplo listo para correr
# ejemplo end-to-end. cambia tickers, fechas,pesos y se exporta a ./exports/.

if __name__ == "__main__":
    asset = "TSLA"
    ptf  = ["AAPL", "MSFT", "NVDA"]
    wgt  = [0.4, 0.3, 0.3]
    start, end = "2024-01-01", "2025-10-01"

    out = comparar_y_exportar(
        asset_ticker=asset,
        portfolio_tickers=ptf,
        portfolio_weights=wgt,
        start=start,
        end=end,
        export_basename=f"{asset}_vs_PORTFOLIO_{start}_{end}"
    )

    # Si quieres ver la figura manualmente en modo script:
    # import matplotlib.pyplot as plt; plt.show()
    print("KPIs:", out["payload"]["kpis"])
    print("Muestra tail df_base100:", list(out["payload"]["sample"].items())[:2])
    # ——— AÑADE ESTO PARA MOSTRAR LA GRÁFICA ———
import matplotlib.pyplot as plt # type: ignore
plt.show()

# Si exportaste PDF/PNG, la fig sigue viva porque ya no la cerramos en export_pdf
# (y export_png no cierra la figura). La mostramos ahora:
try:
    # (Opcional) cambia el título de la ventana si tu backend lo soporta
    if hasattr(out["figure"].canvas, "set_window_title"):
        out["figure"].canvas.set_window_title("Comparación Activo vs Portafolio")
except Exception:
    pass

# %%
