
# Autor: Rafy (R_Profit)
# Descripción: Cálculo de indicadores, generación de señales, visualización y análisis de rendimiento.
# ===============================================================
# %% ======================================================
# Núcleo de Indicadores & Señales para el BOT (provider-first)
# - Sin descargas, sin gráficos, sin main()
# - Requiere: DataFrame con 'open','high','low','close' (minúsculas)
# - Devuelve: indicadores y señales vectorizadas
# =========================================================

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# %% ---------------- Seguridad ligera (sin prints) ----------------
__all__ = [
    "IndicatorSignalEngine",
    "compute_indicators",
    "generate_signals",
    "build_signal_rows",
]

try:
    import ta  # type: ignore
    _HAS_TA = True
except Exception:
    _HAS_TA = False

# %% ----------------------- Helpers base --------------------------

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def _macd_fallback(close: pd.Series, fast=12, slow=26, signal=9):
    ema_f = _ema(close, fast); ema_s = _ema(close, slow)
    macd = ema_f - ema_s
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def _rsi_fallback(close: pd.Series, period=14):
    delta = close.diff()
    up = np.clip(delta, 0, None)
    dn = -np.clip(delta, None, 0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_dn.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def _bollinger_fallback(close: pd.Series, window=20, dev=2.0):
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std()
    upper = ma + dev * sd
    lower = ma - dev * sd
    return ma, upper, lower

# %% -------------------- Cálculo de indicadores -------------------

def compute_indicators(
    df: pd.DataFrame,
    ema_fast: int = 12,
    ema_slow: int = 26,
    rsi_period: int = 14,
    bb_window: int = 20,
    bb_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Salidas: close, EMA_short/long, MACD/Signal/Hist, RSI, BOL_M/U/D, BandWidth, BW_MA, BW_diff.
    """
    need = {"open", "high", "low", "close"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    close = df["close"].astype("float64")
    out["close"] = close

    out["EMA_short"] = _ema(close, ema_fast)
    out["EMA_long"]  = _ema(close, ema_slow)

    if _HAS_TA:
        macd = ta.trend.MACD(close=close, window_fast=ema_fast, window_slow=ema_slow, window_sign=9)
        out["MACD"]        = macd.macd()
        out["Signal_Line"] = macd.macd_signal()
        out["MACD_Hist"]   = macd.macd_diff()
    else:
        out["MACD"], out["Signal_Line"], out["MACD_Hist"] = _macd_fallback(close, ema_fast, ema_slow, 9)

    if _HAS_TA:
        out["RSI"] = ta.momentum.RSIIndicator(close=close, window=rsi_period).rsi()
    else:
        out["RSI"] = _rsi_fallback(close, rsi_period)

    if _HAS_TA:
        bb = ta.volatility.BollingerBands(close=close, window=bb_window, window_dev=bb_dev)
        out["BOL_M"] = bb.bollinger_mavg()
        out["BOL_U"] = bb.bollinger_hband()
        out["BOL_D"] = bb.bollinger_lband()
    else:
        out["BOL_M"], out["BOL_U"], out["BOL_D"] = _bollinger_fallback(close, bb_window, bb_dev)

    out["BandWidth"] = (out["BOL_U"] - out["BOL_D"]).astype("float64")
    out["BW_MA"] = out["BandWidth"].rolling(window=20, min_periods=10).mean()
    out["BW_diff"] = out["BandWidth"].diff()
    
    # Medtrica simple de muy baja o excesiva volatilidad
    # --- ATR% y rango (régimen) ---
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean()
    out["ATR"] = atr
    out["ATR_pct"] = (atr / out["close"]).astype("float64")
    # compacidad de rango (0=compacto, 1=amplio) – aproximación
    roll = 20
    out["range_compact"] = ((out["BOL_U"] - out["BOL_D"]) / out["close"]).rolling(roll, min_periods=roll).mean()

    return out.dropna()
# %%

# %% -------------------- Generación de señales --------------------

def generate_signals(
    feats: pd.DataFrame,
    ema_fast: int = 12, ema_slow: int = 26,
    rsi_buy: float = 55.0, rsi_sell: float = 45.0,
    width_thresh: float = 0.10,
) -> pd.DataFrame:
    if feats.empty:
        return feats

    ema_s = feats["EMA_short"]; ema_l = feats["EMA_long"]
    macd = feats["MACD"]; sig = feats["Signal_Line"]
    rsi  = feats["RSI"]
    bol_u = feats["BOL_U"]; bol_d = feats["BOL_D"]
    bw = feats["BandWidth"]; bw_ma = feats["BW_MA"]; bw_diff = feats["BW_diff"]
    px = feats["close"]

    cross_up = (ema_s > ema_l) & (ema_s.shift(1) <= ema_l.shift(1))
    cross_dn = (ema_s < ema_l) & (ema_s.shift(1) >= ema_l.shift(1))
    macd_buy = macd > sig
    macd_sell = macd < sig

    buy_ema  = cross_up & macd_buy & (rsi >= rsi_buy)
    sell_ema = cross_dn & macd_sell & (rsi <= rsi_sell)

    out = feats.copy()
    out["Signals_EMA"] = 0
    out.loc[buy_ema,  "Signals_EMA"] =  1
    out.loc[sell_ema, "Signals_EMA"] = -1
    out["Buy_EMA"]  = np.where(out["Signals_EMA"]== 1, px, np.nan)
    out["Sell_EMA"] = np.where(out["Signals_EMA"]==-1, px, np.nan)

    widening  = (bw > bw_ma*(1+width_thresh)) & (bw_diff > 0)
    narrowing = (bw < bw_ma*(1-width_thresh)) & (bw_diff < 0)

    buy_boll  = (widening & (px > bol_u) & (rsi > 50)) | (narrowing & (px <= bol_d) & (rsi < 30))
    sell_boll = (widening & (px < bol_d) & (rsi < 50)) | (narrowing & (px >= bol_u) & (rsi > 70))

    out["Signals_BOLL"] = 0
    out.loc[buy_boll,  "Signals_BOLL"] =  1
    out.loc[sell_boll, "Signals_BOLL"] = -1
    out["Buy_BOLL"]  = np.where(out["Signals_BOLL"]== 1, px, np.nan)
    out["Sell_BOLL"] = np.where(out["Signals_BOLL"]==-1, px, np.nan)
    out["Buy_BOLL_SQUEEZE"]  = np.where(narrowing & (px <= bol_d) & (rsi < 30), px, np.nan)
    out["Sell_BOLL_SQUEEZE"] = np.where(narrowing & (px >= bol_u) & (rsi > 70), px, np.nan)

    return out

# %% --------------- Serializar señales a filas estándar ------------

def build_signal_rows(
    df_signals: pd.DataFrame,
    symbol: str, timeframe: str,
    last_n: int = 200
) -> List[Dict]:
    from datetime import datetime
    rows: List[Dict] = []
    px = df_signals["close"]
    idxs = df_signals.index[-last_n:]
    for idx in idxs:
        ts_now = datetime.utcnow().isoformat() + "Z"
        ts_bar = pd.to_datetime(idx).to_pydatetime().isoformat() + "Z"
        if df_signals.loc[idx, "Signals_EMA"] == 1:
            rows.append({"ts": ts_now,"date": ts_bar,"symbol": symbol,"timeframe": timeframe,
                         "side": "BUY","price": float(px.loc[idx]),
                         "reason": "EMA_fast>EMA_slow & MACD>signal & RSI"})
        if df_signals.loc[idx, "Signals_EMA"] == -1:
            rows.append({"ts": ts_now,"date": ts_bar,"symbol": symbol,"timeframe": timeframe,
                         "side": "SELL","price": float(px.loc[idx]),
                         "reason": "EMA_fast<EMA_slow & MACD<signal & RSI"})
        if df_signals.loc[idx, "Signals_BOLL"] == 1:
            rows.append({"ts": ts_now,"date": ts_bar,"symbol": symbol,"timeframe": timeframe,
                         "side": "BUY","price": float(px.loc[idx]),
                         "reason": "Bollinger widening/squeeze + RSI"})
        if df_signals.loc[idx, "Signals_BOLL"] == -1:
            rows.append({"ts": ts_now,"date": ts_bar,"symbol": symbol,"timeframe": timeframe,
                         "side": "SELL","price": float(px.loc[idx]),
                         "reason": "Bollinger widening/squeeze + RSI"})
    return rows

# %% ------------------------ Motor POO -----------------------------

class IndicatorSignalEngine:
    """
    - normalize_df(): índice datetime UTC-naive, columnas minúsculas.
    - compute(): indicadores
    - signals(): señales
    """
    def __init__(self,
                 ema_fast: int = 12, ema_slow: int = 26,
                 rsi_buy: float = 55.0, rsi_sell: float = 45.0,
                 rsi_period: int = 14,
                 bb_window: int = 20, bb_dev: float = 2.0,
                 width_thresh: float = 0.10):
        self.ema_fast = ema_fast; self.ema_slow = ema_slow
        self.rsi_buy = rsi_buy;   self.rsi_sell = rsi_sell
        self.rsi_period = rsi_period
        self.bb_window = bb_window; self.bb_dev = bb_dev
        self.width_thresh = width_thresh

    def normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if not isinstance(d.index, pd.DatetimeIndex):
            d.index = pd.to_datetime(d.index, errors="coerce")
        if d.index.tz is not None:
            d = d.tz_convert("UTC").tz_localize(None)
        d.columns = [str(c).lower() for c in d.columns]
        
        # --- saneamiento adicional (no cambia nombres) ---
        d = d[~d.index.duplicated(keep="last")]         # quita duplicados
        d = d.sort_index()                               # orden cronológico
        # descarta barras completamente nulas
        d = d.dropna(how="all")
        # mínimo de barras para estabilidad de indicadores
        if len(d) < max(self.ema_slow*3, 200):
            return pd.DataFrame()  # devolver vacío: strategy lo maneja silenciosa
        
        return d

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self.normalize_df(df)
        return compute_indicators(
            d, ema_fast=self.ema_fast, ema_slow=self.ema_slow,
            rsi_period=self.rsi_period, bb_window=self.bb_window, bb_dev=self.bb_dev
        )

    def signals(self, feats: pd.DataFrame) -> pd.DataFrame:
        return generate_signals(
            feats, ema_fast=self.ema_fast, ema_slow=self.ema_slow,
            rsi_buy=self.rsi_buy, rsi_sell=self.rsi_sell, width_thresh=self.width_thresh
        )
