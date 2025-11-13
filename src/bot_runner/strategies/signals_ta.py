# %% ============================================
# Strategy: Señales EMA/MACD + Bollinger/RSI (MT5)
# ============================================
#la estrategia normaliza en defensivo, pide HTF si se configura, filtra por MTF,
#arma filas con SL/TP por ATR, exporta CSV liviano para auditoría,mantiene tus 
#nombres en dataframe de señales
# ============================================
# Strategy: Signals TA (provider-agnóstico + MTF)
# ============================================
from __future__ import annotations
import os, json
import pandas as pd
from typing import Dict, Any, List, Optional
from pro_riesgo_finanzas.indicadores_señales import IndicatorSignalEngine
from pro_riesgo_finanzas.viz.candles import plot_candles, plot_ema_rsi

# df: tiene Close/Adj Close (y Open/High/Low/Volume si quieres velas)
png1 = plot_candles(df, symbol)
png2 = plot_ema_rsi(df, symbol)
# El runner ya recoge PNGs si tu estrategia los retorna o si los guardas en outputs/


class SignalsTaStrategy:
    """
    Espera en params:
      symbol, timeframe, lookback
      ema_fast, ema_slow, rsi_window, bb_window, bb_dev, width_thresh,
      cooldown_bars, atr_min_pct, atr_max_pct, adx_min,
      htf_timeframe (opcional), htf_ema (opcional)
    El runner inyecta self.provider con get_df(symbol,timeframe,count|start/end)
    """

    def __init__(self, provider, params: Dict[str, Any], logger=None):
        self.provider = provider
        self.params = params or {}
        self.logger = logger

        # Básicos
        self.symbol    = self.params.get("symbol")
        self.timeframe = self.params.get("timeframe", "M15")
        self.lookback  = int(self.params.get("lookback", 1500))

        # Engine params
        self.engine = IndicatorSignalEngine(
            ema_fast     = int(self.params.get("ema_fast", 12)),
            ema_slow     = int(self.params.get("ema_slow", 26)),
            rsi_window   = int(self.params.get("rsi_window", 14)),
            bb_window    = int(self.params.get("bb_window", 20)),
            bb_dev       = float(self.params.get("bb_dev", 2.0)),
            width_thresh = float(self.params.get("width_thresh", 0.10)),
            cooldown_bars= int(self.params.get("cooldown_bars", 5)),
            atr_min_pct  = float(self.params.get("atr_min_pct", 0.001)),
            atr_max_pct  = float(self.params.get("atr_max_pct", 0.03)),
            adx_min      = float(self.params.get("adx_min", 20.0)),
        )

        # MTF opcional
        self.htf_timeframe = self.params.get("htf_timeframe")
        self.htf_ema = int(self.params.get("htf_ema", 50))

    # ---------- helpers ----------
    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        d = df.copy()
        if not isinstance(d.index, pd.DatetimeIndex):
            d.index = pd.to_datetime(d.index, errors="coerce")
        if d.index.tz is not None:
            d = d.tz_convert("UTC").tz_localize(None)
        d.columns = [str(c).lower() for c in d.columns]
        return d

    def _build_signal_rows(self, df_signals: pd.DataFrame) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        px = df_signals["close"]
        atr = df_signals["ATR"] if "ATR" in df_signals.columns else None
        ts = df_signals.index

        # SL/TP por ATR
        atr_mult_sl = 1.5
        atr_mult_tp = 2.5

        # EMA
        buys  = df_signals.index[df_signals["Signals_EMA"]== 1]
        sells = df_signals.index[df_signals["Signals_EMA"]==-1]
        for idx in buys:
            price = float(px.loc[idx])
            a = float(atr.loc[idx]) if atr is not None else None
            rows.append({
                "timestamp": ts[ts.get_loc(idx)],
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "source": "EMA",
                "side": "BUY",
                "price": price,
                "atr": a,
                "sl": (price - atr_mult_sl*a) if a else None,
                "tp": (price + atr_mult_tp*a) if a else None,
                "score": float(df_signals.get("score", pd.Series(index=ts)).get(idx, 0.0)),
                "reason": df_signals.get("reason_txt", pd.Series(index=ts)).get(idx, "")
            })
        for idx in sells:
            price = float(px.loc[idx])
            a = float(atr.loc[idx]) if atr is not None else None
            rows.append({
                "timestamp": ts[ts.get_loc(idx)],
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "source": "EMA",
                "side": "SELL",
                "price": price,
                "atr": a,
                "sl": (price + atr_mult_sl*a) if a else None,
                "tp": (price - atr_mult_tp*a) if a else None,
                "score": float(df_signals.get("score", pd.Series(index=ts)).get(idx, 0.0)),
                "reason": df_signals.get("reason_txt", pd.Series(index=ts)).get(idx, "")
            })

        # BOLL
        buys  = df_signals.index[df_signals["Signals_BOLL"]== 1]
        sells = df_signals.index[df_signals["Signals_BOLL"]==-1]
        for idx in buys:
            price = float(px.loc[idx]); a = float(atr.loc[idx]) if atr is not None else None
            rows.append({
                "timestamp": ts[ts.get_loc(idx)],
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "source": "BOLL",
                "side": "BUY",
                "price": price,
                "atr": a,
                "sl": (price - atr_mult_sl*a) if a else None,
                "tp": (price + atr_mult_tp*a) if a else None,
                "score": float(df_signals.get("score", pd.Series(index=ts)).get(idx, 0.0)),
                "reason": df_signals.get("reason_txt", pd.Series(index=ts)).get(idx, "")
            })
        for idx in sells:
            price = float(px.loc[idx]); a = float(atr.loc[idx]) if atr is not None else None
            rows.append({
                "timestamp": ts[ts.get_loc(idx)],
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "source": "BOLL",
                "side": "SELL",
                "price": price,
                "atr": a,
                "sl": (price + atr_mult_sl*a) if a else None,
                "tp": (price - atr_mult_tp*a) if a else None,
                "score": float(df_signals.get("score", pd.Series(index=ts)).get(idx, 0.0)),
                "reason": df_signals.get("reason_txt", pd.Series(index=ts)).get(idx, "")
            })
        return rows

    # ---------- run ----------
    def run(self) -> Dict[str, Any]:
        # 1) Data
        df = self.provider.get_df(self.symbol, self.timeframe, count=self.lookback)
        df = self._normalize_df(df)
        if df.empty:
            return {"ok": False, "error": "DataFrame vacío del provider."}

        # 2) Features
        feats = self.engine.compute(df)
        if feats.empty:
            return {"ok": False, "error": "Insuficientes barras para indicadores."}

        # 3) Señales
        sigs = self.engine.signals(feats)
        if sigs.empty:
            return {"ok": True, "signals": [], "df": feats.tail(300)}  # sin señales pero ok

        # 4) Confirmación MTF (opcional)
        if self.htf_timeframe:
            df_htf = self.provider.get_df(self.symbol, self.htf_timeframe, count=400)
            df_htf = self._normalize_df(df_htf)
            if not df_htf.empty and "close" in df_htf.columns:
                htf_ema = df_htf["close"].ewm(span=self.htf_ema, adjust=False,
                                              min_periods=self.htf_ema).mean()
                htf_ok = bool(df_htf["close"].iloc[-1] > htf_ema.iloc[-1])
                if not htf_ok:
                    sigs["Signals_EMA"] = 0
                    sigs["Signals_BOLL"] = 0

        # 5) Rows + export opcional
        rows = self._build_signal_rows(sigs)

        # Export liviano para auditoría
        out_dir = os.path.join("outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"signals_{self.symbol}_{self.timeframe}.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

        return {
            "ok": True,
            "signals": rows,
            "last_bar": feats.iloc[-1:].to_dict(orient="records")[0],
            "export": out_csv,
        }


