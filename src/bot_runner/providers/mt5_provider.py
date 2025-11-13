#--------------------------------
# MT5Provider: Proveedor de datos usando MetaTrader 5
#-----------------------------------------------------

#%%
# adaptador ;devuelve dataframe UTC con las columnas esperadas

from __future__ import annotations
from typing import Optional
from datetime import datetime, timedelta

from bot_runner.providers.mt5_client import MT5Client, TF  # ya lo tienes

class MT5Provider:
    def __init__(self, mt5: MT5Client):
        self.mt5 = mt5

    def get_df(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        count: Optional[int] = None,
    ):
        tf: TF = timeframe  # type narrowing p/ mypy
        if start and end:
            rates_df = self.mt5.copy_rates_range_df(symbol, tf, start, end)
        else:
            # Si no dan fechas, trae 'count' últimas velas (por defecto 2k)
            if count is None:
                count = 2000
            rates_df = self.mt5.copy_rates_from_pos_df(symbol, tf, 0, count)
        return rates_df

# %%