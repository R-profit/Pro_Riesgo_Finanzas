from __future__ import annotations
from typing import Protocol, Optional
from datetime import datetime

class DataProvider(Protocol):
    def get_df(
        self,
        symbol: str,
        timeframe: str,                      # "M1","M5","M15","M30","H1","H4","D1"
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        count: Optional[int] = None,         # si no das rango de fechas
    ):
        """
        Debe devolver un pandas.DataFrame con índice 'time' (tz=UTC) y columnas:
        ['open','high','low','close','tick_volume','spread','real_volume'] (las que existan).
        """
        ...
