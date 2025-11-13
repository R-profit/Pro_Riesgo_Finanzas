# Decorarador reutilizabe pára reintentat funciones inestables con backoff exponencial y jitter.
#(reduce picos y evita bucles agresivos)

# Decorador retry con backoff exponencial + jitter
from __future__ import annotations
import time, random
from typing import Tuple, Type

def retry(
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    max_attempts: int = 3,
    base_delay: float = 0.5,   # segundos
    max_delay: float = 8.0,    # tope
    jitter: float = 0.25,      # +/- aleatorio
):
    """
    Uso:
      @retry((RuntimeError,), max_attempts=4, base_delay=0.75)
      def mi_func(...):
          ...
    Backoff: delay = min(max_delay, base_delay * 2**(attempt-1)) +/- jitter
    """
    def _wrap(fn):
        def _inner(*args, **kwargs):
            attempt = 1
            last_exc = None
            while attempt <= max_attempts:
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == max_attempts:
                        break
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    delay += random.uniform(-jitter, jitter)
                    if delay < 0:
                        delay = 0
                    time.sleep(delay)
                    attempt += 1
            raise RuntimeError(f"{fn.__name__} falló tras {max_attempts} intentos: {last_exc}") from last_exc
        return _inner
    return _wrap

