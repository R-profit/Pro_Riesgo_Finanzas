
# %%
def graficar_candlestick(self):
        """
        Grafica velas japonesas con mplfinance; comprueba que existan Open, High, Low, Close.
        """
        if self.data.empty:
            print("[INFO] No hay datos para graficar velas.")
            return

        required = {'Open', 'High', 'Low', 'Close'}
        if not required.issubset(set(self.data.columns)):
            print("[Qubit!] DataFrame no contiene columnas mínimas para velas (Open,High,Low,Close).")
            return

        try:
            mpf.plot(self.data.tail(120), type='candle', style='charles', # type: ignore
                     title=f'Últimos 120 días - {self.symbol}', volume=True)
        except Exception as e:
            print(f"[Qubit!] al graficar candlestick: {e}")
# %%
# %%
def graficar_cruce_ema_rsi_120(self, filename="cruce_ema_rsi_120.png", mostrar_leyenda=True):
        """
        Grafica precio, EMA_20 y EMA_50 y panel inferior con RSI para los últimos 120 días.
        Guarda la figura en 'filename' (no muestra ventana). Método seguro: valida datos y usa copia local.
        """
        # Seguridad y validez mínima
        if self.data is None or getattr(self.data, "empty", True):
            print("[INFO] No hay datos para graficar cruce EMA/RSI.")
            return

        # Copia local para no alterar self.data
        df_local = self.data.copy()

        # Selección de la serie de cierre (Adj Close preferido)
        if 'Adj Close' in df_local.columns:
            price = df_local['Adj Close'].astype(float)
        elif 'Close' in df_local.columns:
            price = df_local['Close'].astype(float)
        else:
            print("[Qubit!] No existe columna de cierre para graficar cruce EMA/RSI.")
            return

        price = price.dropna()
        if price.empty:
            print("[WARN] Serie de precios vacía para graficar.")
            return

        # Tomar últimos 120 puntos
        ventana = 120
        price = price.tail(ventana)
        if price.shape[0] < 2:
            print("[WARN] No hay suficientes puntos para graficar (menos de 2).")
            return

        # EMAs: preferir columnas precalculadas si existen y cubren la ventana, si no calcular localmente
        if 'EMA_20' in df_local.columns:
            ema20 = df_local['EMA_20'].astype(float).reindex(price.index).ffill().bfill()
        else:
            ema20 = price.ewm(span=20, adjust=False).mean()

        if 'EMA_50' in df_local.columns:
            ema50 = df_local['EMA_50'].astype(float).reindex(price.index).ffill().bfill()
        else:
            ema50 = price.ewm(span=50, adjust=False).mean()

        # RSI: preferir columna existente, sino calcular RSI(14) localmente
        if 'RSI' in df_local.columns:
            rsi = df_local['RSI'].astype(float).reindex(price.index).ffill().bfill()
        else:
            delta = price.diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            roll_up = up.ewm(span=14, adjust=False).mean()
            roll_down = down.ewm(span=14, adjust=False).mean()
            rs = roll_up / roll_down
            rsi = 100 - (100 / (1 + rs))

        # Detectar eventos de cruce EMA20/EMA50 (alcista/bajista)
        cross = (ema20 > ema50).astype(int)
        cross_shift = cross.shift(1).fillna(cross.iloc[0])
        cross_events = cross - cross_shift
        crosses_up = cross_events[cross_events == 1].index
        crosses_down = cross_events[cross_events == -1].index

        # Graficar en dos paneles usando estilo similar al resto del módulo
        try:
            import matplotlib.dates as mdates # type: ignore
            fig, (ax_price, ax_rsi) = plt.subplots(
                2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True
            )

            # Panel precio + EMAs
            ax_price.plot(price.index, price.values, label='Precio', color='#222222', linewidth=1)
            ax_price.plot(ema20.index, ema20.values, label='EMA 20', color='#1f77b4', linewidth=1.5)
            ax_price.plot(ema50.index, ema50.values, label='EMA 50', color='#ff7f0e', linewidth=1.5)

            # Líneas verticales para cruces (mantener transparencia)
            for xi in crosses_up:
                if xi in price.index:
                    ax_price.axvline(x=xi, color='green', linestyle='--', alpha=0.6, linewidth=0.8)
            for xi in crosses_down:
                if xi in price.index:
                    ax_price.axvline(x=xi, color='red', linestyle='--', alpha=0.6, linewidth=0.8)

            if mostrar_leyenda:
                ax_price.legend(loc='upper left', fontsize='small')

            ax_price.set_title(f'Últimos {ventana} días - Precio y Cruce EMA(20/50) - {self.symbol}')
            ax_price.set_ylabel('Precio')

            # Panel RSI
            ax_rsi.plot(rsi.index, rsi.values, color='#2ca02c', linewidth=1)
            ax_rsi.axhline(70, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
            ax_rsi.axhline(30, color='blue', linestyle='--', linewidth=0.8, alpha=0.6)
            ax_rsi.set_ylabel('RSI')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.set_xlabel('Fecha')

            # Formato de fechas coherente
            ax_rsi.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax_rsi.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_rsi.xaxis.get_major_locator()))

            plt.tight_layout() # type: ignore
            # Guardar figura en disco para uso por bot (no usar plt.show() en entornos server)
            plt.savefig(filename, bbox_inches='tight') # type: ignore
            plt.close(fig) # type: ignore
            print(f"[INFO] Gráfica guardada en {filename}")

        except Exception as e:
            print(f"[Qubit!] Error al graficar cruce EMA/RSI: {e}")
            try:
                plt.close('all') # type: ignore
            except Exception:
                pass
            return
# %%        

