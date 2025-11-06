from modules.data_loader import descargar_datos
from modules.data_processing import escalar_precios
from modules.data_visualization import graficar_comparacion, graficar_escalados
from modules.candles_plot import graficar_velas

def main():
    # Descarga de datos
    df_tsla = descargar_datos('TSLA', '2020-01-01', '2021-01-01')
    df_sp500 = descargar_datos('^GSPC', '2020-01-01', '2021-01-01')

    # Procesamiento
    df_tsla = escalar_precios(df_tsla)
    df_sp500 = escalar_precios(df_sp500)

    # Visualización
    graficar_comparacion(df_tsla, df_sp500, 'Tesla', 'S&P500')
    graficar_escalados(df_tsla, df_sp500, 'Tesla', 'S&P500')
    graficar_velas(df_tsla, 'Velas japonesas Tesla', 
                   hlines=dict(hlines=[500, 600], colors=['red', 'green'], linestyle='-.'))

if __name__ == "__main__":
    main()
