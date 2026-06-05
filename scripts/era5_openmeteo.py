"""
era5_openmeteo.py
Descarga variables climáticas históricas desde Open-Meteo (ERA5)
Enfoque: Guayaquil y sus bosques urbanos
Sin cuenta, sin API key
"""

import os
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime

# ─── CONFIGURACIÓN ───────────────────────────────────────────────────────────
CARPETA_DATOS  = os.path.join(os.path.dirname(__file__), '..', 'datos')
CARPETA_SALIDA = os.path.join(os.path.dirname(__file__), '..', 'outputs')

URL_OPENMETEO = "https://archive-api.open-meteo.com/v1/archive"

VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "relative_humidity_2m_mean",
    "et0_fao_evapotranspiration"
]

# Bosques urbanos de Guayaquil
ZONAS = [
    {"nombre": "Cerro_Blanco",    "lat": -2.1800, "lon": -80.0200},
    {"nombre": "Cerro_Paraiso",   "lat": -2.1200, "lon": -79.9500},
    {"nombre": "Manglares_Salado","lat": -2.1700, "lon": -79.9300},
    {"nombre": "Papagayo",        "lat": -2.2000, "lon": -79.9800},
    {"nombre": "Prosperina",      "lat": -2.1400, "lon": -79.9700},
]

FECHA_INICIO = "2023-01-01"
FECHA_FIN    = "2024-12-31"

os.makedirs(CARPETA_DATOS,  exist_ok=True)
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# ─── FUNCIONES ────────────────────────────────────────────────────────────────
def descargar_clima(zona):
    nombre = zona['nombre']
    print(f"  Descargando clima: {nombre}...")
    params = {
        "latitude":   zona['lat'],
        "longitude":  zona['lon'],
        "start_date": FECHA_INICIO,
        "end_date":   FECHA_FIN,
        "daily":      ",".join(VARIABLES),
        "timezone":   "America/Guayaquil"
    }
    try:
        resp = requests.get(URL_OPENMETEO, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data['daily'])
        df['zona']      = nombre
        df['latitude']  = zona['lat']
        df['longitude'] = zona['lon']
        df.rename(columns={'time': 'fecha'}, inplace=True)
        print(f"  [OK] {nombre}: {len(df)} días de datos")
        return df
    except Exception as e:
        print(f"  [ERROR] {nombre}: {e}")
        return None


def generar_grafico_temperatura(df_total):
    print("\n  Generando gráfico de temperatura máxima...")
    df_total['fecha'] = pd.to_datetime(df_total['fecha'])
    df_total['mes']   = df_total['fecha'].dt.to_period('M')
    temp_mes = df_total.groupby(['mes', 'zona'])['temperature_2m_max'].mean().reset_index()
    temp_mes['mes_str'] = temp_mes['mes'].astype(str)

    fig, ax = plt.subplots(figsize=(14, 6))
    for zona in temp_mes['zona'].unique():
        subset = temp_mes[temp_mes['zona'] == zona]
        ax.plot(subset['mes_str'], subset['temperature_2m_max'],
                marker='o', linewidth=1.5, markersize=4,
                label=zona.replace('_', ' '))

    ax.set_title('Temperatura Máxima Mensual — Bosques de Guayaquil\n(Open-Meteo ERA5 · 2023–2024)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Mes', fontsize=11)
    ax.set_ylabel('Temperatura Máxima (°C)', fontsize=11)
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()

    ruta = os.path.join(CARPETA_SALIDA, 'temperatura_maxima_guayaquil.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Gráfico guardado: {ruta}")


def generar_grafico_precipitacion(df_total):
    print("  Generando gráfico de precipitación...")
    df_total['fecha'] = pd.to_datetime(df_total['fecha'])
    df_total['mes']   = df_total['fecha'].dt.to_period('M')
    prec_mes = df_total.groupby(['mes', 'zona'])['precipitation_sum'].sum().reset_index()
    prec_mes['mes_str'] = prec_mes['mes'].astype(str)

    zonas = prec_mes['zona'].unique()
    meses = prec_mes['mes_str'].unique()
    x     = range(len(meses))
    ancho = 0.15

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, zona in enumerate(zonas):
        subset = prec_mes[prec_mes['zona'] == zona]
        vals   = [subset[subset['mes_str'] == m]['precipitation_sum'].values[0]
                  if m in subset['mes_str'].values else 0 for m in meses]
        offset = (i - len(zonas)/2) * ancho
        ax.bar([xi + offset for xi in x], vals, ancho,
               label=zona.replace('_', ' '), alpha=0.85)

    ax.set_title('Precipitación Mensual Acumulada — Bosques de Guayaquil\n(Open-Meteo ERA5 · 2023–2024)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Mes', fontsize=11)
    ax.set_ylabel('Precipitación Acumulada (mm)', fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(meses, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    plt.tight_layout()

    ruta = os.path.join(CARPETA_SALIDA, 'precipitacion_guayaquil.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Gráfico guardado: {ruta}")


def resumen(df_total):
    print("\n" + "="*55)
    print("  RESUMEN — VARIABLES CLIMÁTICAS GUAYAQUIL")
    print("="*55)
    print(f"  Total de registros        : {len(df_total):,}")
    print(f"  Zonas cubiertas           : {df_total['zona'].nunique()}")
    print(f"  Período                   : {df_total['fecha'].min()} → {df_total['fecha'].max()}")
    print(f"  Temp. máx. promedio       : {df_total['temperature_2m_max'].mean():.1f} °C")
    print(f"  Precipitación total       : {df_total['precipitation_sum'].sum():.0f} mm")
    print(f"  Humedad relativa prom.    : {df_total['relative_humidity_2m_mean'].mean():.1f} %")
    print("="*55)


# ─── EJECUCIÓN PRINCIPAL ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  DESCARGA CLIMA ERA5 — BOSQUES DE GUAYAQUIL")
    print(f"  Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*55)

    frames = []
    for zona in ZONAS:
        df = descargar_clima(zona)
        if df is not None:
            frames.append(df)
        time.sleep(1)

    if frames:
        df_total = pd.concat(frames, ignore_index=True)
        csv_path = os.path.join(CARPETA_DATOS, 'era5_guayaquil_clima.csv')
        df_total.to_csv(csv_path, index=False)
        print(f"\n  [OK] Dataset climático guardado: {csv_path}")
        resumen(df_total)
        generar_grafico_temperatura(df_total)
        generar_grafico_precipitacion(df_total)
        print("\n  Variables climáticas de Guayaquil listas.")
    else:
        print("\n  [ERROR] No se pudieron obtener datos climáticos.")