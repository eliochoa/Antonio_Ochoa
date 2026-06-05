"""
cargar_firms.py
Descarga datos de incendios forestales desde NASA FIRMS
Enfoque: Guayaquil y sus bosques urbanos
(Cerro Blanco, Cerro Paraíso, Manglares del Salado, Papagayo, Prosperina)
"""

import os
import zipfile
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime

# ─── CONFIGURACIÓN ───────────────────────────────────────────────────────────
CARPETA_DATOS  = os.path.join(os.path.dirname(__file__), '..', 'datos')
CARPETA_SALIDA = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# Coordenadas Guayaquil y bosques urbanos
LAT_MIN, LAT_MAX = -2.35, -1.85
LON_MIN, LON_MAX = -80.10, -79.75

ANIOS    = [2023, 2024]
URL_BASE = "https://firms.modaps.eosdis.nasa.gov/data/country/zips/viirs-snpp_{anio}_all_countries.zip"

os.makedirs(CARPETA_DATOS,  exist_ok=True)
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# ─── FUNCIONES ────────────────────────────────────────────────────────────────
def descargar_firms(anio):
    url      = URL_BASE.format(anio=anio)
    zip_path = os.path.join(CARPETA_DATOS, f'viirs-snpp_{anio}.zip')
    if os.path.exists(zip_path):
        print(f"  [OK] ZIP {anio} ya existe, se omite descarga.")
        return zip_path
    print(f"  Descargando FIRMS {anio} (~385 MB)...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    descargado = 0
    with open(zip_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024*1024):
            f.write(chunk)
            descargado += len(chunk)
            if total:
                print(f"\r    {descargado/total*100:.0f}%", end='', flush=True)
    print()
    print(f"  [OK] Descarga completa: {zip_path}")
    return zip_path


def filtrar_guayaquil(zip_path, anio):
    print(f"  Filtrando Guayaquil del ZIP {anio}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_files = [n for n in z.namelist() if n.endswith('.csv')]
        if not csv_files:
            print("  [ERROR] No se encontraron CSV en el ZIP.")
            return None
        frames = []
        for csv_name in csv_files:
            with z.open(csv_name) as f:
                df = pd.read_csv(f, low_memory=False)
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    df_gye = df[
                        (df['latitude']  >= LAT_MIN) & (df['latitude']  <= LAT_MAX) &
                        (df['longitude'] >= LON_MIN) & (df['longitude'] <= LON_MAX)
                    ].copy()
                    frames.append(df_gye)
    if not frames:
        print("  [ERROR] Sin datos para Guayaquil.")
        return None
    resultado = pd.concat(frames, ignore_index=True)
    salida = os.path.join(CARPETA_DATOS, f'firms_guayaquil_{anio}.csv')
    resultado.to_csv(salida, index=False)
    print(f"  [OK] Guayaquil {anio}: {len(resultado):,} registros → {salida}")
    return resultado


def generar_grafico(df_total):
    print("\n  Generando gráfico de focos de calor por mes...")
    df_total['acq_date'] = pd.to_datetime(df_total['acq_date'], errors='coerce')
    df_total['anio_mes'] = df_total['acq_date'].dt.to_period('M')
    conteo = df_total.groupby('anio_mes').size().reset_index(name='focos')
    conteo['mes_str'] = conteo['anio_mes'].astype(str)

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(conteo['mes_str'], conteo['focos'],
                  color='#E74C3C', edgecolor='#C0392B', linewidth=0.5)

    total = conteo['focos'].sum()
    for bar, val in zip(bars, conteo['focos']):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.002,
                f'{pct:.0f}%',
                ha='center', va='bottom', fontsize=8, color='#333333')

    ax.set_title('Focos de Calor en Guayaquil por Mes\n(NASA FIRMS — VIIRS S-NPP · Cerro Blanco, Cerro Paraíso, Manglares del Salado)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Mes', fontsize=11)
    ax.set_ylabel('Número de Focos de Calor', fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()

    ruta = os.path.join(CARPETA_SALIDA, 'focos_calor_guayaquil_mensual.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Gráfico guardado: {ruta}")


def resumen(df_total):
    print("\n" + "="*55)
    print("  RESUMEN — FOCOS DE CALOR GUAYAQUIL")
    print("="*55)
    print(f"  Total de registros        : {len(df_total):,}")
    print(f"  Período                   : {df_total['acq_date'].min()} → {df_total['acq_date'].max()}")
    if 'acq_date' in df_total.columns:
        df_total['mes'] = pd.to_datetime(df_total['acq_date'], errors='coerce').dt.month
        mes_pico = df_total['mes'].value_counts().idxmax()
        meses = ['Ene','Feb','Mar','Abr','May','Jun',
                 'Jul','Ago','Sep','Oct','Nov','Dic']
        print(f"  Mes con más focos         : {meses[mes_pico-1]}")
    print("="*55)


# ─── EJECUCIÓN PRINCIPAL ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  CARGA DE DATOS NASA FIRMS — GUAYAQUIL")
    print(f"  Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*55)

    todos = []
    for anio in ANIOS:
        print(f"\n[Año {anio}]")
        zip_path = descargar_firms(anio)
        df = filtrar_guayaquil(zip_path, anio)
        if df is not None:
            todos.append(df)

    if todos:
        df_total = pd.concat(todos, ignore_index=True)
        csv_total = os.path.join(CARPETA_DATOS, 'firms_guayaquil_total.csv')
        df_total.to_csv(csv_total, index=False)
        print(f"\n  [OK] Dataset consolidado: {csv_total}")
        resumen(df_total)
        generar_grafico(df_total)
        print("\n  Datos de Guayaquil listos.")
    else:
        print("\n  [ERROR] No se pudieron obtener datos.")