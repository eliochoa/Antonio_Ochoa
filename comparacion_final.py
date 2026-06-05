"""
comparacion_final.py
Genera tabla comparativa y gráficos finales
Brazo A (MLOps) vs Brazo B (Sin MLOps)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime

# ─── CONFIGURACIÓN ───────────────────────────────────────────────────────────
CARPETA_A      = os.path.join(os.path.dirname(__file__), 'brazo_a_mlops')
CARPETA_B      = os.path.join(os.path.dirname(__file__), 'brazo_b_solo_genai')
CARPETA_SALIDA = os.path.join(os.path.dirname(__file__), 'outputs')

os.makedirs(CARPETA_SALIDA, exist_ok=True)

# ─── CARGAR MÉTRICAS ─────────────────────────────────────────────────────────
def cargar_metricas():
    a = pd.read_csv(os.path.join(CARPETA_A, 'metricas_brazo_a.csv')).iloc[0]
    b = pd.read_csv(os.path.join(CARPETA_B, 'metricas_brazo_b.csv')).iloc[0]
    return a, b


# ─── GRÁFICO COMPARATIVO DE MÉTRICAS ─────────────────────────────────────────
def grafico_metricas(a, b):
    print("  Generando gráfico comparativo de métricas...")

    categorias = ['MAE', 'RMSE']
    vals_a = [a['MAE'], a['RMSE']]
    vals_b = [b['MAE'], b['RMSE']]

    x     = np.arange(len(categorias))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_a = ax.bar(x - ancho/2, vals_a, ancho,
                    label='Brazo A (Con MLOps)', color='#27AE60', edgecolor='#1E8449')
    bars_b = ax.bar(x + ancho/2, vals_b, ancho,
                    label='Brazo B (Sin MLOps)', color='#E74C3C', edgecolor='#C0392B')

    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.05,
                f'{bar.get_height():.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.05,
                f'{bar.get_height():.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title('Comparación de Métricas Predictivas\nBrazo A (MLOps) vs Brazo B (Sin MLOps)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Métrica', fontsize=12)
    ax.set_ylabel('Error (focos de calor)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categorias, fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()

    ruta = os.path.join(CARPETA_SALIDA, 'comparacion_metricas.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {ruta}")


# ─── GRÁFICO COMPARATIVO OPERATIVO ───────────────────────────────────────────
def grafico_operativo(a, b):
    print("  Generando gráfico operativo...")

    categorias  = ['Tiempo\nEntrenamiento (s)', 'Tiempo\nTotal (s)']
    vals_a = [a['t_entrenamiento'], a['t_total']]
    vals_b = [b['t_entrenamiento'], b['t_total']]

    x     = np.arange(len(categorias))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_a = ax.bar(x - ancho/2, vals_a, ancho,
                    label='Brazo A (Con MLOps)', color='#27AE60', edgecolor='#1E8449')
    bars_b = ax.bar(x + ancho/2, vals_b, ancho,
                    label='Brazo B (Sin MLOps)', color='#E74C3C', edgecolor='#C0392B')

    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f'{bar.get_height():.1f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f'{bar.get_height():.1f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title('Comparación de Tiempos Operativos\nBrazo A (MLOps) vs Brazo B (Sin MLOps)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Componente', fontsize=12)
    ax.set_ylabel('Tiempo (segundos)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categorias, fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()

    ruta = os.path.join(CARPETA_SALIDA, 'comparacion_operativa.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {ruta}")


# ─── GRÁFICO CAPACIDADES ─────────────────────────────────────────────────────
def grafico_capacidades():
    print("  Generando gráfico de capacidades MLOps...")

    capacidades = [
        'Versionado\nde datos',
        'Registro de\nexperimentos',
        'Monitoreo\nde drift',
        'API de\ninferencia',
        'Reproducible',
        'CI/CD',
    ]

    brazo_a = [1, 1, 1, 1, 1, 1]
    brazo_b = [0, 0, 0, 0, 0, 0]

    x     = np.arange(len(capacidades))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - ancho/2, brazo_a, ancho,
           label='Brazo A (Con MLOps)', color='#27AE60', edgecolor='#1E8449')
    ax.bar(x + ancho/2, brazo_b, ancho,
           label='Brazo B (Sin MLOps)', color='#E74C3C', edgecolor='#C0392B')

    for xi, va, vb in zip(x, brazo_a, brazo_b):
        ax.text(xi - ancho/2, va + 0.02, '100%',
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1E8449')
        ax.text(xi + ancho/2, vb + 0.02, '0%',
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#C0392B')

    ax.set_title('Capacidades Operativas\nBrazo A (MLOps) vs Brazo B (Sin MLOps)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Disponible', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(capacidades, fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Sí'], fontsize=11)
    ax.legend(fontsize=11)
    plt.tight_layout()

    ruta = os.path.join(CARPETA_SALIDA, 'comparacion_capacidades.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {ruta}")


# ─── TABLA COMPARATIVA CSV ────────────────────────────────────────────────────
def generar_tabla(a, b):
    tabla = pd.DataFrame({
        'Métrica / Capacidad': [
            'MAE', 'RMSE', 'R²',
            'Tiempo entrenamiento (s)', 'Tiempo total (s)',
            'Reproducible',
            'Versionado de datos',
            'Registro de experimentos',
            'Monitoreo de drift',
            'API de inferencia',
        ],
        'Brazo A (Con MLOps)': [
            a['MAE'], a['RMSE'], a['R2'],
            a['t_entrenamiento'], a['t_total'],
            'SÍ', 'SÍ', 'SÍ', 'SÍ', 'SÍ',
        ],
        'Brazo B (Sin MLOps)': [
            b['MAE'], b['RMSE'], b['R2'],
            b['t_entrenamiento'], b['t_total'],
            'NO', 'NO', 'NO', 'NO', 'NO',
        ],
    })

    ruta = os.path.join(CARPETA_SALIDA, 'comparacion_brazos.csv')
    tabla.to_csv(ruta, index=False)
    print(f"  [OK] Tabla guardada: {ruta}")
    return tabla


# ─── RESUMEN FINAL ────────────────────────────────────────────────────────────
def resumen_final(a, b, tabla):
    print("\n" + "="*60)
    print("  COMPARACIÓN FINAL — BRAZO A vs BRAZO B")
    print("="*60)
    print(f"\n  {'Métrica':<30} {'Brazo A':>12} {'Brazo B':>12}")
    print(f"  {'-'*54}")
    for _, row in tabla.iterrows():
        print(f"  {row['Métrica / Capacidad']:<30} {str(row['Brazo A (Con MLOps)']):>12} {str(row['Brazo B (Sin MLOps)']):>12}")
    print("="*60)
    print("\n  CONCLUSIÓN:")
    print("  Las métricas predictivas son equivalentes en ambos brazos.")
    print("  El valor de MLOps está en la capacidad operativa:")
    print("  reproducibilidad, monitoreo, versionado y despliegue.")
    print("="*60)


# ─── EJECUCIÓN PRINCIPAL ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  COMPARACIÓN FINAL — PROYECTO INCENDIOS GUAYAQUIL")
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

    a, b = cargar_metricas()

    grafico_metricas(a, b)
    grafico_operativo(a, b)
    grafico_capacidades()
    tabla = generar_tabla(a, b)
    resumen_final(a, b, tabla)

    print("\n  Proyecto completado.")