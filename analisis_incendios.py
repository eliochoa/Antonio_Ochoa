"""
analisis_incidentes.py
Análisis cualitativo de incidentes — Brazo A vs Brazo B
Autor: Antonio Eliceo Ochoa Padilla
Universidad Politécnica Salesiana — Sede Guayaquil
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

CARPETA_SALIDA = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(CARPETA_SALIDA, exist_ok=True)

INCIDENTES = [
    {
        "id": 1,
        "nombre": "Drift de datos climáticos",
        "descripcion": "Las variables ERA5 cambian de distribución por fenómeno El Niño.",
        "brazo_a": {
            "deteccion": "Automática — Evidently detecta drift en < 1 hora",
            "tiempo_respuesta": "< 1 hora",
            "accion": "Alerta automática al equipo. Reentrenamiento programado con DVC.",
            "recuperacion": "Automática",
            "registro": "Registrado en MLflow con Run ID y métricas de drift"
        },
        "brazo_b": {
            "deteccion": "Manual — requiere inspección del investigador",
            "tiempo_respuesta": "Días o semanas",
            "accion": "Reentrenamiento manual sin registro del incidente.",
            "recuperacion": "Manual",
            "registro": "No registrado"
        }
    },
    {
        "id": 2,
        "nombre": "Fallo del servidor de inferencia",
        "descripcion": "El servicio de predicción deja de responder durante la temporada de incendios.",
        "brazo_a": {
            "deteccion": "Automática — monitoreo de latencia en MLflow detecta timeout",
            "tiempo_respuesta": "< 5 minutos",
            "accion": "Reinicio automático del contenedor Docker. API FastAPI restaurada.",
            "recuperacion": "Automática (Docker restart policy)",
            "registro": "Log de incidente en MLflow con timestamp y métricas"
        },
        "brazo_b": {
            "deteccion": "Manual — alguien nota que el script no responde",
            "tiempo_respuesta": "Horas (depende de disponibilidad del investigador)",
            "accion": "Reinicio manual del script Python.",
            "recuperacion": "Manual",
            "registro": "No registrado"
        }
    },
    {
        "id": 3,
        "nombre": "Necesidad de reproducir una predicción histórica",
        "descripcion": "La autoridad municipal necesita verificar la predicción del 15 de septiembre 2024.",
        "brazo_a": {
            "deteccion": "N/A — consulta de auditoría",
            "tiempo_respuesta": "< 1 minuto",
            "accion": "Se recupera el Run ID de MLflow. Se reproduce exactamente con los mismos datos y parámetros.",
            "recuperacion": "Reproducción exacta garantizada",
            "registro": "Run ID: 5c85b0d347884e1784d9c2b8a77e7873"
        },
        "brazo_b": {
            "deteccion": "N/A — consulta de auditoría",
            "tiempo_respuesta": "Imposible reproducir exactamente",
            "accion": "No es posible reproducir: no hay versionado de datos ni registro de parámetros exactos.",
            "recuperacion": "No reproducible",
            "registro": "No existe registro"
        }
    },
    {
        "id": 4,
        "nombre": "Actualización del modelo con nuevos datos de incendios",
        "descripcion": "Se dispone de nuevos datos de focos de calor de enero 2025.",
        "brazo_a": {
            "deteccion": "N/A — proceso planificado",
            "tiempo_respuesta": "< 30 minutos",
            "accion": "DVC versiona los nuevos datos. Pipeline de reentrenamiento se ejecuta automáticamente. Nueva versión registrada en MLflow.",
            "recuperacion": "Automática",
            "registro": "Nueva versión de datos y modelo en DVC + MLflow"
        },
        "brazo_b": {
            "deteccion": "N/A — proceso planificado",
            "tiempo_respuesta": "Horas (configuración manual)",
            "accion": "El investigador descarga los datos manualmente, reemplaza el CSV y reentrena el script.",
            "recuperacion": "Manual — versión anterior perdida",
            "registro": "No registrado. Versión anterior del modelo no recuperable."
        }
    },
    {
        "id": 5,
        "nombre": "Degradación gradual del rendimiento",
        "descripcion": "El modelo pierde precisión por cambios estacionales en los patrones de vegetación.",
        "brazo_a": {
            "deteccion": "Automática — Evidently monitorea métricas en producción continuamente",
            "tiempo_respuesta": "Detectado en el ciclo semanal de monitoreo",
            "accion": "Alerta generada automáticamente. Análisis de drift por variable. Reentrenamiento selectivo.",
            "recuperacion": "Automática con reentrenamiento programado",
            "registro": "Historial completo de degradación en MLflow"
        },
        "brazo_b": {
            "deteccion": "Manual — solo si alguien nota predicciones erróneas",
            "tiempo_respuesta": "Semanas o meses (invisible hasta que alguien lo detecta)",
            "accion": "Reentrenamiento manual sin análisis de causa raíz.",
            "recuperacion": "Manual — sin garantía de mejora",
            "registro": "No registrado"
        }
    },
]

print("\n" + "="*70)
print("  ANÁLISIS CUALITATIVO DE INCIDENTES — BRAZO A vs BRAZO B")
print("="*70)

filas = []
for inc in INCIDENTES:
    print(f"\n[Incidente {inc['id']}] {inc['nombre']}")
    print(f"  Descripción: {inc['descripcion']}")
    print(f"  Brazo A — Detección  : {inc['brazo_a']['deteccion']}")
    print(f"  Brazo A — Respuesta  : {inc['brazo_a']['tiempo_respuesta']}")
    print(f"  Brazo A — Registro   : {inc['brazo_a']['registro']}")
    print(f"  Brazo B — Detección  : {inc['brazo_b']['deteccion']}")
    print(f"  Brazo B — Respuesta  : {inc['brazo_b']['tiempo_respuesta']}")
    print(f"  Brazo B — Registro   : {inc['brazo_b']['registro']}")
    filas.append({
        'Incidente': inc['nombre'],
        'Brazo A — Detección': inc['brazo_a']['deteccion'],
        'Brazo A — Tiempo respuesta': inc['brazo_a']['tiempo_respuesta'],
        'Brazo A — Recuperación': inc['brazo_a']['recuperacion'],
        'Brazo B — Detección': inc['brazo_b']['deteccion'],
        'Brazo B — Tiempo respuesta': inc['brazo_b']['tiempo_respuesta'],
        'Brazo B — Recuperación': inc['brazo_b']['recuperacion'],
    })

print("="*70)

df = pd.DataFrame(filas)
csv_path = os.path.join(CARPETA_SALIDA, 'analisis_incidentes.csv')
df.to_csv(csv_path, index=False)
print(f"\n[OK] Tabla guardada: {csv_path}")

print("Generando gráfico de tiempo de respuesta...")

incidentes_nombres = ["Drift de\ndatos", "Fallo del\nservidor",
                      "Reproducir\npredicción", "Actualizar\nmodelo",
                      "Degradación\ngradual"]
tiempos_a = [1, 5/60, 1/60, 30, 10080]
tiempos_b_graf = [2880, 120, 50000, 240, 43200]

x = range(len(incidentes_nombres))
ancho = 0.35

fig, ax = plt.subplots(figsize=(13, 7))
bars_a = ax.bar([xi - ancho/2 for xi in x], tiempos_a, ancho,
                label='Brazo A (MLOps)', color='#27AE60', edgecolor='#1E8449')
bars_b = ax.bar([xi + ancho/2 for xi in x], tiempos_b_graf, ancho,
                label='Brazo B (Sin MLOps)', color='#E74C3C', edgecolor='#C0392B')

etiquetas_a = ['< 1 min', '< 5 min', '< 1 min', '< 30 min', '1 semana']
etiquetas_b = ['2 días', '2 horas', 'Imposible', '4 horas', '1 mes']

for bar, label in zip(bars_a, etiquetas_a):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
            label, ha='center', va='bottom', fontsize=8,
            fontweight='bold', color='#1E8449')

for bar, label in zip(bars_b, etiquetas_b):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
            label, ha='center', va='bottom', fontsize=8,
            fontweight='bold', color='#C0392B')

ax.set_yscale('log')
ax.set_title('Tiempo de Respuesta ante Incidentes\nBrazo A (MLOps) vs Brazo B (Sin MLOps)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Tipo de Incidente', fontsize=11)
ax.set_ylabel('Tiempo de respuesta (minutos, escala logarítmica)', fontsize=11)
ax.set_xticks(list(x))
ax.set_xticklabels(incidentes_nombres, fontsize=9)
ax.legend(fontsize=11)
plt.tight_layout()

ruta = os.path.join(CARPETA_SALIDA, 'analisis_incidentes_tiempo.png')
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] Gráfico guardado: {ruta}")
print(f"\n  Análisis completado: {datetime.now().strftime('%Y-%m-%d %H:%M')}")