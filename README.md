# Predicción de Incendios Forestales en Guayaquil con MLOps

Trabajo de titulación — Ingeniería en Ciencias de la Computación  
Universidad Politécnica Salesiana, Sede Guayaquil  
Autor: Antonio Eliceo Ochoa Padilla  
Tutor: Guillermo Pizarro, Ing. Msc.

---

## ¿De qué trata este proyecto?

Guayaquil tiene varios bosques urbanos que cada año enfrentan temporadas de incendios: 
Cerro Blanco, Cerro Paraíso, Manglares del Salado, Papagayo y Prosperina. 
Este proyecto construye dos prototipos para predecir focos de calor usando datos 
reales de NASA FIRMS y variables climáticas de ERA5, y compara qué tan diferente 
es trabajar con y sin prácticas MLOps.

La idea no es ver cuál modelo predice mejor — ambos usan el mismo LSTM. 
Lo que se compara es qué pasa cuando uno tiene infraestructura MLOps completa 
y el otro no tiene nada de eso.

---

## Datos utilizados

- **NASA FIRMS VIIRS S-NPP** — focos de calor 2023 y 2024
  - Fuente: https://firms.modaps.eosdis.nasa.gov/country/
  - Filtro: coordenadas de Guayaquil (lat -2.35 a -1.85, lon -80.10 a -79.75)
  - Total: 998 registros

- **Open-Meteo ERA5** — variables climáticas diarias 2023-2024
  - Temperatura máxima, precipitación, velocidad del viento, humedad relativa
  - 5 zonas: Cerro Blanco, Cerro Paraíso, Manglares del Salado, Papagayo, Prosperina

---

## Estructura del proyecto
├── scripts/
│   ├── cargar_firms.py         # Descarga y filtra datos NASA FIRMS
│   └── era5_openmeteo.py       # Descarga variables climáticas ERA5
├── brazo_a_mlops.py            # LSTM con MLflow + Evidently (Brazo A)
├── brazo_b_solo_genai.py       # LSTM sin MLOps (Brazo B)
├── comparacion_final.py        # Gráficos y tabla comparativa
├── brazo_a_mlops/              # Modelo, métricas y reporte drift del Brazo A
├── brazo_b_solo_genai/         # Modelo y métricas del Brazo B
├── datos/                      # CSVs generados por los scripts
├── outputs/                    # Gráficos y tabla comparativa final
└── venv_incendios/             # Entorno virtual Python (no versionar)

## Cómo correr el proyecto

### 1. Requisitos
- Python 3.11
- Git

### 2. Clonar y preparar el entorno

    git clone https://github.com/tu_usuario/Antonio_Ochoa.git
    cd Antonio_Ochoa
    python -m venv venv_incendios
    venv_incendios\Scripts\activate
    pip install mlflow dvc fastapi uvicorn torch numpy pandas scikit-learn xgboost requests matplotlib seaborn evidently python-multipart ujson==5.8.0

### 3. Descargar datos

    python scripts\cargar_firms.py
    python scripts\era5_openmeteo.py

### 4. Iniciar MLflow

Abrir una ventana de cmd dedicada:

    venv_incendios\Scripts\activate
    mlflow server --host 127.0.0.1 --port 5000

Verificar en: http://127.0.0.1:5000

### 5. Ejecutar los dos brazos

    python brazo_b_solo_genai.py
    python brazo_a_mlops.py

### 6. Generar comparación final

    python comparacion_final.py

---

## Resultados

| Métrica | Brazo A (Con MLOps) | Brazo B (Sin MLOps) |
|---|---|---|
| MAE | 3.92 | 3.92 |
| RMSE | 6.45 | 6.45 |
| R² | -0.003 | -0.003 |
| Tiempo entrenamiento | 7.7s | 9.8s |
| Reproducible | SÍ | NO |
| Version