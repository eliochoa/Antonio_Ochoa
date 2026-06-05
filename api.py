"""
api.py
API REST para inferencia del modelo LSTM — Brazo A (MLOps)
Endpoint: POST /predecir
Autor: Antonio Eliceo Ochoa Padilla
Universidad Politécnica Salesiana — Sede Guayaquil
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import mlflow
import uvicorn
from datetime import datetime

# ─── CONFIGURACIÓN ───────────────────────────────────────────
CARPETA_MODELO = os.path.join(os.path.dirname(__file__), 'brazo_a_mlops')
MODELO_PATH    = os.path.join(CARPETA_MODELO, 'lstm_brazo_a.pth')

HIDDEN_SIZE = 64
NUM_LAYERS  = 2
INPUT_SIZE  = 5
SECUENCIA   = 30

# ─── MODELO LSTM ─────────────────────────────────────────────
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ─── CARGAR MODELO ───────────────────────────────────────────
print("Cargando modelo LSTM Brazo A...")
checkpoint = torch.load(MODELO_PATH, map_location='cpu', weights_only=False)
modelo = LSTMPredictor(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
modelo.load_state_dict(checkpoint['model_state'])
modelo.eval()
print(f"[OK] Modelo cargado — Run ID: {checkpoint.get('run_id', 'N/A')}")

# ─── FASTAPI APP ──────────────────────────────────────────────
app = FastAPI(
    title="API Predicción Incendios Forestales — Guayaquil",
    description="Predice focos de calor en bosques urbanos de Guayaquil usando LSTM con MLOps",
    version="1.0.0"
)

# ─── SCHEMAS ──────────────────────────────────────────────────
class DatosClimaticos(BaseModel):
    temperatura_max: float
    precipitacion:   float
    viento_max:      float
    humedad_media:   float
    focos_previos:   float

class SolicitudPrediccion(BaseModel):
    secuencia: List[DatosClimaticos]
    zona: str = "Guayaquil-bosques"

class RespuestaPrediccion(BaseModel):
    zona:             str
    focos_predichos:  float
    nivel_riesgo:     str
    latencia_ms:      float
    modelo_run_id:    str
    timestamp:        str

# ─── ENDPOINTS ────────────────────────────────────────────────
@app.get("/")
def raiz():
    return {
        "servicio": "API Predicción Incendios Forestales — Guayaquil",
        "version": "1.0.0",
        "modelo": "LSTM Brazo A (MLOps)",
        "run_id": checkpoint.get('run_id', 'N/A'),
        "estado": "activo",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/salud")
def salud():
    return {"estado": "OK", "modelo_cargado": True}

@app.post("/predecir", response_model=RespuestaPrediccion)
def predecir(solicitud: SolicitudPrediccion):

    if len(solicitud.secuencia) != SECUENCIA:
        raise HTTPException(
            status_code=400,
            detail=f"Se requieren exactamente {SECUENCIA} días de datos. Recibidos: {len(solicitud.secuencia)}"
        )

    # Convertir a tensor
    datos = np.array([[
        d.temperatura_max,
        d.precipitacion,
        d.viento_max,
        d.humedad_media,
        d.focos_previos
    ] for d in solicitud.secuencia], dtype=np.float32)

    # Normalización simple (0-1)
    datos_norm = (datos - datos.min(axis=0)) / (datos.max(axis=0) - datos.min(axis=0) + 1e-8)
    tensor = torch.FloatTensor(datos_norm).unsqueeze(0)

    # Inferencia
    t_inicio = time.time()
    with torch.no_grad():
        pred = modelo(tensor).item()
    latencia_ms = (time.time() - t_inicio) * 1000

    # Desnormalizar aproximado
    focos_max = datos[:, 4].max()
    focos_predichos = max(0.0, round(pred * focos_max * 10, 2))

    # Nivel de riesgo
    if focos_predichos >= 50:
        nivel = "ALTO"
    elif focos_predichos >= 20:
        nivel = "MEDIO"
    else:
        nivel = "BAJO"

    return RespuestaPrediccion(
        zona            = solicitud.zona,
        focos_predichos = focos_predichos,
        nivel_riesgo    = nivel,
        latencia_ms     = round(latencia_ms, 2),
        modelo_run_id   = checkpoint.get('run_id', 'N/A'),
        timestamp       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.get("/modelo/info")
def info_modelo():
    return {
        "arquitectura":  "LSTM",
        "hidden_size":   HIDDEN_SIZE,
        "num_layers":    NUM_LAYERS,
        "secuencia_dias": SECUENCIA,
        "variables":     ["temperatura_max", "precipitacion", "viento_max", "humedad_media", "focos_previos"],
        "run_id":        checkpoint.get('run_id', 'N/A'),
        "metricas":      checkpoint.get('metricas', {}),
        "zona":          "Bosques urbanos de Guayaquil",
        "sensor_datos":  "NASA FIRMS VIIRS S-NPP + Open-Meteo ERA5"
    }

# ─── EJECUCIÓN ────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  API PREDICCIÓN INCENDIOS — GUAYAQUIL")
    print("  Brazo A — LSTM con MLOps")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*55)
    print("  Endpoints disponibles:")
    print("    GET  http://127.0.0.1:8000/")
    print("    GET  http://127.0.0.1:8000/salud")
    print("    GET  http://127.0.0.1:8000/modelo/info")
    print("    POST http://127.0.0.1:8000/predecir")
    print("    GET  http://127.0.0.1:8000/docs")
    print("="*55)
    uvicorn.run(app, host="127.0.0.1", port=8000)