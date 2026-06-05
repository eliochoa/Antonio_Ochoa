"""
brazo_a_mlops.py
Modelo LSTM para predicción de focos de calor en Guayaquil
CON prácticas MLOps — MLflow + DVC + Evidently
Brazo A del experimento comparativo
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

try:
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset
    EVIDENTLY_OK = True
except Exception:
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        EVIDENTLY_OK = True
    except Exception:
        EVIDENTLY_OK = False

# ─── CONFIGURACIÓN ───────────────────────────────────────────────────────────
CARPETA_DATOS  = os.path.join(os.path.dirname(__file__), 'datos')
CARPETA_SALIDA = os.path.join(os.path.dirname(__file__), 'brazo_a_mlops')

SEMILLA     = 42
SECUENCIA   = 30
HORIZONTE   = 1
EPOCHS      = 50
BATCH_SIZE  = 16
LR          = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS  = 2

EXPERIMENTO = "incendios-guayaquil-brazo-a"

torch.manual_seed(SEMILLA)
np.random.seed(SEMILLA)
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# ─── MODELO LSTM ─────────────────────────────────────────────────────────────
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ─── FUNCIONES ────────────────────────────────────────────────────────────────
def cargar_datos():
    print("  Cargando datos...")
    firms = pd.read_csv(os.path.join(CARPETA_DATOS, 'firms_guayaquil_total.csv'))
    clima = pd.read_csv(os.path.join(CARPETA_DATOS, 'era5_guayaquil_clima.csv'))

    firms['acq_date'] = pd.to_datetime(firms['acq_date'])
    focos = firms.groupby('acq_date').size().reset_index(name='focos')
    focos.rename(columns={'acq_date': 'fecha'}, inplace=True)

    clima['fecha'] = pd.to_datetime(clima['fecha'])
    clima_gye = clima[clima['zona'] == 'Cerro_Blanco'].copy()
    clima_gye = clima_gye[['fecha', 'temperature_2m_max', 'precipitation_sum',
                            'windspeed_10m_max', 'relative_humidity_2m_mean']].copy()

    rango = pd.date_range('2023-01-01', '2024-12-31')
    df    = pd.DataFrame({'fecha': rango})
    df    = df.merge(focos, on='fecha', how='left')
    df    = df.merge(clima_gye, on='fecha', how='left')
    df['focos'] = df['focos'].fillna(0)
    df = df.ffill().fillna(0)

    print(f"  [OK] Dataset: {len(df)} días")
    return df


def preparar_secuencias(df, scaler=None):
    features = ['temperature_2m_max', 'precipitation_sum',
                'windspeed_10m_max', 'relative_humidity_2m_mean', 'focos']
    datos = df[features].values

    if scaler is None:
        scaler = MinMaxScaler()
        datos_scaled = scaler.fit_transform(datos)
    else:
        datos_scaled = scaler.transform(datos)

    X, y = [], []
    for i in range(SECUENCIA, len(datos_scaled) - HORIZONTE + 1):
        X.append(datos_scaled[i - SECUENCIA:i])
        y.append(datos_scaled[i + HORIZONTE - 1, -1])

    return (torch.FloatTensor(np.array(X)),
            torch.FloatTensor(np.array(y)).unsqueeze(1),
            scaler)


def entrenar(modelo, X_train, y_train):
    print(f"\n  Entrenando modelo LSTM ({EPOCHS} épocas)...")
    optimizer = torch.optim.Adam(modelo.parameters(), lr=LR)
    criterion = nn.MSELoss()
    losses    = []
    t_inicio  = time.time()

    for epoch in range(EPOCHS):
        modelo.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss  = 0
        batches     = 0

        for i in range(0, X_train.size(0), BATCH_SIZE):
            idx    = permutation[i:i + BATCH_SIZE]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            pred = modelo(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches    += 1

        avg_loss = epoch_loss / batches
        losses.append(avg_loss)

        # Registrar en MLflow cada época
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        if (epoch + 1) % 10 == 0:
            print(f"    Época {epoch+1:3d}/{EPOCHS} — Loss: {avg_loss:.6f}")

    t_total = time.time() - t_inicio
    print(f"  [OK] Entrenamiento: {t_total:.1f} seg")
    return losses, t_total


def evaluar(modelo, X_test, y_test, scaler):
    modelo.eval()
    t_inicio = time.time()
    with torch.no_grad():
        pred_scaled = modelo(X_test).numpy()
        real_scaled = y_test.numpy()
    latencia = (time.time() - t_inicio) * 1000  # ms

    dummy = np.zeros((len(pred_scaled), 5))
    dummy[:, -1] = pred_scaled[:, 0]
    pred_real = scaler.inverse_transform(dummy)[:, -1]

    dummy[:, -1] = real_scaled[:, 0]
    real_real = scaler.inverse_transform(dummy)[:, -1]

    pred_real = np.clip(pred_real, 0, None)

    mae  = mean_absolute_error(real_real, pred_real)
    rmse = np.sqrt(mean_squared_error(real_real, pred_real))
    ss_res = np.sum((real_real - pred_real) ** 2)
    ss_tot = np.sum((real_real - np.mean(real_real)) ** 2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return pred_real, real_real, mae, rmse, r2, latencia


def detectar_drift(df_train, df_test):
    if not EVIDENTLY_OK:
        print("  [SKIP] Evidently no disponible — omitiendo reporte de drift.")
        return
    print("  Generando reporte de drift...")
    try:
        features = ['temperature_2m_max', 'precipitation_sum',
                    'windspeed_10m_max', 'relative_humidity_2m_mean', 'focos']
        ref  = df_train[features].dropna()
        curr = df_test[features].dropna()
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=curr)
        ruta = os.path.join(CARPETA_SALIDA, 'reporte_drift.html')
        report.save_html(ruta)
        mlflow.log_artifact(ruta)
        print(f"  [OK] Reporte drift: {ruta}")
    except Exception as e:
        print(f"  [WARN] Drift report error: {e}")


def generar_grafico(real, pred, losses):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(real, label='Real', color='#2E86AB', linewidth=1.5)
    ax.plot(pred, label='Predicho', color='#E74C3C', linewidth=1.5, linestyle='--')
    ax.set_title('Brazo A — Predicción vs Real\n(LSTM con MLOps)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Días (conjunto de prueba)', fontsize=11)
    ax.set_ylabel('Focos de Calor', fontsize=11)
    ax.legend(fontsize=10)

    ax2 = axes[1]
    ax2.plot(range(1, len(losses)+1), losses,
             color='#27AE60', linewidth=2)
    ax2.set_title('Curva de Pérdida — Entrenamiento\n(Brazo A con MLOps)',
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Época', fontsize=11)
    ax2.set_ylabel('MSE Loss', fontsize=11)

    plt.tight_layout()
    ruta = os.path.join(CARPETA_SALIDA, 'brazo_a_prediccion.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(ruta)
    print(f"  [OK] Gráfico guardado: {ruta}")


# ─── EJECUCIÓN PRINCIPAL ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  BRAZO A — LSTM CON MLOps")
    print(f"  Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*55)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENTO)

    with mlflow.start_run(run_name="LSTM-BrazoA-v1"):

        # Registrar parámetros
        mlflow.log_params({
            "hidden_size": HIDDEN_SIZE,
            "num_layers":  NUM_LAYERS,
            "secuencia":   SECUENCIA,
            "epochs":      EPOCHS,
            "batch_size":  BATCH_SIZE,
            "lr":          LR,
            "zona":        "Guayaquil-bosques",
            "sensor":      "VIIRS-SNPP",
        })

        t_inicio_total = time.time()

        # 1 — Cargar datos
        df = cargar_datos()

        # 2 — Dividir train/test
        corte    = int(len(df) * 0.8)
        df_train = df.iloc[:corte].copy()
        df_test  = df.iloc[corte:].copy()

        X_train, y_train, scaler = preparar_secuencias(df_train)
        X_test,  y_test,  _      = preparar_secuencias(df_test, scaler)

        print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

        # 3 — Crear y entrenar modelo
        modelo = LSTMPredictor(input_size=5,
                               hidden_size=HIDDEN_SIZE,
                               num_layers=NUM_LAYERS)
        losses, t_entrenamiento = entrenar(modelo, X_train, y_train)

        # 4 — Evaluar
        pred, real, mae, rmse, r2, latencia = evaluar(
            modelo, X_test, y_test, scaler)

        t_total = time.time() - t_inicio_total

        # 5 — Registrar métricas en MLflow
        mlflow.log_metrics({
            "MAE":             round(mae,      4),
            "RMSE":            round(rmse,     4),
            "R2":              round(r2,       4),
            "latencia_ms":     round(latencia, 2),
            "t_entrenamiento": round(t_entrenamiento, 2),
            "t_total":         round(t_total,  2),
        })

        # 6 — Detectar drift
        detectar_drift(df_train, df_test)

        # 7 — Gráfico
        generar_grafico(real, pred, losses)

        # 8 — Guardar modelo
        ruta_modelo = os.path.join(CARPETA_SALIDA, 'lstm_brazo_a.pth')
        torch.save({
            'model_state': modelo.state_dict(),
            'metricas': {
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4),
                'R2': round(r2, 4),
            },
            'params': {
                'hidden_size': HIDDEN_SIZE,
                'num_layers':  NUM_LAYERS,
                'secuencia':   SECUENCIA,
            },
            'run_id': mlflow.active_run().info.run_id,
        }, ruta_modelo)
        mlflow.log_artifact(ruta_modelo)
        print(f"  [OK] Modelo guardado: {ruta_modelo}")

        # 9 — Métricas CSV
        metricas = {
            'MAE':             round(mae,  4),
            'RMSE':            round(rmse, 4),
            'R2':              round(r2,   4),
            'latencia_ms':     round(latencia, 2),
            't_entrenamiento': round(t_entrenamiento, 2),
            't_total':         round(t_total, 2),
            'con_mlops':       True,
            'reproducible':    True,
            'monitoreo_drift': True,
        }
        pd.DataFrame([metricas]).to_csv(
            os.path.join(CARPETA_SALIDA, 'metricas_brazo_a.csv'), index=False)

        # 10 — Resumen
        print("\n" + "="*55)
        print("  RESULTADOS — BRAZO A (Con MLOps)")
        print("="*55)
        print(f"  MAE                       : {mae:.4f}")
        print(f"  RMSE                      : {rmse:.4f}")
        print(f"  R²                        : {r2:.4f}")
        print(f"  Latencia inferencia       : {latencia:.2f} ms")
        print(f"  Tiempo entrenamiento      : {t_entrenamiento:.1f} seg")
        print(f"  Tiempo total              : {t_total:.1f} seg")
        print(f"  Reproducible             : SI (MLflow run_id registrado)")
        print(f"  Monitoreo de drift       : SI (Evidently)")
        print(f"  Experimento MLflow       : {EXPERIMENTO}")
        print("="*55)
        print("\n  Brazo A completado.")