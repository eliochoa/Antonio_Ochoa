"""
brazo_b_solo_genai.py
Modelo LSTM para predicción de focos de calor en Guayaquil
SIN prácticas MLOps — script manual, sin versionado, sin monitoreo
Brazo B del experimento comparativo
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

# ─── CONFIGURACIÓN ───────────────────────────────────────────────────────────
CARPETA_DATOS  = os.path.join(os.path.dirname(__file__), 'datos')
CARPETA_SALIDA = os.path.join(os.path.dirname(__file__), 'brazo_b_solo_genai')

SEMILLA     = 42
SECUENCIA   = 30       # días de historia para predecir
HORIZONTE   = 1        # días a predecir
EPOCHS      = 50
BATCH_SIZE  = 16
LR          = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS  = 2

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
    df = df.fillna(method='ffill').fillna(0)

    print(f"  [OK] Dataset: {len(df)} días · {df.columns.tolist()}")
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
        if (epoch + 1) % 10 == 0:
            print(f"    Época {epoch+1:3d}/{EPOCHS} — Loss: {avg_loss:.6f}")

    t_total = time.time() - t_inicio
    print(f"  [OK] Entrenamiento: {t_total:.1f} seg")
    return losses, t_total


def evaluar(modelo, X_test, y_test, scaler):
    modelo.eval()
    with torch.no_grad():
        pred_scaled = modelo(X_test).numpy()
        real_scaled = y_test.numpy()

    # Desnormalizar solo la columna focos
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

    return pred_real, real_real, mae, rmse, r2


def generar_grafico_prediccion(real, pred, losses):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1 — Predicción vs Real
    ax = axes[0]
    ax.plot(real, label='Real', color='#2E86AB', linewidth=1.5)
    ax.plot(pred, label='Predicho', color='#E74C3C', linewidth=1.5, linestyle='--')
    ax.set_title('Brazo B — Predicción vs Real\n(LSTM sin MLOps)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Días (conjunto de prueba)', fontsize=11)
    ax.set_ylabel('Focos de Calor', fontsize=11)
    ax.legend(fontsize=10)

    # Gráfico 2 — Curva de pérdida
    ax2 = axes[1]
    ax2.plot(range(1, len(losses)+1), losses,
             color='#E67E22', linewidth=2)
    ax2.set_title('Curva de Pérdida — Entrenamiento\n(Brazo B sin MLOps)',
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Época', fontsize=11)
    ax2.set_ylabel('MSE Loss', fontsize=11)

    plt.tight_layout()
    ruta = os.path.join(CARPETA_SALIDA, 'brazo_b_prediccion.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Gráfico guardado: {ruta}")


def guardar_modelo(modelo, metricas):
    ruta = os.path.join(CARPETA_SALIDA, 'lstm_brazo_b.pth')
    torch.save({
        'model_state': modelo.state_dict(),
        'metricas':    metricas,
        'params': {
            'hidden_size': HIDDEN_SIZE,
            'num_layers':  NUM_LAYERS,
            'secuencia':   SECUENCIA,
            'epochs':      EPOCHS,
        }
    }, ruta)
    print(f"  [OK] Modelo guardado: {ruta}")


# ─── EJECUCIÓN PRINCIPAL ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  BRAZO B — LSTM SIN MLOps")
    print(f"  Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*55)

    t_inicio_total = time.time()

    # 1 — Cargar datos
    df = cargar_datos()

    # 2 — Dividir train/test (80/20)
    corte = int(len(df) * 0.8)
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
    pred, real, mae, rmse, r2 = evaluar(modelo, X_test, y_test, scaler)

    t_total = time.time() - t_inicio_total

    metricas = {
        'MAE':              round(mae,  4),
        'RMSE':             round(rmse, 4),
        'R2':               round(r2,   4),
        't_entrenamiento':  round(t_entrenamiento, 2),
        't_total':          round(t_total, 2),
        'epochs':           EPOCHS,
        'sin_mlops':        True,
    }

    # 5 — Resultados
    print("\n" + "="*55)
    print("  RESULTADOS — BRAZO B (Sin MLOps)")
    print("="*55)
    print(f"  MAE                       : {mae:.4f}")
    print(f"  RMSE                      : {rmse:.4f}")
    print(f"  R²                        : {r2:.4f}")
    print(f"  Tiempo entrenamiento      : {t_entrenamiento:.1f} seg")
    print(f"  Tiempo total              : {t_total:.1f} seg")
    print(f"  Reproducible             : NO (sin versionado)")
    print(f"  Monitoreo de drift       : NO")
    print(f"  API de inferencia        : NO")
    print("="*55)

    # 6 — Guardar
    generar_grafico_prediccion(real, pred, losses)
    guardar_modelo(modelo, metricas)

    # 7 — Guardar métricas en CSV
    pd.DataFrame([metricas]).to_csv(
        os.path.join(CARPETA_SALIDA, 'metricas_brazo_b.csv'), index=False)
    print(f"  [OK] Métricas guardadas en brazo_b_solo_genai/metricas_brazo_b.csv")
    print("\n  Brazo B completado.")