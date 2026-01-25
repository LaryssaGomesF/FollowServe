import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt

# ================================================================
# 1. PARÂMETROS FÍSICOS E DE CALIBRAÇÃO
# ================================================================

# --- Geometria do robô ---
WHEEL_DIAMETER_MM = 22.0
WHEEL_RADIUS_MM = WHEEL_DIAMETER_MM / 2.0
WHEEL_BASE_MM = 120.0

# --- Transmissão ---
GEAR_RATIO = 1.538
WHEEL_CIRCUMFERENCE_MM = 2.0 * math.pi * WHEEL_RADIUS_MM

# --- Caminhos de arquivos ---
LOG_FILE = "./data/log.txt"
OUTPUT_DIR = "./data/ins"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Constantes do sensor IMU ---
GYRO_SENSITIVITY = 131.0     # LSB / (°/s)
ACCEL_SENSITIVITY = 16384.0  # LSB / g
G_TO_MS2 = 9.80665           # 1 g = 9.80665 m/s²

# ================================================================
# 2. LEITURA E PROCESSAMENTO DO ARQUIVO DE LOG
# ================================================================

COLUMN_MAP = [
    'timestamp',
    'gyroX', 'gyroY', 'gyroZ',
    'accelX', 'accelY', 'accelZ'
]

def map_reading_to_dict(reading: List[Any]) -> Dict[str, Any]:
    if len(reading) < 7:
        raise ValueError("Leitura menor que o esperado.")
    return {
        'timestamp': reading[0],
        'gyroX': reading[1], 
        'gyroY': reading[2], 
        'gyroZ': reading[3], 
        'accelX': reading[4], 
        'accelY': reading[5], 
        'accelZ': reading[6],           
    }

def convert_imu_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Converte leituras brutas do IMU para unidades físicas (dps e m/s²)."""
    df['gyro_x_dps'] = df['gyroX'] / GYRO_SENSITIVITY
    df['gyro_y_dps'] = df['gyroY'] / GYRO_SENSITIVITY
    df['gyro_z_dps'] = df['gyroZ'] / GYRO_SENSITIVITY

    df['accel_x_ms2'] = (df['accelX'] / ACCEL_SENSITIVITY) * G_TO_MS2
    df['accel_y_ms2'] = (df['accelY'] / ACCEL_SENSITIVITY) * G_TO_MS2
    df['accel_z_ms2'] = (df['accelZ'] / ACCEL_SENSITIVITY) * G_TO_MS2
    return df

def load_log_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        print(f"Arquivo de log não encontrado: {filepath}")
        return pd.DataFrame()

    data_list = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                reading_array = json.loads(line.strip())
                data_list.append(map_reading_to_dict(reading_array))
            except Exception:
                continue

    if not data_list:
        print("Nenhum dado válido encontrado no log.")
        return pd.DataFrame()

    df = pd.DataFrame(data_list)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df['dt'] = df['timestamp'].diff().fillna(0)  # diferença em µs
    df = convert_imu_readings(df)
    return df

# ================================================================
# 3. FILTRO PASSA-BAIXA
# ================================================================

def apply_low_pass_filter(data, cutoff_freq, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# ================================================================
# 3. CÁLCULO DE INS
# ================================================================

def compute_ins_pure(df: pd.DataFrame) -> pd.DataFrame:
    """INS Pura usando a Regra Trapezoidal e Filtro Passa-Baixa."""
    df_ins = df.copy()
    
    # --- PASSO 0: Configuração do Filtro ---
    # Calculamos a frequência de amostragem (fs) real baseada no log
    dt_mean = df_ins['timestamp'].diff().mean() * 1e-6
    fs = 1.0 / dt_mean
    cutoff = 5.0  # Frequência de corte (5Hz é bom para vibração de motores)

    # --- PASSO 1: Aplicação do Filtro nos dados brutos ---
    # Filtramos o giro e as acelerações ANTES de começar os cálculos
    df_ins['gyro_z_dps'] = apply_low_pass_filter(df_ins['gyro_z_dps'].values, cutoff, fs)
    df_ins['accel_x_ms2'] = apply_low_pass_filter(df_ins['accel_x_ms2'].values, cutoff, fs)
    df_ins['accel_y_ms2'] = apply_low_pass_filter(df_ins['accel_y_ms2'].values, cutoff, fs)

    # 1. Preparação dos dados (agora já filtrados)
    gyro_rad_s = np.radians(df_ins['gyro_z_dps'].to_numpy())
    time_s = df_ins['timestamp'].to_numpy() * 1e-6 

    # 2. Integração do ângulo (Giroscópio -> Theta)
    theta = cumulative_trapezoid(gyro_rad_s, x=time_s, initial=0.0)
    df_ins['theta_ins'] = theta

    # 3. Rotação das acelerações para o referencial global
    ax = df_ins['accel_x_ms2'].to_numpy()
    ay = df_ins['accel_y_ms2'].to_numpy()
    
    ax_global = ax * np.cos(theta) - ay * np.sin(theta)
    ay_global = ax * np.sin(theta) + ay * np.cos(theta)

    # 4. Integração da aceleração -> velocidade
    vx = cumulative_trapezoid(ax_global, x=time_s, initial=0.0)
    vy = cumulative_trapezoid(ay_global, x=time_s, initial=0.0)

    # 5. Integração da velocidade -> posição (em metros)
    x_m = cumulative_trapezoid(vx, x=time_s, initial=0.0)
    y_m = cumulative_trapezoid(vy, x=time_s, initial=0.0)

    # 6. Conversão para mm
    df_ins['x_ins'] = x_m * 1000.0
    df_ins['y_ins'] = y_m * 1000.0

    return df_ins
# ================================================================
# 4. PLOTAGEM
# ================================================================

def plot_trajectories(df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(12, 10))
    if 'x_ins' in df.columns:
        plt.plot(df['x_ins'], df['y_ins'], '-o', markersize=2, linewidth=1.5, label='INS Pura')
        plt.plot(df.loc[0, 'x_ins'], df.loc[0, 'y_ins'], 'go', markersize=8, label='Início')
        plt.plot(df.loc[len(df)-1, 'x_ins'], df.loc[len(df)-1, 'y_ins'], 'ro', markersize=8, label='Fim')

    plt.title('INS')
    plt.xlabel('Posição X (mm)')
    plt.ylabel('Posição Y (mm)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico salvo em: {save_path}")

# ================================================================
# 5. MAIN
# ================================================================

def main():
    df = load_log_data(LOG_FILE)
    if df.empty:
        return

    formatted_file = os.path.join(OUTPUT_DIR, "formatted_ins.csv")
    df.to_csv(formatted_file, index=False)
    print(f"Dados convertidos e formatados salvos em: {formatted_file}")

    df_ins = compute_ins_pure(df.copy())

    ins_file = os.path.join(OUTPUT_DIR, "ins.csv")
    df_ins.to_csv(ins_file, index=False)
    plot_trajectories(df_ins, os.path.join(OUTPUT_DIR, "ins_image.png"))
    print(f"Dados de odometria salvos em: {ins_file}")

if __name__ == "__main__":
    main()
