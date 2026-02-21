import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt
import matplotlib.ticker as ticker 

# ================================================================
# 1. PARÂMETROS FÍSICOS E DE CALIBRAÇÃO
# ================================================================

# --- Geometria do robô ---
WHEEL_DIAMETER_MM = 22.0
WHEEL_RADIUS_MM = WHEEL_DIAMETER_MM / 2.0


# --- Transmissão ---
GEAR_RATIO = 1.538
WHEEL_CIRCUMFERENCE_MM = 2.0 * math.pi * WHEEL_RADIUS_MM

# --- Caminhos de arquivos ---
LOG_FILE = "./data/dados_mqtt.txt"
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
# 3. CÁLCULO DE INS
# ================================================================

def compute_ins_pure(df: pd.DataFrame) -> pd.DataFrame:
    """INS Pura usando a Regra Trapezoidal e Filtro Passa-Baixa."""
    df_ins = df.copy()
    
    # --- PASSO 0: Configuração do Filtro ---

    df_ins['gyro_z_dps'] = df_ins['gyro_z_dps']
    df_ins['accel_x_ms2'] = df_ins['accel_x_ms2']
    df_ins['accel_y_ms2'] = df_ins['accel_y_ms2']

    # 1. Preparação dos dados (agora já filtrados)
    gyro_rad_s = np.radians(df_ins['gyro_z_dps'].to_numpy())
    time_s = df_ins['timestamp'].to_numpy() * 1e-3

    # 2. Integração do ângulo (Giroscópio -> Theta)
    theta = cumulative_trapezoid(gyro_rad_s, x=time_s, initial=0.0)
    df_ins['theta_ins'] = theta

    # 3. Rotação das acelerações para o referencial global
    ax = df_ins['accel_x_ms2'].to_numpy()
    ay = df_ins['accel_y_ms2'].to_numpy()
    ay_local=-ay
    ax_local=-ax
    
    ax_global = ax_local * np.cos(theta) - ay_local * np.sin(theta)
    ay_global = ax_local * np.sin(theta) + ay_local * np.cos(theta)


    df_ins['ax_global_ms2'] = ax_global
    df_ins['ay_global_ms2'] = ay_global

    # 4. Integração da aceleração -> velocidade
    vx = cumulative_trapezoid(ax_global, x=time_s, initial=0.0)
    vy = cumulative_trapezoid(ay_global, x=time_s, initial=0.0)

    df_ins['vx_ins_ms'] = vx
    df_ins['vy_ins_ms'] = vy
    df_ins['v_ins_ms'] = np.sqrt(vx**2 + vy**2)


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

def plot_acc_vel_x(df_ins,  save_path: str):
    time_s = df_ins['timestamp'] * 1e-3

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(time_s, df_ins['ax_global_ms2'], label='a_x (m/s²)')
    ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel('Aceleração X (m/s²)')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(time_s, df_ins['vx_ins_ms'], '--', label='v_x (m/s)')
    ax2.set_ylabel('Velocidade X (m/s)')

    # Legenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Aceleração e Velocidade no Eixo X')
    plt.savefig(save_path)    
    plt.tight_layout()  
    plt.close()

def plot_acc_vel_y(df_ins,  save_path: str):
    time_s = df_ins['timestamp'] * 1e-3

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(time_s, df_ins['ay_global_ms2'], label='a_y (m/s²)')
    ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel('Aceleração Y (m/s²)')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(time_s, df_ins['vy_ins_ms'], '--', label='v_y (m/s)')
    ax2.set_ylabel('Velocidade Y (m/s)')

    # Legenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Aceleração e Velocidade no Eixo Y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico salvo em: {save_path}")

def plot_trajectories(df: pd.DataFrame, save_path: str):
    """Gera o gráfico de INS com quadrados de 50mm de tamanho real fixo."""
    if 'x_ins' not in df.columns:
        return

    # Cálculo dos limites arredondados para múltiplos de 50
    margin = 50
    x_min, x_max = np.floor((df['x_ins'].min() - margin) / 50) * 50, np.ceil((df['x_ins'].max() + margin) / 50) * 50
    y_min, y_max = np.floor((df['y_ins'].min() - margin) / 50) * 50, np.ceil((df['y_ins'].max() + margin) / 50) * 50

    # Mantendo a mesma proporção de pixels por mm da função anterior
    res_scale = 0.5 / 50 
    fig_w = (x_max - x_min) * res_scale
    fig_h = (y_max - y_min) * res_scale

    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    plt.plot(df['x_ins'], df['y_ins'], '-o', markersize=2, linewidth=1.5, label='INS Pura')
    plt.plot(df.loc[0, 'x_ins'], df.loc[0, 'y_ins'], 'go', markersize=8, label='Início')
    plt.plot(df['x_ins'].iloc[-1], df['y_ins'].iloc[-1], 'ro', markersize=8, label='Fim')

    # Ajuste de Escala e Grid
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.title('INS')
    plt.xlabel('Posição X (mm)')
    plt.ylabel('Posição Y (mm)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Gráfico INS salvo: {save_path}")
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
    plot_acc_vel_y(df_ins,  os.path.join(OUTPUT_DIR, "acc_vel_y_image.png"))
    plot_acc_vel_x(df_ins,  os.path.join(OUTPUT_DIR, "acc_vel_x_image.png"))
    print(f"Dados de odometria salvos em: {ins_file}")

if __name__ == "__main__":
    main()


