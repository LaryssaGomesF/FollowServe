import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any

# ================================================================
# 1. PARÂMETROS FÍSICOS E DE CALIBRAÇÃO
# ================================================================

# --- Geometria do robô ---
WHEEL_DIAMETER_MM = 22.0     # 22 mm
WHEEL_RADIUS_MM = WHEEL_DIAMETER_MM / 2.0
WHEEL_BASE_MM = 120.0        # 120 mm

# --- Transmissão ---
GEAR_RATIO = 1.538  # Relação de redução (motor : roda)
WHEEL_CIRCUMFERENCE_MM = 2.0 * math.pi * WHEEL_RADIUS_MM  # mm

# --- Constantes do sensor IMU ---
GYRO_SENSITIVITY = 131.0     # LSB / (°/s)
ACCEL_SENSITIVITY = 16384.0  # LSB / g
G_TO_MS2 = 9.80665           # 1 g = 9.80665 m/s²

# --- Caminhos de arquivos ---
LOG_FILE = "data/log.txt"
OUTPUT_DIR = "data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 2. FUNÇÕES DE CONVERSÃO DE SENSORES
# ================================================================

def convert_imu_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Converte as leituras brutas do IMU para unidades físicas (dps e m/s²)."""
    df['gyro_x_dps'] = df['gyroX'] / GYRO_SENSITIVITY
    df['gyro_y_dps'] = df['gyroY'] / GYRO_SENSITIVITY
    df['gyro_z_dps'] = df['gyroZ'] / GYRO_SENSITIVITY

    df['accel_x_ms2'] = (df['accelX'] / ACCEL_SENSITIVITY) * G_TO_MS2
    df['accel_y_ms2'] = (df['accelY'] / ACCEL_SENSITIVITY) * G_TO_MS2
    df['accel_z_ms2'] = (df['accelZ'] / ACCEL_SENSITIVITY) * G_TO_MS2
    return df

# ================================================================
# 3. LEITURA E PROCESSAMENTO DO ARQUIVO DE LOG
# ================================================================

COLUMN_MAP = [
    'timestamp',
    'gyroX', 'gyroY', 'gyroZ',
    'accelX', 'accelY', 'accelZ',
    'encoderR', 'encoderL'
]

def map_reading_to_dict(reading: List[Any]) -> Dict[str, Any]:
    """Mapeia uma única leitura (array) para um dicionário usando o COLUMN_MAP."""
    if len(reading) != len(COLUMN_MAP):
        raise ValueError(f"Tamanho da leitura ({len(reading)}) não corresponde ao mapa de colunas ({len(COLUMN_MAP)}).")
    return dict(zip(COLUMN_MAP, reading))

def load_log_data(filepath: str) -> pd.DataFrame:
    """Carrega o arquivo de log e retorna um DataFrame ordenado por tempo."""
    if not os.path.exists(filepath):
        print(f"Arquivo de log não encontrado: {filepath}")
        return pd.DataFrame()

    data_list = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                reading_array: List[Any] = json.loads(line.strip())
                reading_dict = map_reading_to_dict(reading_array)
                data_list.append(reading_dict)
            except Exception:
                continue

    if not data_list:
        print("Nenhum dado válido encontrado no log.")
        return pd.DataFrame()

    df = pd.DataFrame(data_list)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df['timestamp_s'] = df['timestamp'] / 1e6
    df['dt'] = df['timestamp_s'].diff().fillna(0)
    df = convert_imu_readings(df)

    return df

# ================================================================
# 4. ABORDAGENS DE ODOMETRIA
# ================================================================

def compute_odometry_pure(df: pd.DataFrame) -> pd.DataFrame:
    """Odometria Pura (Encoders) - unidades em mm."""
    df_odom = df.copy()
    df_odom['x_odom'] = 0.0
    df_odom['y_odom'] = 0.0
    df_odom['theta_odom'] = 0.0

    prev_R = df_odom.loc[0, 'encoderR']
    prev_L = df_odom.loc[0, 'encoderL']

    for i in range(1, len(df_odom)):
        prev_x = df_odom.loc[i - 1, 'x_odom']
        prev_y = df_odom.loc[i - 1, 'y_odom']
        prev_theta = df_odom.loc[i - 1, 'theta_odom']

        curr_R = df_odom.loc[i, 'encoderR']
        curr_L = df_odom.loc[i, 'encoderL']

        delta_R = curr_R - prev_R
        delta_L = curr_L - prev_L
        prev_R, prev_L = curr_R, curr_L

        THRESHOLD_LAPS = 0.25
        if abs(delta_R) < THRESHOLD_LAPS and abs(delta_L) < THRESHOLD_LAPS:
            dist_R = dist_L = 0.0
        else:
            dist_R = (delta_R / GEAR_RATIO) * WHEEL_CIRCUMFERENCE_MM
            dist_L = (delta_L / GEAR_RATIO) * WHEEL_CIRCUMFERENCE_MM

        delta_S = (dist_R + dist_L) / 2.0
        delta_theta = (dist_R - dist_L) / WHEEL_BASE_MM
        new_theta = prev_theta + delta_theta
        theta_avg = prev_theta + (delta_theta / 2.0)

        new_x = prev_x + delta_S * math.cos(theta_avg)
        new_y = prev_y + delta_S * math.sin(theta_avg)

        df_odom.loc[i, 'x_odom'] = new_x
        df_odom.loc[i, 'y_odom'] = new_y
        df_odom.loc[i, 'theta_odom'] = new_theta

    return df_odom

def compute_ins_pure(df: pd.DataFrame) -> pd.DataFrame:
    """INS Pura (IMU) - posição final em mm."""
    df_ins = df.copy()
    df_ins['x_ins'] = 0.0
    df_ins['y_ins'] = 0.0
    df_ins['theta_ins'] = 0.0

    vx = vy = theta = 0.0

    for i in range(1, len(df_ins)):
        dt = df_ins.loc[i, 'dt']
        gyro_z_rad = math.radians(df_ins.loc[i, 'gyro_z_dps'])
        theta += gyro_z_rad * dt

        ax_local = df_ins.loc[i, 'accel_x_ms2']
        ay_local = df_ins.loc[i, 'accel_y_ms2']

        ax_global = ax_local * math.cos(theta) - ay_local * math.sin(theta)
        ay_global = ax_local * math.sin(theta) + ay_local * math.cos(theta)

        vx += ax_global * dt
        vy += ay_global * dt

        # Convertendo m → mm (1 m = 1000 mm)
        df_ins.loc[i, 'x_ins'] = df_ins.loc[i - 1, 'x_ins'] + vx * dt * 1000
        df_ins.loc[i, 'y_ins'] = df_ins.loc[i - 1, 'y_ins'] + vy * dt * 1000
        df_ins.loc[i, 'theta_ins'] = theta

    return df_ins

def compute_odometry_fused(df: pd.DataFrame) -> pd.DataFrame:
    """Odometria Fusionada (Encoders + Giroscópio) - unidades em mm."""
    df_fused = df.copy()
    df_fused['x_fused'] = 0.0
    df_fused['y_fused'] = 0.0
    df_fused['theta_fused'] = 0.0

    prev_R = df_fused.loc[0, 'encoderR']
    prev_L = df_fused.loc[0, 'encoderL']
    theta = 0.0

    for i in range(1, len(df_fused)):
        prev_x = df_fused.loc[i - 1, 'x_fused']
        prev_y = df_fused.loc[i - 1, 'y_fused']
        dt = df_fused.loc[i, 'dt']

        curr_R = df_fused.loc[i, 'encoderR']
        curr_L = df_fused.loc[i, 'encoderL']
        delta_R = curr_R - prev_R
        delta_L = curr_L - prev_L
        prev_R, prev_L = curr_R, curr_L

        THRESHOLD_LAPS = 0.695
        if abs(delta_R) < THRESHOLD_LAPS and abs(delta_L) < THRESHOLD_LAPS:
            dist_R = dist_L = 0.0
        else:
            dist_R = (delta_R / GEAR_RATIO) * WHEEL_CIRCUMFERENCE_MM
            dist_L = (delta_L / GEAR_RATIO) * WHEEL_CIRCUMFERENCE_MM

        delta_S = (dist_R + dist_L) / 2.0
        gyro_z_rad = math.radians(df_fused.loc[i, 'gyro_z_dps'])
        theta += gyro_z_rad * dt

        new_x = prev_x + delta_S * math.cos(theta)
        new_y = prev_y + delta_S * math.sin(theta)

        df_fused.loc[i, 'x_fused'] = new_x
        df_fused.loc[i, 'y_fused'] = new_y
        df_fused.loc[i, 'theta_fused'] = theta

    return df_fused

# ================================================================
# 5. VISUALIZAÇÃO
# ================================================================

def plot_trajectories(df: pd.DataFrame, save_path: str):
    """Gera e salva o gráfico das trajetórias (em mm)."""
    plt.figure(figsize=(12, 10))
    if 'x_odom' in df.columns:
        plt.plot(df['x_odom'], df['y_odom'], '-o', markersize=2, linewidth=1.5, label='1. Odometria Pura (Encoders)')
        plt.plot(df.loc[0, 'x_odom'], df.loc[0, 'y_odom'], 'go', markersize=8, label='Início')
        plt.plot(df.loc[len(df)-1, 'x_odom'], df.loc[len(df)-1, 'y_odom'], 'ro', markersize=8, label='Fim')

    if 'x_ins' in df.columns:
        plt.plot(df['x_ins'], df['y_ins'], '-o', markersize=2, linewidth=1.5, label='2. INS Pura (IMU)')
        plt.plot(df.loc[0, 'x_ins'], df.loc[0, 'y_ins'], 'go', markersize=8, label='Início')
        plt.plot(df.loc[len(df)-1, 'x_ins'], df.loc[len(df)-1, 'y_ins'], 'ro', markersize=8, label='Fim')

    if 'x_fused' in df.columns:
        plt.plot(df['x_fused'], df['y_fused'], '-o', markersize=2, linewidth=1.5, label='3. Odometria + Giroscópio')
        plt.plot(df.loc[0, 'x_fused'], df.loc[0, 'y_fused'], 'go', markersize=8, label='Início')
        plt.plot(df.loc[len(df)-1, 'x_fused'], df.loc[len(df)-1, 'y_fused'], 'ro', markersize=8, label='Fim')

    plt.title('Comparação das Trajetórias Estimadas')
    plt.xlabel('Posição X (mm)')
    plt.ylabel('Posição Y (mm)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico salvo em: {save_path}")

# ================================================================
# 6. MAIN
# ================================================================

def main():
    df = load_log_data(LOG_FILE)
    if df.empty:
        return

    df_odom = compute_odometry_pure(df.copy())
    df_ins = compute_ins_pure(df.copy())
    df_fused = compute_odometry_fused(df.copy())

    df_results = df[['timestamp_s', 'dt']].copy()
    df_results['x_odom'] = df_odom['x_odom']
    df_results['y_odom'] = df_odom['y_odom']
    df_results['theta_odom'] = df_odom['theta_odom']
    df_results['x_ins'] = df_ins['x_ins']
    df_results['y_ins'] = df_ins['y_ins']
    df_results['theta_ins'] = df_ins['theta_ins']
    df_results['x_fused'] = df_fused['x_fused']
    df_results['y_fused'] = df_fused['y_fused']
    df_results['theta_fused'] = df_fused['theta_fused']

    processed_file = os.path.join(OUTPUT_DIR, "processed_trajectories_mm.csv")
    df_results.to_csv(processed_file, index=False)
    print(f"Dados processados salvos em: {processed_file}")

    plot_path = os.path.join(OUTPUT_DIR, "trajectory_comparison_mm.png")
    plot_trajectories(df_results, plot_path)

if __name__ == "__main__":
    main()
