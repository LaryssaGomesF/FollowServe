import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import os

# ================================================================
# 1. PARÂMETROS FÍSICOS E DE CALIBRAÇÃO
# ================================================================

# --- Geometria do robô ---
WHEEL_DIAMETER_M = 0.022  # Diâmetro da roda (m)
WHEEL_BASE_M = 0.093      # Distância entre rodas (m)

# --- Transmissão ---
GEAR_RATIO = 1.538  # Relação de redução (motor : roda)

# --- Constantes do sensor IMU ---
GYRO_SENSITIVITY = 131.0     # LSB / (°/s)
ACCEL_SENSITIVITY = 16384.0  # LSB / g
G_TO_MS2 = 9.80665           # 1 g = 9.80665 m/s²

# --- Caminhos de arquivos ---
LOG_FILE = "data/log.txt"
PROCESSED_FILE = "data/processed_data.csv"
TRAJECTORY_PLOT = "data/trajectory_plot.png"

# ================================================================
# 2. FUNÇÕES DE CONVERSÃO DE SENSORES
# ================================================================

def convert_imu_readings(row: pd.Series) -> pd.Series:
    """Converte as leituras brutas do IMU para unidades físicas."""
    row['gyro_x_dps'] = row['gyroX'] / GYRO_SENSITIVITY
    row['gyro_y_dps'] = row['gyroY'] / GYRO_SENSITIVITY
    row['gyro_z_dps'] = row['gyroZ'] / GYRO_SENSITIVITY

    row['accel_x_ms2'] = (row['accelX'] / ACCEL_SENSITIVITY) * G_TO_MS2
    row['accel_y_ms2'] = (row['accelY'] / ACCEL_SENSITIVITY) * G_TO_MS2
    row['accel_z_ms2'] = (row['accelZ'] / ACCEL_SENSITIVITY) * G_TO_MS2
    return row

# ================================================================
# 3. LEITURA E PROCESSAMENTO DO ARQUIVO DE LOG
# ================================================================

def load_log_data(filepath: str) -> pd.DataFrame:
    """Carrega o arquivo de log (JSON por linha) e retorna um DataFrame ordenado por tempo."""
    if not os.path.exists(filepath):
        print(f"Arquivo de log não encontrado: {filepath}")
        return pd.DataFrame()

    data_list = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data_list.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Linha inválida ignorada: {line.strip()}")
                continue

    if not data_list:
        print("Nenhum dado válido encontrado no log.")
        return pd.DataFrame()

    df = pd.DataFrame(data_list)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df['timestamp_s'] = df['timestamp'] / 1000.0
    df['dt'] = df['timestamp_s'].diff().fillna(0)

    return df

# ================================================================
# 4. ODOMETRIA E FUSÃO SENSORIAL
# ================================================================

def compute_odometry_and_fusion(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df

    # Converte IMU
    df = df.apply(convert_imu_readings, axis=1)

    # Inicializa colunas de pose
    df['x'] = 0.0
    df['y'] = 0.0
    df['theta'] = 0.0

    df['linear_displacement'] = 0.0
    df['delta_theta_odom'] = 0.0
    df['delta_theta_gyro'] = 0.0

    # Pega valores iniciais dos encoders
    prev_rev_R = df.loc[0, 'encoderR']
    prev_rev_L = df.loc[0, 'encoderL']

    wheel_radius = WHEEL_DIAMETER_M / 2
    wheel_circumference = 2 * math.pi * wheel_radius

    # Loop principal de integração
    for i in range(1, len(df)):
        dt = df.loc[i, 'dt']
        prev_x = df.loc[i - 1, 'x']
        prev_y = df.loc[i - 1, 'y']
        prev_theta = df.loc[i - 1, 'theta']

        # --- 1. ODOMETRIA ---
        current_rev_R = df.loc[i, 'encoderR'] / GEAR_RATIO
        current_rev_L = df.loc[i, 'encoderL'] / GEAR_RATIO

        # Variação das rotações medidas pelos encoders (motor)
        delta_rev_R_motor = current_rev_R - prev_rev_R
        delta_rev_L_motor = current_rev_L - prev_rev_L

        # Atualiza estados anteriores
        prev_rev_R = current_rev_R
        prev_rev_L = current_rev_L

        # Corrige para rotação real da roda 
        delta_rev_R_wheel = delta_rev_R_motor 
        delta_rev_L_wheel = delta_rev_L_motor 

        # Distância percorrida por cada roda
        dist_R = delta_rev_R_wheel * wheel_circumference
        dist_L = delta_rev_L_wheel * wheel_circumference

        # Deslocamento médio e variação angular da odometria
        delta_S = (dist_R + dist_L) / 2.0
        delta_theta_odom = (dist_R - dist_L) / WHEEL_BASE_M

        df.loc[i, 'linear_displacement'] = delta_S
        df.loc[i, 'delta_theta_odom'] = delta_theta_odom

        # --- 2. GIROSCÓPIO ---
        gyro_z_rad_s = math.radians(df.loc[i, 'gyro_z_dps'])
        delta_theta_gyro = gyro_z_rad_s * dt
        df.loc[i, 'delta_theta_gyro'] = delta_theta_gyro

        # --- 3. FUSÃO (atual: usa apenas giroscópio para rotação) ---
        delta_theta_fused = delta_theta_gyro  # Pode futuramente usar filtro complementar

        # Atualiza pose
        new_theta = prev_theta + delta_theta_fused
        theta_avg = prev_theta + (delta_theta_fused / 2.0)

        new_x = prev_x + delta_S * math.cos(theta_avg)
        new_y = prev_y + delta_S * math.sin(theta_avg)

        df.loc[i, 'x'] = new_x
        df.loc[i, 'y'] = new_y
        df.loc[i, 'theta'] = new_theta

    return df

# ================================================================
# 5. VISUALIZAÇÃO
# ================================================================

def plot_robot_trajectory(df: pd.DataFrame, save_path: str = TRAJECTORY_PLOT):
    """Gera e salva o gráfico da trajetória (x, y)."""
    if df.empty or 'x' not in df.columns:
        print("Sem dados para plotar.")
        return

    plt.figure(figsize=(10, 8))
    plt.plot(df['x'], df['y'], label='Trajetória (com redução 1:1.538)')
    plt.xlabel('Posição X (m)')
    plt.ylabel('Posição Y (m)')
    plt.title('Trajetória Estimada do Robô (ajustada pela redução)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Gráfico salvo em: {save_path}")

# ================================================================
# 6. EXECUÇÃO PRINCIPAL
# ================================================================

def main():
    print("=== Iniciando processamento de dados do robô ===")
    os.makedirs("data", exist_ok=True)

    # Cria log de exemplo se não existir
    if not os.path.exists(LOG_FILE):
        print(f"Arquivo {LOG_FILE} não encontrado. Criando exemplo...")
        example_data = [
            {"gyroX": 0, "gyroY": 0, "gyroZ": 0,    "accelX": 0, "accelY": 0, "accelZ": 16384, "encoderR": 0.0,  "encoderL": 0.0,  "timestamp": 1000},
            {"gyroX": 0, "gyroY": 0, "gyroZ": 0,    "accelX": 0, "accelY": 0, "accelZ": 16384, "encoderR": 1.0,  "encoderL": 1.0,  "timestamp": 2000},
            {"gyroX": 0, "gyroY": 0, "gyroZ": 1310, "accelX": 0, "accelY": 0, "accelZ": 16384, "encoderR": 2.0,  "encoderL": 1.5,  "timestamp": 3000},
            {"gyroX": 0, "gyroY": 0, "gyroZ": 1310, "accelX": 0, "accelY": 0, "accelZ": 16384, "encoderR": 3.0,  "encoderL": 1.5,  "timestamp": 4000},
            {"gyroX": 0, "gyroY": 0, "gyroZ": 0,    "accelX": 0, "accelY": 0, "accelZ": 16384, "encoderR": 4.0,  "encoderL": 2.5,  "timestamp": 5000},
        ]
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            for item in example_data:
                f.write(json.dumps(item) + "\n")
        print("Arquivo de exemplo criado.")

    # Processamento
    df_raw = load_log_data(LOG_FILE)
    df_processed = compute_odometry_and_fusion(df_raw)
    df_processed.to_csv(PROCESSED_FILE, index=False)
    print(f"Dados processados salvos em: {PROCESSED_FILE}")

    # Gráfico
    plot_robot_trajectory(df_processed)

if __name__ == "__main__":
    main()
