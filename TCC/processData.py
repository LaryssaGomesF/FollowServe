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
PROCESSED_FILE_ODOMETRY = "data/processed_data_odometry.csv"
TRAJECTORY_PLOT_ODOMETRY = "data/trajectory_plot_odometry.png"
TRAJECTORY_PLOT = "data/trajectory_plot.png"

PROCESSED_FILE_INS = "data/processed_data_ins.csv"
TRAJECTORY_PLOT_INS = "data/trajectory_plot_ins.png"
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
# 4. ODOMETRIA 
# ================================================================
def compute_odometry(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df

    # Inicializa colunas de pose
    df['x'] = 0.0
    df['y'] = 0.0
    df['theta'] = 0.0

    df['linear_displacement'] = 0.0
    df['delta_theta_odom'] = 0.0


    # Pega valores iniciais dos encoders
    prev_accumulated_laps_R = df.loc[0, 'encoderR']
    prev_accumulated_laps_L = df.loc[0, 'encoderL']

    wheel_radius = WHEEL_DIAMETER_M / 2
    wheel_circumference = 2 * math.pi * wheel_radius

    # Loop principal de integração
    for i in range(1, len(df)):
        prev_x = df.loc[i - 1, 'x']
        prev_y = df.loc[i - 1, 'y']
        prev_theta = df.loc[i - 1, 'theta']

        # --- 1. ODOMETRIA ---
        current_acummulated_laps_R = df.loc[i, 'encoderR'] 
        current_acummulated_laps_L = df.loc[i, 'encoderL'] 

        # Variação das rotações medidas pelos encoders (motor)
        delta_laps_R = current_acummulated_laps_R - prev_accumulated_laps_R
        delta_laps_L= current_acummulated_laps_L - prev_accumulated_laps_L

        # Atualiza estados anteriores
        prev_accumulated_laps_R = current_acummulated_laps_R
        prev_accumulated_laps_L = current_acummulated_laps_L


        # Distância percorrida por cada roda
        dist_R = delta_laps_R * wheel_circumference
        dist_L = delta_laps_L * wheel_circumference

        # Deslocamento médio e variação angular da odometria
        delta_S = (dist_R + dist_L) / 2.0
        delta_theta_odom = (dist_R - dist_L) / WHEEL_BASE_M

        df.loc[i, 'linear_displacement'] = delta_S
        df.loc[i, 'delta_theta_odom'] = delta_theta_odom

        # Atualiza pose
        new_theta = prev_theta + delta_theta_odom
        theta_avg = prev_theta + (delta_theta_odom / 2.0)

        new_x = prev_x + delta_S * math.cos(theta_avg)
        new_y = prev_y + delta_S * math.sin(theta_avg)

        df.loc[i, 'x'] = new_x
        df.loc[i, 'y'] = new_y
        df.loc[i, 'theta'] = new_theta

    return df



# ================================================================
# 4. Inertial Navigation System 
# ================================================================
def compute_ins(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df

    # Converte IMU
    df = df.apply(convert_imu_readings, axis=1)

    # Inicializa colunas de pose
    df['x'] = 0.0
    df['y'] = 0.0
    df['theta'] = 0.0

    vx = vy = 0.0
    new_x = new_y = 0.0
    theta = 0.0

    df['delta_theta_ins'] = 0.0

    # Loop principal de integração
    for i in range(1, len(df)):
        prev_t = df.loc[i - 1 , 'dt']
        prev_theta = df.loc[i - 1, 'theta']

        current_gyro = df.loc[i, 'gyro_z_dps']
        current_accel_x = df.loc[i, 'accel_x_ms2']
        current_accel_y = df.loc[i, 'accel_y_ms2']
        current_t =  df.loc[i, 'dt']
      
        dt = current_t - prev_t
        gyro_z_rad = math.radians(current_gyro)
        # Estima a orientacao
        theta = prev_theta + gyro_z_rad*dt

        # Muda pra referencia global
        a_global_x = current_accel_x * math.cos(theta) - current_accel_y * math.sin(theta)
        a_global_y = current_accel_x * math.sin(theta) + current_accel_y * math.cos(theta)

        # Duas integracoes para calcular a posicao
        vx += a_global_x * dt
        vy += a_global_y * dt
        new_x += vx * dt
        new_y += vy * dt

        df.loc[i, 'x'] = new_x
        df.loc[i, 'y'] = new_y
        df.loc[i, 'theta'] = theta


    return df



# ================================================================
# 5. VISUALIZAÇÃO
# ================================================================

def plot_robot_trajectory(df: pd.DataFrame, save_path: str = TRAJECTORY_PLOT):
    """Gera e salva o gráfico da trajetória (x, y) com formatação aprimorada."""
    if df.empty or 'x' not in df.columns or 'y' not in df.columns:
        print("Sem dados para plotar.")
        return

    plt.figure(figsize=(10, 8))

    # Linha contínua + pontos marcados
    plt.plot(df['x'], df['y'], '-o', markersize=4, linewidth=1.5, label='Trajetória (1:1.538)')

    # Mostra a direção do movimento com setas
    step = max(1, len(df) // 30)  # controla número de setas (≈30 setas máx)
    for i in range(0, len(df)-step, step):
        plt.arrow(df['x'].iloc[i], df['y'].iloc[i],
                  df['x'].iloc[i+step] - df['x'].iloc[i],
                  df['y'].iloc[i+step] - df['y'].iloc[i],
                  shape='full', lw=0, length_includes_head=True,
                  head_width=0.02, color='orange', alpha=0.6)

    # Marca início e fim
    plt.scatter(df['x'].iloc[0], df['y'].iloc[0], c='green', s=60, label='Início')
    plt.scatter(df['x'].iloc[-1], df['y'].iloc[-1], c='red', s=60, label='Fim')

    plt.xlabel('Posição X (m)')
    plt.ylabel('Posição Y (m)')
    plt.title('Trajetória Estimada do Robô (ajustada pela redução)')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Gráfico salvo em: {save_path}")


# ================================================================
# 6. EXECUÇÃO PRINCIPAL
# ================================================================

def main():
    print("=== Iniciando processamento de dados do robô ===")
    os.makedirs("data", exist_ok=True)

    # Cria log de exemplo se não existir
    if not os.path.exists(LOG_FILE):
        print(f"Arquivo {LOG_FILE} não encontrado.")
        return

    # Processamento Odometria
    df_raw = load_log_data(LOG_FILE)
    df_processed_odometry = compute_odometry(df_raw)
    df_processed_odometry.to_csv(PROCESSED_FILE_ODOMETRY, index=False)
    print(f"Dados processados salvos em: {PROCESSED_FILE_ODOMETRY}")
    plot_robot_trajectory(df_processed_odometry, TRAJECTORY_PLOT_ODOMETRY)

    df_processed_ins = compute_ins(df_raw)
    df_processed_ins.to_csv(PROCESSED_FILE_INS, index=False)
    plot_robot_trajectory(df_processed_ins, TRAJECTORY_PLOT_INS)
    # Gráfico
  

if __name__ == "__main__":
    main()
