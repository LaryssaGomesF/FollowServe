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
WHEEL_DIAMETER_M = 0.022   # 22 mm
WHEEL_BASE_M = 0.120       # 120 mm


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
    
    # Usa a função zip para combinar as chaves (COLUMN_MAP) com os valores (reading)
    return dict(zip(COLUMN_MAP, reading))

def load_log_data(filepath: str) -> pd.DataFrame:
    """Carrega o arquivo de log (JSON por linha, onde cada linha é um array de leitura única) 
    e retorna um DataFrame ordenado por tempo."""
    if not os.path.exists(filepath):
        print(f"Arquivo de log não encontrado: {filepath}")
        return pd.DataFrame()

    data_list = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # 1. Deserializa a linha, que agora é um array de leitura única
                reading_array: List[Any] = json.loads(line.strip())
                
                # 2. Mapeia o array para um dicionário
                try:
                    reading_dict = map_reading_to_dict(reading_array)
                    data_list.append(reading_dict)
                except ValueError as e:
                    print(f"Erro ao mapear leitura: {e}. Leitura ignorada.")
                    continue
                        
            except json.JSONDecodeError:
                print(f"Linha inválida (não é um array JSON válido) ignorada: {line.strip()}")
                continue

    if not data_list:
        print("Nenhum dado válido encontrado no log.")
        return pd.DataFrame()

    # 3. Cria o DataFrame a partir da lista de dicionários
    df = pd.DataFrame(data_list)
    
    # 4. Processamento final (mantido do código original)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df['timestamp_s'] = df['timestamp'] / 1e6
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

    wheel_radius = WHEEL_DIAMETER_M / 2.0
    wheel_circumference = 2.0 * math.pi * wheel_radius

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
        dist_R = (delta_laps_R / GEAR_RATIO) * wheel_circumference
        dist_L = (delta_laps_L / GEAR_RATIO) * wheel_circumference

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

    vx = 0.0
    vy = 0.0
    new_x = new_y = 0.0
    theta = 0.0

    # Loop principal de integração
    for i in range(1, len(df)):
   
              
      
        dt =  df.loc[i, 'dt']
      
        current_gyro = df.loc[i, 'gyro_z_dps']
        gyro_z_rad = math.radians(current_gyro)
        # Estima a orientacao
        theta = gyro_z_rad*dt

        accel_x_local = df.loc[i, 'accel_x_ms2']
        accel_y_local = df.loc[i, 'accel_y_ms2']

        # Muda pra referencia global
        a_global_x = accel_x_local * math.cos(theta) - accel_y_local * math.sin(theta)
        a_global_y = accel_x_local * math.sin(theta) + accel_y_local * math.cos(theta)
        
        # Duas integracoes para calcular a posicao
        vx += a_global_x * dt
        vy += a_global_y * dt
        new_x += vx * dt
        new_y += vy * dt

        df.loc[i, 'x'] =  df.loc[i - 1, 'x'] + vx * dt
        df.loc[i, 'y'] =  df.loc[i - 1, 'y'] + vy * dt
        df.loc[i, 'theta'] = theta


    return df


# ================================================================
# 5. FUSÃO DE SENSORES (Encoders + Giroscópio)
# ================================================================
def compute_sensor_fusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estima a trajetória do robô combinando:
    - deslocamento linear pela odometria (encoders)
    - orientação (theta) pelo giroscópio
    """

    if df.empty:
        return df

    df = df.apply(convert_imu_readings, axis=1)

    df['x_fused'] = 0.0
    df['y_fused'] = 0.0
    df['theta_gyro'] = 0.0
    df['delta_theta_gyro'] = 0.0
    df['linear_displacement'] = 0.0

    wheel_radius = WHEEL_DIAMETER_M / 2.0
    wheel_circumference = 2.0 * math.pi * wheel_radius

    prev_encoder_R = df.loc[0, 'encoderR']
    prev_encoder_L = df.loc[0, 'encoderL']

    x = y = theta = 0.0

    for i in range(1, len(df)):
        # --- 1. ODOMETRIA (deslocamento linear) ---
        current_encoder_R = df.loc[i, 'encoderR']
        current_encoder_L = df.loc[i, 'encoderL']

        delta_laps_R = current_encoder_R - prev_encoder_R
        delta_laps_L = current_encoder_L - prev_encoder_L

        prev_encoder_R = current_encoder_R
        prev_encoder_L = current_encoder_L

        dist_R = (delta_laps_R / GEAR_RATIO) * wheel_circumference
        dist_L = (delta_laps_L / GEAR_RATIO) * wheel_circumference
        delta_S = (dist_R + dist_L) / 2.0
        df.loc[i, 'linear_displacement'] = delta_S

        # --- 2. ORIENTAÇÃO PELO GIROSCÓPIO ---
        dt = df.loc[i, 'dt']  # intervalo entre amostras (s)
        gyro_z = df.loc[i, 'gyro_z_dps']
        delta_theta_gyro = math.radians(gyro_z) * dt
        theta += delta_theta_gyro

        df.loc[i, 'theta_gyro'] = theta
        df.loc[i, 'delta_theta_gyro'] = delta_theta_gyro

        # --- 3. ATUALIZA POSE ---
        x += delta_S * math.cos(theta)
        y += delta_S * math.sin(theta)

        df.loc[i, 'x_fused'] = x
        df.loc[i, 'y_fused'] = y

    return df


# ================================================================
# 5. VISUALIZAÇÃO
# ================================================================

def plot_robot_trajectory(df: pd.DataFrame, save_path: str = TRAJECTORY_PLOT):
    """Gera e salva o gráfico da trajetória (x, y) com setas orientadas por theta."""
    if df.empty or not all(col in df.columns for col in ['x', 'y', 'theta']):
        print("Sem dados suficientes para plotar (x, y, theta necessários).")
        return

    plt.figure(figsize=(10, 8))

    # Linha contínua + pontos
    plt.plot(df['x'], df['y'], '-o', markersize=3, linewidth=1.2, label='Trajetória', color='tab:blue')

    # --- Parâmetros das setas ---
    step = max(1, len(df) )  # mostra cerca de 40 setas
    scale = max(df['x'].max() - df['x'].min(), df['y'].max() - df['y'].min())
    arrow_length = 0.05 * scale   # comprimento relativo (~5% do tamanho total)
    head_width = 0.015 * scale
    head_length = 0.03 * scale

    # --- Plota as setas conforme o ângulo theta ---
    for i in range(0, len(df), step):
        x = df['x'].iloc[i]
        y = df['y'].iloc[i]
        theta = df['theta'].iloc[i]
        dx = arrow_length * math.cos(theta)
        dy = arrow_length * math.sin(theta)

        plt.arrow(x, y, dx, dy,
                  shape='full', lw=0, length_includes_head=True,
                  head_width=head_width, head_length=head_length,
                  color='orange', alpha=0.7)

    # Marca início e fim
    plt.scatter(df['x'].iloc[0], df['y'].iloc[0], c='green', s=50, label='Início')
    plt.scatter(df['x'].iloc[-1], df['y'].iloc[-1], c='red', s=50, label='Fim')

    plt.xlabel('Posição X (m)')
    plt.ylabel('Posição Y (m)')
    plt.title('Trajetória Estimada do Robô (orientação pelo θ)')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Gráfico salvo em: {save_path}")

# ================================================================
# 5. VISUALIZAÇÃO (Comparação lado a lado)
# ================================================================

def plot_all_trajectories(df_odom: pd.DataFrame, df_ins: pd.DataFrame, df_fusion: pd.DataFrame, save_path: str):
    """Plota as três trajetórias (Odômetro, INS e Fusão) lado a lado."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ['Odometria (Encoders)', 'Navegação Inercial (IMU)', 'Fusão Encoders + Giroscópio']
    datasets = [
        (df_odom, 'x', 'y', 'theta'),
        (df_ins, 'x', 'y', 'theta'),
        (df_fusion, 'x_fused', 'y_fused', 'theta_gyro')
    ]

    for ax, (df, x_col, y_col, theta_col), title in zip(axes, datasets, titles):
        if df.empty or not all(col in df.columns for col in [x_col, y_col, theta_col]):
            ax.text(0.5, 0.5, 'Dados insuficientes', ha='center', va='center', fontsize=10)
            ax.set_title(title)
            ax.axis('off')
            continue

        ax.plot(df[x_col], df[y_col], '-o', markersize=3, linewidth=1.2, color='tab:blue', label='Trajetória')

        # --- Parâmetros das setas ---
        step = max(1, len(df) // 40)
        scale = max(df[x_col].max() - df[x_col].min(), df[y_col].max() - df[y_col].min())
        arrow_length = 0.05 * scale
        head_width = 0.015 * scale
        head_length = 0.03 * scale

        # --- Plota setas de orientação ---
        for i in range(0, len(df), step):
            x = df[x_col].iloc[i]
            y = df[y_col].iloc[i]
            theta = df[theta_col].iloc[i]
            dx = arrow_length * math.cos(theta)
            dy = arrow_length * math.sin(theta)
            ax.arrow(x, y, dx, dy, shape='full', lw=0,
                     length_includes_head=True, head_width=head_width,
                     head_length=head_length, color='orange', alpha=0.7)

        ax.scatter(df[x_col].iloc[0], df[y_col].iloc[0], c='green', s=50, label='Início')
        ax.scatter(df[x_col].iloc[-1], df[y_col].iloc[-1], c='red', s=50, label='Fim')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Gráfico comparativo salvo em: {save_path}")

# ================================================================
# 6. EXECUÇÃO PRINCIPAL
# ================================================================

def main():
    print("=== Iniciando processamento de dados do robô ===")
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(LOG_FILE):
        print(f"Arquivo {LOG_FILE} não encontrado.")
        return

    # --- Processa ODOMETRIA ---
    df_raw = load_log_data(LOG_FILE)
    df_processed_odometry = compute_odometry(df_raw)
    df_processed_odometry.to_csv(PROCESSED_FILE_ODOMETRY, index=False)

    # --- Processa INS ---
    df_processed_ins = compute_ins(df_raw)
    df_processed_ins.to_csv(PROCESSED_FILE_INS, index=False)

    # --- Processa FUSÃO ---
    df_processed_fusion = compute_sensor_fusion(df_raw)
    df_processed_fusion.to_csv("data/processed_data_fusion.csv", index=False)

    # --- Plota os três resultados lado a lado ---
    plot_all_trajectories(
        df_processed_odometry,
        df_processed_ins,
        df_processed_fusion,
        "data/trajectory_comparison.png"
    )
  

if __name__ == "__main__":
    main()
