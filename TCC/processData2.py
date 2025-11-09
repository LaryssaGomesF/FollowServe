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
WHEEL_DIAMETER_M = 0.022  # 22 mm
WHEEL_RADIUS_M = WHEEL_DIAMETER_M / 2.0
WHEEL_BASE_M = 0.120       # 120 mm


# --- Transmissão ---
GEAR_RATIO = 1.538  # Relação de redução (motor : roda)
WHEEL_CIRCUMFERENCE = 2.0 * math.pi * WHEEL_RADIUS_M

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
                # O log.txt contém arrays JSON por linha
                reading_array: List[Any] = json.loads(line.strip())
                
                # Mapeia o array para um dicionário
                try:
                    reading_dict = map_reading_to_dict(reading_array)
                    data_list.append(reading_dict)
                except ValueError as e:
                    # print(f"Erro ao mapear leitura: {e}. Leitura ignorada.")
                    continue
                        
            except json.JSONDecodeError:
                # print(f"Linha inválida (não é um array JSON válido) ignorada: {line.strip()}")
                continue

    if not data_list:
        print("Nenhum dado válido encontrado no log.")
        return pd.DataFrame()

    df = pd.DataFrame(data_list)
    
    # Processamento de tempo
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    # Converte o timestamp de microssegundos para segundos
    df['timestamp_s'] = df['timestamp'] / 1e6
    # Calcula o intervalo de tempo (dt) entre as amostras
    df['dt'] = df['timestamp_s'].diff().fillna(0)
    
    # Converte leituras do IMU para unidades físicas
    df = convert_imu_readings(df)

    return df

# ================================================================
# 4. ABORDAGENS DE ODOMETRIA
# ================================================================

def compute_odometry_pure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Abordagem 1: Odometria Pura (Encoders)
    Usa encoders para deslocamento linear (delta_S) e angular (delta_theta).
    """
    df_odom = df.copy()
    
    df_odom['x_odom'] = 0.0
    df_odom['y_odom'] = 0.0
    df_odom['theta_odom'] = 0.0

    prev_accumulated_laps_R = df_odom.loc[0, 'encoderR']
    prev_accumulated_laps_L = df_odom.loc[0, 'encoderL']
    
    # O loop começa de 1, pois a linha 0 é o estado inicial (0, 0, 0)
    for i in range(1, len(df_odom)):
        prev_x = df_odom.loc[i - 1, 'x_odom']
        prev_y = df_odom.loc[i - 1, 'y_odom']
        prev_theta = df_odom.loc[i - 1, 'theta_odom']

        # 1. Variação das rotações (em voltas)
        current_acummulated_laps_R = df_odom.loc[i, 'encoderR'] 
        current_acummulated_laps_L = df_odom.loc[i, 'encoderL'] 

        delta_laps_R = current_acummulated_laps_R - prev_accumulated_laps_R
        delta_laps_L = current_acummulated_laps_L - prev_accumulated_laps_L

        # Atualiza estados anteriores
        prev_accumulated_laps_R = current_acummulated_laps_R
        prev_accumulated_laps_L = current_acummulated_laps_L

        # --- CORREÇÃO PARA PEQUENOS DELTA_LAPS (Ruído/Quantização) ---
        # Se a variação for muito pequena, assume-se que o robô não se moveu.
        # O valor 0.0001 é um limiar arbitrário em voltas, pode precisar de ajuste.
        THRESHOLD_LAPS = 0.25
        
        if abs(delta_laps_R) < THRESHOLD_LAPS and abs(delta_laps_L) < THRESHOLD_LAPS:
            dist_R = 0.0
            dist_L = 0.0
        else:
            # 2. Distância percorrida por cada roda (em metros)
            # delta_laps / GEAR_RATIO = voltas da roda
            dist_R = (delta_laps_R / GEAR_RATIO) * WHEEL_CIRCUMFERENCE
            dist_L = (delta_laps_L / GEAR_RATIO) * WHEEL_CIRCUMFERENCE

        # 3. Deslocamento médio e variação angular
        delta_S = (dist_R + dist_L) / 2.0
        delta_theta_odom = (dist_R - dist_L) / WHEEL_BASE_M

        # 4. Atualiza pose (Método de Euler aprimorado/aproximação de arco)
        # Atualiza a orientação (theta)
        new_theta = prev_theta + delta_theta_odom
        
        # Usa a média angular para o deslocamento (Euler aprimorado)
        theta_avg = prev_theta + (delta_theta_odom / 2.0)

        # Atualiza a posição (x, y)
        new_x = prev_x + delta_S * math.cos(theta_avg)
        new_y = prev_y + delta_S * math.sin(theta_avg)

        df_odom.loc[i, 'x_odom'] = new_x
        df_odom.loc[i, 'y_odom'] = new_y
        df_odom.loc[i, 'theta_odom'] = new_theta
        
    return df_odom

def compute_ins_pure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Abordagem 2: INS Pura (Acelerômetro e Giroscópio)
    Usa giroscópio para orientação (theta) e acelerômetros para velocidade e posição.
    """
    df_ins = df.copy()
    
    df_ins['x_ins'] = 0.0
    df_ins['y_ins'] = 0.0
    df_ins['theta_ins'] = 0.0

    vx = 0.0
    vy = 0.0
    theta = 0.0
    
    # O loop começa de 1, pois a linha 0 é o estado inicial (0, 0, 0)
    for i in range(1, len(df_ins)):
        dt = df_ins.loc[i, 'dt']
        
        # 1. Orientação (Giroscópio)
        # O giroscópio mede a taxa de rotação em dps (graus por segundo)
        gyro_z_dps = df_ins.loc[i, 'gyro_z_dps']
        gyro_z_rad = math.radians(gyro_z_dps)
        
        # Integração do giroscópio para obter o ângulo (theta)
        theta += gyro_z_rad * dt
        
        # 2. Aceleração (Acelerômetro)
        # Aceleração no referencial do robô (local)
        accel_x_local = df_ins.loc[i, 'accel_x_ms2']
        accel_y_local = df_ins.loc[i, 'accel_y_ms2']
        
        # Transforma a aceleração para o referencial global (eixo X e Y fixos)
        # Aceleração global = R(theta) * Aceleração local
        a_global_x = accel_x_local * math.cos(theta) - accel_y_local * math.sin(theta)
        a_global_y = accel_x_local * math.sin(theta) + accel_y_local * math.cos(theta)
        
        # 3. Integração para Velocidade (v = v0 + a*dt)
        vx += a_global_x * dt
        vy += a_global_y * dt
        
        # 4. Integração para Posição (p = p0 + v*dt)
        df_ins.loc[i, 'x_ins'] = df_ins.loc[i - 1, 'x_ins'] + vx * dt
        df_ins.loc[i, 'y_ins'] = df_ins.loc[i - 1, 'y_ins'] + vy * dt
        df_ins.loc[i, 'theta_ins'] = theta
        
    return df_ins

def compute_odometry_fused(df: pd.DataFrame) -> pd.DataFrame:
    """
    Abordagem 3: Odometria Fusionada (Encoders + Giroscópio)
    Usa encoders para deslocamento linear (delta_S) e giroscópio para angular (delta_theta).
    """
    df_fused = df.copy()
    
    df_fused['x_fused'] = 0.0
    df_fused['y_fused'] = 0.0
    df_fused['theta_fused'] = 0.0

    prev_encoder_R = df_fused.loc[0, 'encoderR']
    prev_encoder_L = df_fused.loc[0, 'encoderL']
    theta = 0.0
    
    # O loop começa de 1, pois a linha 0 é o estado inicial (0, 0, 0)
    for i in range(1, len(df_fused)):
        prev_x = df_fused.loc[i - 1, 'x_fused']
        prev_y = df_fused.loc[i - 1, 'y_fused']
        
        dt = df_fused.loc[i, 'dt']
        
        # --- 1. Deslocamento Linear (Encoders) ---
        current_encoder_R = df_fused.loc[i, 'encoderR']
        current_encoder_L = df_fused.loc[i, 'encoderL']

        delta_laps_R = current_encoder_R - prev_encoder_R
        delta_laps_L = current_encoder_L - prev_encoder_L

        prev_encoder_R = current_encoder_R
        prev_encoder_L = current_encoder_L

        THRESHOLD_LAPS = 0.695
        
        if abs(delta_laps_R) < THRESHOLD_LAPS and abs(delta_laps_L) < THRESHOLD_LAPS:
            dist_R = 0.0
            dist_L = 0.0
        else:
            # 2. Distância percorrida por cada roda (em metros)
            # delta_laps / GEAR_RATIO = voltas da roda
            dist_R = (delta_laps_R / GEAR_RATIO) * WHEEL_CIRCUMFERENCE
            dist_L = (delta_laps_L / GEAR_RATIO) * WHEEL_CIRCUMFERENCE

        delta_S = (dist_R + dist_L) / 2.0
        
        # --- 2. Orientação Angular (Giroscópio) ---
        gyro_z_dps = df_fused.loc[i, 'gyro_z_dps']
        gyro_z_rad = math.radians(gyro_z_dps)
        
        # Integração do giroscópio para obter o ângulo (theta)
        delta_theta_gyro = gyro_z_rad * dt
        theta += delta_theta_gyro
        
        # 3. Atualiza Pose (Método de Euler simples, pois delta_theta é pequeno)
        # Usa o theta atualizado (do giroscópio) para o deslocamento
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
    """Gera e salva o gráfico das três trajetórias (x, y)."""
    
    plt.figure(figsize=(12, 10))
    
    # 1. Odometria Pura
    if 'x_odom' in df.columns:
        plt.plot(df['x_odom'], df['y_odom'], '-o', markersize=2, linewidth=1.5, label='1. Odometria Pura (Encoders)', color='tab:blue')
        plt.plot(df.loc[0, 'x_odom'], df.loc[0, 'y_odom'], 'go', markersize=8, label='Início', color='green')
        plt.plot(df.loc[len(df)-1, 'x_odom'], df.loc[len(df)-1, 'y_odom'], 'ro', markersize=8, label='Fim', color='red')
        
    # 2. INS Pura
    if 'x_ins' in df.columns:
        plt.plot(df['x_ins'], df['y_ins'], '-o', markersize=2, linewidth=1.5, label='2. INS Pura (IMU)', color='tab:orange')
        
    # 3. Odometria Fusionada
    if 'x_fused' in df.columns:
        plt.plot(df['x_fused'], df['y_fused'], '-o', markersize=2, linewidth=1.5, label='3. Odometria + Giroscópio', color='tab:green')
        
    plt.title('Comparação das Trajetórias Estimadas')
    plt.xlabel('Posição X (m)')
    plt.ylabel('Posição Y (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Garante que a escala X e Y seja a mesma
    
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico salvo em: {save_path}")

def main():
    # 1. Carrega e pré-processa os dados
    df = load_log_data(LOG_FILE)
    if df.empty:
        return

    # 2. Calcula as três abordagens
    df_odom = compute_odometry_pure(df.copy())
    df_ins = compute_ins_pure(df.copy())
    df_fused = compute_odometry_fused(df.copy())
    
    # 3. Combina os resultados para visualização
    df_results = df[['timestamp_s', 'dt']].copy()
    
    # Adiciona colunas de odometria pura
    df_results['x_odom'] = df_odom['x_odom']
    df_results['y_odom'] = df_odom['y_odom']
    df_results['theta_odom'] = df_odom['theta_odom']
    
    # Adiciona colunas de INS pura
    df_results['x_ins'] = df_ins['x_ins']
    df_results['y_ins'] = df_ins['y_ins']
    df_results['theta_ins'] = df_ins['theta_ins']
    
    # Adiciona colunas de odometria fusionada
    df_results['x_fused'] = df_fused['x_fused']
    df_results['y_fused'] = df_fused['y_fused']
    df_results['theta_fused'] = df_fused['theta_fused']
    
    # 4. Salva os dados processados
    processed_file = os.path.join(OUTPUT_DIR, "processed_trajectories.csv")
    df_results.to_csv(processed_file, index=False)
    print(f"Dados processados salvos em: {processed_file}")
    
    # 5. Visualiza as trajetórias
    plot_path = os.path.join(OUTPUT_DIR, "trajectory_comparison.png")
    plot_trajectories(df_results, plot_path)

if __name__ == "__main__":
    main()
