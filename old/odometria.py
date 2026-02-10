import json
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# ================================================================
# 1. PARÂMETROS FÍSICOS E DE CALIBRAÇÃO
# ================================================================

# --- Geometria do robô ---
WHEEL_DIAMETER_MM = 22.0
WHEEL_RADIUS_MM = WHEEL_DIAMETER_MM / 2.0
WHEEL_BASE_MM = 145.0

# --- Transmissão ---
# O GEAR_RATIO é necessário, pois o encoder está no motor.
GEAR_RATIO = 1.538 
WHEEL_CIRCUMFERENCE_MM = 2.0 * math.pi * WHEEL_RADIUS_MM

# --- Caminhos de arquivos ---
LOG_FILE = "./data/dados_mqtt.txt"
OUTPUT_DIR = "./data/odometria"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 2. LEITURA E PROCESSAMENTO DO ARQUIVO DE LOG
# ================================================================

COLUMN_MAP = [
    'timestamp',
    'encoderR', 'encoderL'
]

def map_reading_to_dict(reading: List[Any]) -> Dict[str, Any]:
    if len(reading) < 9:  # garante que existam todos os campos
        raise ValueError("Leitura menor que o esperado.")

    return {
        'timestamp': reading[0],
        'encoderR': reading[-2],  
        'encoderL': reading[-1],  
    }


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

    return df

# ================================================================
# 3. CÁLCULO DE ODOMETRIA
# ================================================================

def compute_odometry_pure(df: pd.DataFrame) -> pd.DataFrame:
    df_odom = df.copy()
    df_odom['x_odom'] = 0.0
    df_odom['y_odom'] = 0.0
    df_odom['theta_odom'] = 0.0
    df_odom['delta_S'] = 0.0   
    df_odom['dist_R'] = 0.0
    df_odom['dist_L'] = 0.0

    # Começamos o loop do zero ou de 1, dependendo se a primeira linha já contém movimento
    for i in range(1, len(df_odom)):
        # Pega cálculos de posição anteriores
        prev_theta = df_odom.loc[i-1, 'theta_odom']
        prev_x = df_odom.loc[i-1, 'x_odom']
        prev_y = df_odom.loc[i-1, 'y_odom']
        
        # Agora 'encoderR' e 'encoderL' já são o DELTA (variação) em graus
        delta_angle_degrees_R = df_odom.loc[i, 'encoderR']
        delta_angle_degrees_L = df_odom.loc[i, 'encoderL']

        # Conversão da variação do angulo de graus do MOTOR para rad da RODA
        # delta_rad = (graus_motor / gear_ratio) * (pi / 180)
        delta_angle_rad_R = math.radians(delta_angle_degrees_R / GEAR_RATIO)
        delta_angle_rad_L = math.radians(delta_angle_degrees_L / GEAR_RATIO)

        # Cálculo da distância linear percorrida por cada roda
        dist_R = delta_angle_rad_R * WHEEL_RADIUS_MM 
        dist_L = delta_angle_rad_L * WHEEL_RADIUS_MM      
        
        # Odometria diferencial clássica
        delta_S = (dist_R + dist_L) / 2.0
        delta_theta = (dist_R - dist_L) / WHEEL_BASE_MM
        
        new_theta = prev_theta + delta_theta
        # Usamos a média do ângulo para uma integração mais suave (Runge-Kutta de 2ª ordem)
        theta_avg = prev_theta + (delta_theta / 2.0)

        # Atualização das coordenadas cartesianas
        new_x = prev_x + delta_S * math.cos(theta_avg)
        new_y = prev_y + delta_S * math.sin(theta_avg)

        # Armazenando resultados
        df_odom.loc[i, 'x_odom'] = new_x
        df_odom.loc[i, 'y_odom'] = new_y
        df_odom.loc[i, 'theta_odom'] = new_theta
        df_odom.loc[i, 'delta_S'] = delta_S
        df_odom.loc[i, 'dist_R'] = dist_R
        df_odom.loc[i, 'dist_L'] = dist_L
        
    return df_odom

def plot_trajectories(df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(12, 10))

    if 'x_odom' in df.columns:
        plt.plot(df['x_odom'], df['y_odom'], '-o',
                 markersize=2, linewidth=1.5,
                 label='1. Odometria Pura (Encoders)')

        plt.plot(df.loc[0, 'x_odom'], df.loc[0, 'y_odom'],
                 'go', markersize=8, label='Início')
        plt.plot(df.loc[len(df)-1, 'x_odom'], df.loc[len(df)-1, 'y_odom'],
                 'ro', markersize=8, label='Fim')

    plt.title('Odometria')
    plt.xlabel('Posição X (mm)')
    plt.ylabel('Posição Y (mm)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # 🔑 fixa o (0,0) como origem visual
    ax = plt.gca()
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.savefig(save_path)
    plt.close()

    print(f"Gráfico salvo em: {save_path}")


# ================================================================
# 4. MAIN
# ================================================================

def main():
    df = load_log_data(LOG_FILE)
    if df.empty:
        return

    # Salvar dados convertidos e formatados (somente encoders e timestamps)
    formatted_file = os.path.join(OUTPUT_DIR, "formatted_encoders.csv")
    df.to_csv(formatted_file, index=False)
    print(f"Dados convertidos e formatados salvos em: {formatted_file}")

    #  Calcular odometria
    df_odom = compute_odometry_pure(df.copy())
  
    odom_file = os.path.join(OUTPUT_DIR, "odometry_mm.csv")
    df_odom.to_csv(odom_file, index=False)
    plot_trajectories(df_odom, "./data/odometria/odometry_image")
    print(f"Dados de odometria salvos em: {odom_file}")

if __name__ == "__main__":
    main()
