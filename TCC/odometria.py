import json
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  
from typing import List, Dict, Any
import numpy as np
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
        'encoderR': reading[-2],  # penúltima posição (Ângulo Acumulado R em Graus)
        'encoderL': reading[-1],  # última posição (Ângulo Acumulado L em Graus)
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

def compute_odometry_pure(df: pd.DataFrame, window_size: int = 10, alpha: float = 0.3) -> pd.DataFrame:
    df_odom = df.copy()
    df_odom['x_odom'] = 0.0
    df_odom['y_odom'] = 0.0
    df_odom['theta_odom'] = 0.0
    df_odom['delta_S'] = 0.0   
    df_odom['dist_R'] = 0.0
    df_odom['dist_L'] = 0.0

    # Inicializa com o primeiro valor de ângulo acumulado
    prev_accumulated_degrees_R = df_odom.loc[0, 'encoderR']
    prev_accumulated_degrees_L = df_odom.loc[0, 'encoderL']
 
    for i in range(1, len(df_odom)):
        # Pega calculos anteriores
        prev_theta = df_odom.loc[i-1, 'theta_odom']
        prev_x = df_odom.loc[i-1, 'x_odom']
        prev_y = df_odom.loc[i-1, 'y_odom']
        
        # Angulo em graus acumulado total neste instante de tempo
        curr_accumulated_degrees_R = df_odom.loc[i, 'encoderR']
        curr_accumulated_degrees_L = df_odom.loc[i, 'encoderL']

        # Variação do angulo em graus do ultimo momento para o atual
        # Ordem correta: delta = atual - anterior
        delta_angle_degrees_R = curr_accumulated_degrees_R - prev_accumulated_degrees_R
        delta_angle_degrees_L = curr_accumulated_degrees_L - prev_accumulated_degrees_L

        # Atualizar o acumulado anterior
        prev_accumulated_degrees_R, prev_accumulated_degrees_L = curr_accumulated_degrees_R, curr_accumulated_degrees_L

        # Conversao da variação do angulo de graus do MOTOR para rad da RODA
        # Divisão pelo GEAR_RATIO para obter o ângulo da roda
        delta_angle_rad_R = math.radians(delta_angle_degrees_R / GEAR_RATIO)
        delta_angle_rad_L = math.radians(delta_angle_degrees_L / GEAR_RATIO)

        # Calculo da distancia
        # Distância = Ângulo (rad) * Raio da Roda (mm)
        dist_R = delta_angle_rad_R * WHEEL_RADIUS_MM 
        dist_L = delta_angle_rad_L * WHEEL_RADIUS_MM      
        
        # O restante do cálculo da odometria diferencial
        delta_S = (dist_R + dist_L) / 2.0
        delta_theta = (dist_R - dist_L) / WHEEL_BASE_MM
        
        new_theta = prev_theta + delta_theta
        theta_avg = prev_theta + delta_theta / 2.0

        # Correção de convenção (cos para X, sin para Y)
        new_x = prev_x + delta_S * math.cos(theta_avg)
        new_y = prev_y + delta_S * math.sin(theta_avg)

        df_odom.loc[i, 'x_odom'] = new_x
        df_odom.loc[i, 'y_odom'] = new_y
        df_odom.loc[i, 'theta_odom'] = new_theta
        df_odom.loc[i, 'delta_S'] = delta_S
        df_odom.loc[i, 'dist_R'] = dist_R
        df_odom.loc[i, 'dist_L'] = dist_L
        
    return df_odom

def plot_trajectories(df: pd.DataFrame, save_path: str):
    """Gera o gráfico de Odometria com quadrados de 50mm de tamanho real fixo."""
    if 'x_odom' not in df.columns:
        return

    # Cálculo dos limites arredondados para múltiplos de 50
    margin = 50
    x_min, x_max = np.floor((df['x_odom'].min() - margin) / 50) * 50, np.ceil((df['x_odom'].max() + margin) / 50) * 50
    y_min, y_max = np.floor((df['y_odom'].min() - margin) / 50) * 50, np.ceil((df['y_odom'].max() + margin) / 50) * 50

    # Define quantos 'inches' cada 50mm vai ter no arquivo final
    # 0.5 inches para cada 50 unidades garante que o quadrado tenha sempre o mesmo tamanho
    res_scale = 0.5 / 50 
    fig_w = (x_max - x_min) * res_scale
    fig_h = (y_max - y_min) * res_scale

    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    plt.plot(df['x_odom'], df['y_odom'], '-o', markersize=2, linewidth=1.5, label='1. Odometria Pura (Encoders)')
    plt.plot(df.loc[0, 'x_odom'], df.loc[0, 'y_odom'], 'go', markersize=8, label='Início')
    plt.plot(df['x_odom'].iloc[-1], df['y_odom'].iloc[-1], 'ro', markersize=8, label='Fim')

    # Ajuste de Escala e Grid
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.title('Odometria')
    plt.xlabel('Posição X (mm)')
    plt.ylabel('Posição Y (mm)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Gráfico Odometria salvo: {save_path}")

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
