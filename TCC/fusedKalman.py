# fused_kalman.py

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ins import load_log_data as load_ins_data, convert_imu_readings, compute_ins_pure
from odometria import load_log_data as load_odom_data, compute_odometry_pure

# ================================================================
# 1. CONFIGURAÇÃO DE PASTAS
# ================================================================
OUTPUT_DIR = "./data/fused_kalman"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 2. FUNÇÃO DE FUSÃO COM KALMAN
# ================================================================

def fuse_kalman(df_odom: pd.DataFrame, df_ins: pd.DataFrame):
    """
    Filtragem com Kalman simples:
    - Estado: x, y, theta
    - Medições: delta_S odometria, theta INS
    """
    n = len(df_odom)
    
    # Inicializa vetores de estado
    x_est = np.zeros(n)
    y_est = np.zeros(n)
    theta_est = np.zeros(n)
    
    # Inicializa covariâncias
    P = np.eye(3) * 1e-3  # pequena incerteza inicial
    Q = np.diag([0.01, 0.01, 0.001])  # ruído do processo
    R = np.diag([0.5, 0.5, 0.05])     # ruído da medição (x, y da odometria, theta INS)
    
    # Interpola theta INS para os timestamps da odometria
    theta_ins_interp = np.interp(df_odom['timestamp'], df_ins['timestamp'], df_ins['theta_ins'])
    
    for i in range(1, n):
        # --- PREDIÇÃO ---
        delta_S = df_odom.loc[i, 'delta_S']
        theta_prev = theta_est[i-1]
        
        # Modelo de movimento (odometria)
        x_pred = x_est[i-1] + delta_S * math.cos(theta_prev)
        y_pred = y_est[i-1] + delta_S * math.sin(theta_prev)
        theta_pred = theta_prev  # assumimos que a odometria não corrige theta
        
        # Jacobiano do modelo de movimento
        F = np.array([
            [1, 0, -delta_S * math.sin(theta_prev)],
            [0, 1,  delta_S * math.cos(theta_prev)],
            [0, 0, 1]
        ])
        
        # Covariância preditiva
        P = F @ P @ F.T + Q
        
        # --- ATUALIZAÇÃO ---
        # Medição: posição da odometria + theta do INS
        z = np.array([
            x_est[i-1] + delta_S * math.cos(theta_prev),
            y_est[i-1] + delta_S * math.sin(theta_prev),
            theta_ins_interp[i]
        ])
        
        H = np.eye(3)  # medição direta
        
        y_residual = z - np.array([x_pred, y_pred, theta_pred])
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        
        state_update = K @ y_residual
        x_est[i] = x_pred + state_update[0]
        y_est[i] = y_pred + state_update[1]
        theta_est[i] = theta_pred + state_update[2]
        
        # Covariância atualizada
        P = (np.eye(3) - K @ H) @ P
    
    df_fused = df_odom.copy()
    df_fused['x_kalman'] = x_est
    df_fused['y_kalman'] = y_est
    df_fused['theta_kalman'] = theta_est
    
    return df_fused

# ================================================================
# 3. PLOTAGEM
# ================================================================

def plot_kalman(df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(12,10))
   
    plt.plot(df['x_kalman'], df['y_kalman'], '-o', markersize=2, linewidth=1.5, label='Kalman Fusion Odometria+INS')
    plt.plot(df.loc[0, 'x_kalman'], df.loc[0, 'y_kalman'], 'go', markersize=8, label='Início')
    plt.plot(df.loc[len(df)-1, 'x_kalman'], df.loc[len(df)-1, 'y_kalman'], 'ro', markersize=8, label='Fim')
    
    plt.title('Mapeamento com Filtro de Kalman')
    plt.xlabel('Posição X (mm)')
    plt.ylabel('Posição Y (mm)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico salvo em: {save_path}")

# ================================================================
# 4. MAIN
# ================================================================

def main():
    df_odom = load_odom_data("./data/log.txt")
    df_ins = load_ins_data("./data/log.txt")

    if df_odom.empty or df_ins.empty:
        print("Algum dos datasets está vazio!")
        return
    
    # Processa odometria e INS
    df_odom = compute_odometry_pure(df_odom.copy())
    df_ins = convert_imu_readings(df_ins.copy())
    df_ins = compute_ins_pure(df_ins.copy())

    # Fusão com Kalman
    df_fused = fuse_kalman(df_odom, df_ins)

    # Salva resultados
    fused_file = os.path.join(OUTPUT_DIR, "fused_kalman.csv")
    df_fused.to_csv(fused_file, index=False)
    plot_kalman(df_fused, os.path.join(OUTPUT_DIR, "fused_kalman.png"))
    print(f"Dados de fusão Kalman salvos em: {fused_file}")

if __name__ == "__main__":
    main()
