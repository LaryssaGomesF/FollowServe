import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ins import load_log_data as load_ins_data, convert_imu_readings, compute_ins_pure
from odometria import load_log_data as load_odom_data, compute_odometry_pure

# ================================================================
# 1. O NÚCLEO DO FILTRO (EKF DE DUPLA ENTRADA)
# ================================================================
def ekf_fusion(state_k_prev, P_k_prev, z_odom, z_ins, delta_s):
    # --- 1. PREDIÇÃO ---
    theta_p = state_k_prev[2, 0]
    
    state_pred = np.array([
        [state_k_prev[0, 0] + delta_s * np.cos(theta_p)],
        [state_k_prev[1, 0] + delta_s * np.sin(theta_p)],
        [theta_p] 
    ])

    # Jacobiano F
    F = np.array([
        [1, 0, -delta_s * np.sin(theta_p)],
        [0, 1,  delta_s * np.cos(theta_p)],
        [0, 0, 1]
    ])


    Q = np.eye(3) * 0.01
    P_pred = F @ P_k_prev @ F.T + Q

    # --- 2. ATUALIZAÇÃO (FUSÃO) ---
    Z = np.vstack((z_odom, z_ins))
    H = np.vstack((np.eye(3), np.eye(3)))

    # R: Ajuste conforme a qualidade real dos seus sensores
    R_odom = np.diag([0.5,0.5, 0.3])   # Odom boa em posição, instável em ângulo
    R_ins  = np.diag([1.5, 1.5,0.1])  # INS deriva em posição, mas é precisa em ângulo
    R = np.block([
        [R_odom, np.zeros((3,3))],
        [np.zeros((3,3)), R_ins]
    ])

    y = Z - (H @ state_pred)


    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    state_new = state_pred + K @ y
    P_new = (np.eye(3) - K @ H) @ P_pred

    return state_new, P_new

# ================================================================
# 2. EXECUÇÃO
# ================================================================
def main():
    df_odom = load_odom_data("./data/dados_mqtt.txt")
    df_ins = load_ins_data("./data/dados_mqtt.txt")

    if df_odom.empty or df_ins.empty:
        print("Algum dos datasets está vazio!")
        return
    
    df_odom = compute_odometry_pure(df_odom.copy())
    df_ins = convert_imu_readings(df_ins.copy())
    df_ins = compute_ins_pure(df_ins.copy())

    # Sincronização: Interpola os dados da INS para os timestamps da Odometria
    # Agora interpolamos X, Y e Theta da INS para usar na medição
    x_ins_interp = np.interp(df_odom['timestamp'], df_ins['timestamp'], df_ins['x_ins'])
    y_ins_interp = np.interp(df_odom['timestamp'], df_ins['timestamp'], df_ins['y_ins'])
    theta_ins_interp = np.interp(df_odom['timestamp'], df_ins['timestamp'], df_ins['theta_ins'])
    
    state = np.array([[0.0], [0.0], [theta_ins_interp[0]]])
    P = np.eye(3)
    
    history = []
    # Adicionamos o estado inicial ao histórico
    history.append(state.flatten())

    for i in range(1, len(df_odom)):
        z_odom = np.array([
            [df_odom.at[i, 'x_odom']], 
            [df_odom.at[i, 'y_odom']], 
            [df_odom.at[i, 'theta_odom']]
        ])
        
        z_ins = np.array([
            [x_ins_interp[i]], 
            [y_ins_interp[i]], 
            [theta_ins_interp[i]]
        ])
        
        delta_s = df_odom.at[i, 'delta_S']

        state, P = ekf_fusion(state, P, z_odom, z_ins, delta_s)
        history.append(state.flatten())

    # CONVERSÃO PARA ARRAY (Corrige o erro do plot)
    history = np.array(history)

    plt.figure(figsize=(10, 6))
    plt.plot(df_odom['x_odom'], df_odom['y_odom'], label='Odometria Pura', alpha=0.4, linestyle='--')
    plt.plot(x_ins_interp, y_ins_interp, label='INS Pura', alpha=0.4, linestyle='--')
    plt.plot(history[:, 0], history[:, 1], label='Fusão EKF', color='red', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.title("Resultados")
    plt.show()

if __name__ == "__main__":
    main()