import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ins import load_log_data as load_ins_data, convert_imu_readings, compute_ins_pure
from odometria import load_log_data as load_odom_data, compute_odometry_pure

# ================================================================
# 1. ESTÁGIO 1: FILTRO DE ESTADO IMU (Leis de Newton)
# ================================================================
def imu_pre_filter(state_imu, P_imu, ax_global, ay_global, dt):
    """
    state_imu: [x, y, vx, vy]
    ax_global, ay_global: Já rotacionados pelo seu módulo INS
    """
    # Matriz de Transição de Estado (A) - Cinemática Linear
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Modelo de entrada (u) baseado nas acelerações globais
    # s = s0 + v*dt + 0.5*a*dt^2
    # v = v0 + a*dt
    B_u = np.array([
        [0.5 * dt**2 * ax_global],
        [0.5 * dt**2 * ay_global],
        [dt * ax_global],
        [dt * ay_global]
    ])

    # Predição do Estado
    state_pred = A @ state_imu + B_u
    
    # Q_imu: Ruído do processo (o quanto o modelo de Newton falha)
    # Aumente se o robô sofrer muitos impactos ou vibrações
    Q_imu = np.eye(4) * 0.01 
    P_pred = A @ P_imu @ A.T + Q_imu
    
    return state_pred, P_pred

# ================================================================
# 2. ESTÁGIO 2: EKF DE FUSÃO (ODOM + IMU FILTRADA)
# ================================================================
def ekf_fusion_cascade(state_k_prev, P_k_prev, z_odom, z_imu_filt, delta_s):
    # --- PREDIÇÃO (Modelo de Movimento Odométrico) ---
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

    # --- ATUALIZAÇÃO ---
    Z = np.vstack((z_odom, z_imu_filt))
    H = np.vstack((np.eye(3), np.eye(3)))

    # R: Confiança nos sensores
    R_odom = np.diag([0.5,0.5, 0.3])   # Odom boa em posição, instável em ângulo
    R_ins  = np.diag([1.0, 1.0,0.1])  # INS deriva em posição, mas é precisa em ângulo
    R = np.block([
        [R_odom, np.zeros((3,3))],
        [np.zeros((3,3)), R_ins]
    ])
    y = Z - (H @ state_pred)
    
    # Normalização de Ângulo
    y[2,0] = (y[2,0] + np.pi) % (2*np.pi) - np.pi
    y[5,0] = (y[5,0] + np.pi) % (2*np.pi) - np.pi

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    state_new = state_pred + K @ y
    P_new = (np.eye(3) - K @ H) @ P_pred

    return state_new, P_new

# ================================================================
# 3. LOOP PRINCIPAL
# ================================================================
def main():
    # Carregamento e Processamento Inicial
    df_odom = load_odom_data("./data/dados_mqtt.txt")
    df_ins = load_ins_data("./data/dados_mqtt.txt")
    
    df_odom = compute_odometry_pure(df_odom.copy())
    df_ins = convert_imu_readings(df_ins.copy())
    df_ins = compute_ins_pure(df_ins.copy())

    # Inicialização Filtro 1 (IMU)
    state_imu = np.zeros((4, 1)) # [x, y, vx, vy]
    P_imu = np.eye(4)
    
    # Inicialização Filtro 2 (Fusão)
    # Começa com o Theta inicial da INS
    theta_0 = df_ins['theta_ins'].iloc[0]
    state_final = np.array([[0.0], [0.0], [theta_0]])
    P_final = np.eye(3)
    
    history = []

    # Sincronização e Loop
    for i in range(1, len(df_odom)):
        # Tempo entre amostras (s)
        dt = (df_odom.at[i, 'timestamp'] - df_odom.at[i-1, 'timestamp']) * 1e-3
        
        # 1. Filtro de Estágio 1 (Pré-filtragem IMU usando acelerações globais da sua INS)
        # Note que usamos 'ax_global_ms2' que seu código já calculou
        ax_g = df_ins.at[i, 'ax_global_ms2'] / 1000.0 # Convertendo mm para m se necessário
        ay_g = df_ins.at[i, 'ay_global_ms2'] / 1000.0
        
        state_imu, P_imu = imu_pre_filter(state_imu, P_imu, ax_g, ay_g, dt)
        
        # 2. Medições para o Estágio 2
        z_odom = np.array([
            [df_odom.at[i, 'x_odom']], 
            [df_odom.at[i, 'y_odom']], 
            [df_odom.at[i, 'theta_odom']]
        ])
        
        # O dado da IMU agora vem do estado filtrado em mm
        z_imu_filt = np.array([
            [state_imu[0,0] * 1000.0], 
            [state_imu[1,0] * 1000.0], 
            [df_ins.at[i, 'theta_ins']]
        ])
        
        delta_s = df_odom.at[i, 'delta_S']

        # 3. Execução da Fusão em Cascata
        state_final, P_final = ekf_fusion_cascade(state_final, P_final, z_odom, z_imu_filt, delta_s)
        history.append(state_final.flatten())

    # Visualização
    history = np.array(history)
    plt.figure(figsize=(10,6))
    plt.plot(df_odom['x_odom'], df_odom['y_odom'], '--', label='Odometria Pura', alpha=0.4)
    plt.plot(df_ins['x_ins'], df_ins['y_ins'], '--', label='INS Pura', alpha=0.4)
    plt.plot(history[:, 0], history[:, 1], 'r', label='Fusão EKF Cascata', linewidth=2)
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.title("Resultado Final: Cascata EKF (INS Newton + Fusão Odom)")
    plt.show()

if __name__ == "__main__":
    main()