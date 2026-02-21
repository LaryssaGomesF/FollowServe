import numpy as np
import pandas as pd
from ins import load_log_data as load_ins_data, convert_imu_readings, compute_ins_pure
from odometria import load_log_data as load_odom_data, compute_odometry_pure

def calcular_matrizes_R_cientifico(file_path):
    # 1. Carrega os dados brutos (log do robô parado)
    df_odom_raw = load_odom_data(file_path)
    df_ins_raw = load_ins_data(file_path)

    if df_odom_raw.empty or df_ins_raw.empty:
        print("Erro: Datasets vazios!")
        return

    # 2. Processa as trajetórias puras (converte bruto para x, y, theta)
    # Como o robô está parado, o ideal é que esses valores fossem constantes.
    # A oscilação que sobrar aqui é o ruído que o Kalman precisa conhecer.
    df_odom = compute_odometry_pure(df_odom_raw.copy())
    df_ins = convert_imu_readings(df_ins_raw.copy())
    df_ins = compute_ins_pure(df_ins.copy())

    # 3. Cálculo da Variância para Odometria
    # Usamos os nomes das colunas que suas funções geram (ajuste se necessário)
    var_x_odom = df_odom['x_odom'].var()
    var_y_odom = df_odom['y_odom'].var()
    var_theta_odom = df_odom['theta_odom'].var()

    # 4. Cálculo da Variância para INS
    var_x_ins = df_ins['x_ins'].var()
    var_y_ins = df_ins['y_ins'].var()
    var_theta_ins = df_ins['theta_ins'].var()

    # 5. Montagem das Matrizes R
    R_odom = np.diag([var_x_odom, var_y_odom, var_theta_odom])
    R_ins = np.diag([var_x_ins, var_y_ins, var_theta_ins])

    # Exibição dos resultados para copiar para o filtro principal
    print("="*50)
    print("MATRIZES R CALCULADAS (CIENTÍFICO - ROBÔ PARADO)")
    print("="*50)
    print("\n[R_odom] - Variância da Odometria:")
    print(R_odom)
    print(f"\nDiagonal para copiar: [{var_x_odom:.10f}, {var_y_odom:.10f}, {var_theta_odom:.10f}]")
    
    print("\n" + "-"*30)
    
    print("\n[R_ins] - Variância da INS:")
    print(R_ins)
    print(f"\nDiagonal para copiar: [{var_x_ins:.10f}, {var_y_ins:.10f}, {var_theta_ins:.10f}]")
    print("="*50)

    return R_odom, R_ins

# Caminho do seu arquivo de log com o robô parado
file_path = "./data/dados_mqtt.txt"
R_odom, R_ins = calcular_matrizes_R_cientifico(file_path)