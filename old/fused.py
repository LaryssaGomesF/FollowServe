# fused_mapping.py

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ins import load_log_data as load_ins_data, convert_imu_readings
from odometria import load_log_data as load_odom_data, compute_odometry_pure

# ================================================================
# 1. PASTAS DE SAÍDA
# ================================================================

OUTPUT_DIR = "./data/fused"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 2. FUSÃO ODOMETRIA + INS
# ================================================================

def fuse_odometry_ins(df_odom: pd.DataFrame, df_ins: pd.DataFrame) -> pd.DataFrame:
    """
    Combina odometria e INS:
    - Mantém cálculo de posição da odometria
    - Corrige ângulo usando INS (theta_ins)
    """
    df_fused = df_odom.copy()
    
    # Interpolando theta_ins para os timestamps da odometria
    theta_ins_interp = np.interp(
        df_odom['timestamp'].to_numpy(),
        df_ins['timestamp'].to_numpy(),
        df_ins['theta_ins'].to_numpy()
    )

    df_fused['theta_fused'] = theta_ins_interp

    # Recalcula a posição com o ângulo corrigido
    for i in range(1, len(df_fused)):
        prev_x = df_fused.loc[i-1, 'x_odom']
        prev_y = df_fused.loc[i-1, 'y_odom']
        delta_S = df_fused.loc[i, 'delta_S']
        theta_avg = (df_fused.loc[i-1, 'theta_fused'] + df_fused.loc[i, 'theta_fused']) / 2.0

        new_x = prev_x + delta_S * math.cos(theta_avg)
        new_y = prev_y + delta_S * math.sin(theta_avg)

        df_fused.loc[i, 'x_fused'] = new_x
        df_fused.loc[i, 'y_fused'] = new_y

    # Primeira posição
    df_fused.loc[0, 'x_fused'] = df_fused.loc[0, 'x_odom']
    df_fused.loc[0, 'y_fused'] = df_fused.loc[0, 'y_odom']

    return df_fused

# ================================================================
# 3. PLOTAGEM
# ================================================================

def plot_fused(df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(12, 10))

    plt.plot(df['x_fused'], df['y_fused'], '-o', markersize=2, linewidth=1.5, label='Fusão Odometria+INS')
    plt.plot(df.loc[0, 'x_fused'], df.loc[0, 'y_fused'], 'go', markersize=8, label='Início')
    plt.plot(df.loc[len(df)-1, 'x_fused'], df.loc[len(df)-1, 'y_fused'], 'ro', markersize=8, label='Fim')

    plt.title('Mapeamento Fusão Odometria + INS')
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

    df_odom = compute_odometry_pure(df_odom.copy())
    df_ins = convert_imu_readings(df_ins.copy())
    # Recalcular INS pura
    from ins import compute_ins_pure
    df_ins = compute_ins_pure(df_ins.copy())

    df_fused = fuse_odometry_ins(df_odom, df_ins)

    fused_file = os.path.join(OUTPUT_DIR, "fused_mapping.csv")
    df_fused.to_csv(fused_file, index=False)
    plot_fused(df_fused, os.path.join(OUTPUT_DIR, "fused_mapping.png"))
    print(f"Dados de fusão salvos em: {fused_file}")

if __name__ == "__main__":
    main()
