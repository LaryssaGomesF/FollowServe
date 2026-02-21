from odometria import load_log_data as load_odom_data, compute_odometry_pure
import numpy as np

def calcular_distancia_percorrida_odom():
    df_odom = load_odom_data("./data/dados_mqtt.txt")
 
    if df_odom.empty:
        print("Algum dos datasets está vazio!")
        return
    
    df_odom = compute_odometry_pure(df_odom.copy())
    
    # A distância total é a norma do vetor entre o ponto final e o ponto inicial
    x_final = df_odom['x_odom'].iloc[-1]
    y_final = df_odom['y_odom'].iloc[-1]
    x_ini = df_odom['x_odom'].iloc[0]
    y_ini = df_odom['y_odom'].iloc[0]
    
    dist_calculada = np.sqrt((x_final - x_ini)**2 + (y_final - y_ini)**2)
    
    # Também calculamos o ângulo final para o Q de orientação
    theta_final = df_odom['theta_odom'].iloc[-1]

    print(f"Distância calculada pela Odom: {dist_calculada:.4f} mm")
    print(f"Ângulo final (Odom): {theta_final:.4f} rad")
    return dist_calculada, theta_final

def main():
    calcular_distancia_percorrida_odom()

if __name__ == "__main__":
    main()