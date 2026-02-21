import numpy as np

# --- INSIRA SEUS DADOS AQUI ---
distancias_reais = [691, 692, 694, 700]  # O que você mediu na fita (ex: 5 metros)
distancias_odom  = [526.5803,  507.9921,  479.1750, 579.5027] # O que o código printou acima
erros_theta      = [ 1.9747, 2.4552, 2.5466, 2.2233] # O quanto ele entortou na reta (em radianos)

def calcular_matriz_Q(reais, odoms, thetas):
    reais = np.array(reais)
    odoms = np.array(odoms)
    thetas = np.array(thetas)
    
    # 1. Erro de Translação (X e Y)
    # Calculamos o erro por metro para poder escalar no Kalman depois
    erros_trans = reais - odoms
    var_trans_por_metro = np.var(erros_trans) / np.mean(reais)
    
    # 2. Erro de Rotação (Theta)
    var_theta_por_metro = np.var(thetas) / np.mean(reais)
    
    # Matriz Q base (Incerteza por metro percorrido)
    Q_base = np.diag([var_trans_por_metro, var_trans_por_metro, var_theta_por_metro])
    
    print("="*50)
    print("MATRIZ Q BASE (POR METRO PERCORRIDO)")
    print("="*50)
    print(Q_base)
    print("\nNo seu EKF, use:")
    print(f"Q = Q_base * delta_s")
    print("="*50)

calcular_matriz_Q(distancias_reais, distancias_odom, erros_theta)