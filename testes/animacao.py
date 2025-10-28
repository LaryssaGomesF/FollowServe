import vedo
import numpy as np
import json
import os
import time

MODEL_PATH = "roboanimacao.stl"
ANGLE_FILE = "data/angle.json"

def ler_angulo(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if all(k in data for k in ('x','y','z')):
            return data
        else:
            return {'x':0, 'y':0, 'z':0}
    except:
        return {'x':0, 'y':0, 'z':0}

if not os.path.exists(MODEL_PATH):
    print(f"Modelo não encontrado: {MODEL_PATH}")
    exit(1)

# Carrega o modelo STL
mesh = vedo.Mesh(MODEL_PATH)

# Configura a janela
plt = vedo.Plotter(bg='black', axes=1)
plt.show(mesh, "Animação do Robô", interactive=False)

try:
    last_angle = {'x':0, 'y':0, 'z':0}
    while True:
        angle = ler_angulo(ANGLE_FILE)

        # Diferença de ângulo em relação ao último (para evitar acúmulo de transformações)
        dx = angle['x'] - last_angle['x']
        dy = angle['y'] - last_angle['y']
        dz = angle['z'] - last_angle['z']

        # Rotaciona a malha incrementalmente (bem mais rápido)
        if dx: mesh.rotate(dx, axis=(1,0,0), rad=False)
        if dy: mesh.rotate(dy, axis=(0,1,0), rad=False)
        if dz: mesh.rotate(dz, axis=(0,0,1), rad=False)

        last_angle = angle

        plt.render()
        time.sleep(0.02)  # ~50 FPS
except KeyboardInterrupt:
    print("Encerrando animação...")
    plt.close()
