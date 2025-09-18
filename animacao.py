import vedo
import numpy as np
import json
import os
import time

MODEL_PATH = "roboanimacao.stl"
ANGLE_FILE = "data/angle.json"

def euler_to_rotation_matrix(roll, pitch, yaw):
    r = np.deg2rad(roll)
    p = np.deg2rad(pitch)
    y = np.deg2rad(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])

    Ry = np.array([[np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])

    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return R

def ler_angulo(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Verifica se as chaves existem
        if all(k in data for k in ('x','y','z')):
            return data
        else:
            print("Arquivo JSON não tem as chaves x, y, z")
            return {'x':0, 'y':0, 'z':0}
    except Exception as e:
        print("Erro lendo ANGLE_FILE:", e)
        return {'x':0, 'y':0, 'z':0}

if not os.path.exists(MODEL_PATH):
    print(f"Modelo não encontrado: {MODEL_PATH}")
    exit(1)

mesh = vedo.Mesh(MODEL_PATH)
original_points = mesh.points  # sem parênteses

plt = vedo.Plotter(bg='black', axes=1)
plt.show(mesh, "Animação do Robô", interactive=False)

try:
    while True:
        angle = ler_angulo(ANGLE_FILE)
        R = euler_to_rotation_matrix(angle['x'], angle['y'], angle['z'])

        pts = np.array(original_points)
        pts_rot = pts @ R.T

        # Atualiza pontos da malha
        try:
            mesh.points(pts_rot, reset=False)
        except TypeError:
            mesh.points = pts_rot

        plt.render()

        time.sleep(0.05)  # 50 ms
except KeyboardInterrupt:
    print("Encerrando animação...")
    plt.close()
