# servidor_final_completo.py
import asyncio
import json
import os
from websockets.asyncio.server import serve

# --- ARQUIVOS DE SAÍDA ---
os.makedirs("data", exist_ok=True)
# 1. SEU ARQUIVO DE LOG ORIGINAL (MANTIDO)
log_file = "data/rotacao.txt"
# 2. O NOVO ARQUIVO PARA O VISUALIZADOR (ADICIONADO)
angle_file_for_viewer = "data/angle.json"

# --- CONSTANTES E ESTADO (SEU CÓDIGO ORIGINAL) ---
GYRO_SENS = 131.0
ACCEL_SENS = 16384.0
G_TO_MS2 = 9.80665
prev_timestamp = None
angle = {'x': 0.0, 'y': 0.0, 'z': 0.0}

async def echo(websocket):
    global prev_timestamp, angle

    # Inicializa o arquivo de ângulo para o visualizador
    with open(angle_file_for_viewer, "w", encoding="utf-8") as f:
        json.dump(angle, f)

    async for message in websocket:
        # TODA A SUA LÓGICA DE CÁLCULO PERMANECE INTACTA
        data = json.loads(message)
        rx, ry, rz = data['gyroX'], data['gyroY'], data['gyroZ']
        ax, ay, az = data['accelX'], data['accelY'], data['accelZ']
        encoderL, encoderR = data['encoderL'], data['encoderR']
        ts = data['timestamp']

        gyro = {'x': rx / GYRO_SENS, 'y': ry / GYRO_SENS, 'z': rz / GYRO_SENS}
        accel_g = {'x': ax / ACCEL_SENS, 'y': ay / ACCEL_SENS, 'z': az / ACCEL_SENS}
        accel = {'x': accel_g['x'] * G_TO_MS2, 'y': accel_g['y'] * G_TO_MS2, 'z': accel_g['z'] * G_TO_MS2}
        enc = {'encoderL': encoderL, 'encoderR': encoderR}

        if prev_timestamp is not None:
            dt = (ts - prev_timestamp) / 1000.0
            angle['x'] += gyro['x'] * dt
            angle['y'] += gyro['y'] * dt
            angle['z'] += gyro['z'] * dt
        prev_timestamp = ts

        # Monta o objeto de saída completo (SEU CÓDIGO ORIGINAL)
        out = {
            'timestamp_ms': ts,
            'accel_m_s2': accel,
            'gyro_dps': gyro,
            'angle_deg': angle,
            'enc': enc,
        }

        # --- TAREFAS DE GRAVAÇÃO ---

        # TAREFA 1: Grava o log completo em 'rotacao.txt' (SEU CÓDIGO ORIGINAL, MANTIDO)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")

        # TAREFA 2: Grava APENAS o ângulo em 'angle.json' (NOVA TAREFA, ADICIONADA)
        with open(angle_file_for_viewer, "w", encoding="utf-8") as f:
            json.dump(out['angle_deg'], f)

        # Imprime o log completo no console (SEU CÓDIGO ORIGINAL, MANTIDO)
        print(json.dumps(out))

async def main():
    async with serve(echo, "0.0.0.0", 8765):
        print("[SERVER] rodando em 0.0.0.0:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
