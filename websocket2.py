import asyncio
import json
from websockets.asyncio.server import serve
import os

# Garante que a pasta 'data' exista
os.makedirs("data", exist_ok=True)
log_file = "data/rotacao.txt"

# Sensibilidades do MPU6050 configuradas no ESP32:
GYRO_SENS = 131.0       # LSB / (°/s) para ±250°/s
ACCEL_SENS = 16384.0    # LSB / g     para ±2g
G_TO_MS2   = 9.80665    # 1g = 9.80665 m/s²

# Estado para integração de ângulo
prev_timestamp = None      # em milissegundos
angle = {'x': 0.0, 'y': 0.0, 'z': 0.0}

async def echo(websocket):
    global prev_timestamp, angle

    async for message in websocket:
        # 1) Parse JSON
        data = json.loads(message)

        # 2) Extrai brutos
        rx, ry, rz = data['gyroX'], data['gyroY'], data['gyroZ']
        ax, ay, az = data['accelX'], data['accelY'], data['accelZ']
        encoderL, encoderR = data['encoderL'], data['encoderR']
        ts = data['timestamp']   # em ms

        # 3) Converte raw → unidades reais
        gyro = {
            'x': rx / GYRO_SENS,
            'y': ry / GYRO_SENS,
            'z': rz / GYRO_SENS
        }
        accel_g = {
            'x': ax / ACCEL_SENS,
            'y': ay / ACCEL_SENS,
            'z': az / ACCEL_SENS
        }
        accel = {
            'x': accel_g['x'] * G_TO_MS2,
            'y': accel_g['y'] * G_TO_MS2,
            'z': accel_g['z'] * G_TO_MS2
        }
        enc = {
            'encoderL': encoderL,
            'encoderR': encoderR,
        }

        # 4) Integra gyros → ângulo (°)
        if prev_timestamp is not None:
            dt = (ts - prev_timestamp) / 1000.0  # de ms → s
            angle['x'] += gyro['x'] * dt
            angle['y'] += gyro['y'] * dt
            angle['z'] += gyro['z'] * dt
        prev_timestamp = ts

        # 5) Monta objeto de saída
        rotFile = {
            'timestamp_ms': ts,
            'accel_m_s2': accel,
            'gyro_dps': gyro,
            'angle_deg': angle,
            'enc': enc,
        }

        # 6) Grava no log
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rotFile) + "\n")

        # Também imprime no console
        print(json.dumps(out))

async def main():
    async with serve(echo, "0.0.0.0", 8765):
        print("[SERVER] rodando em 0.0.0.0:8765")
        await asyncio.Future()  # bloqueia para sempre

if __name__ == "__main__":
    asyncio.run(main())
    