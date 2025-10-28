import asyncio
import json
import os
from websockets.asyncio.server import serve

# Configurações
GYRO_SENS = 131.0
LOG_FILE = "data/log.txt"
ANGLE_FILE = "data/angle.json"

async def echo(websocket):
    prev_timestamp = None
    angle = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    print("Cliente conectado!")

    async for message in websocket:
        print("Mensagem recebida:", message)
        data = json.loads(message)
        rx, ry, rz = data['gyroX'], data['gyroY'], data['gyroZ']
        ts = data['timestamp']

        gyro = {
            'x': rx / GYRO_SENS,
            'y': ry / GYRO_SENS,
            'z': rz / GYRO_SENS
        }

        if prev_timestamp is not None:
            dt = (ts - prev_timestamp) / 1000.0
            angle['x'] += gyro['x'] * dt
            angle['y'] += gyro['y'] * dt
            angle['z'] += gyro['z'] * dt
        prev_timestamp = ts

        print(f"Gravando ângulo: {angle}")

        try:
            os.makedirs("data", exist_ok=True)  # Garante pasta 'data'
            with open(ANGLE_FILE, "w", encoding="utf-8") as f:
                json.dump(angle, f)
            print("Arquivo angle.json atualizado!")
        except Exception as e:
            print("Erro ao salvar angle.json:", e)

        # Teste simples de gravação de arquivo para confirmar permissão
        try:
            with open("teste.txt", "w") as f:
                f.write("Gravação de teste OK!\n")
            print("Arquivo teste.txt criado com sucesso!")
        except Exception as e:
            print("Erro ao criar arquivo teste.txt:", e)

        # Log completo
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                'timestamp_ms': ts,
                'gyro_dps': gyro,
                'angle_deg': angle
            }) + "\n")

async def start_server():
    async with serve(echo, "0.0.0.0", 8765):
        print("[SERVER] rodando em 0.0.0.0:8765")
        await asyncio.Future()  # Loop infinito

def run_server():
    asyncio.run(start_server())

if __name__ == "__main__":
    run_server()
