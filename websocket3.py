# servidor_receptor_final.py
import asyncio
import json
import os
from websockets.asyncio.server import serve

# --- ARQUIVO DE SAÍDA PARA O VISUALIZADOR ---
os.makedirs("data", exist_ok=True)
angle_file_for_viewer = "data/angle.json"

# --- ESTADO INICIAL ---
# Apenas para garantir que o arquivo exista no início.
initial_angle = {'x': 0.0, 'y': 0.0, 'z': 0.0}

async def echo(websocket):
    """
    Esta função agora apenas recebe os ângulos finais do robô
    e os salva no arquivo JSON para a animação.
    """
    print(f"Novo cliente conectado: {websocket.remote_address}")

    try:
        async for message in websocket:
            try:
                # 1. Recebe a mensagem do robô
                data = json.loads(message)

                # 2. Extrai os ângulos finais das chaves que você está enviando
                #    (Assumindo que 'gyroX' agora contém Roll, 'gyroY' contém Pitch, etc.)
                final_angles = {
                    'x': data['roll'],  # Roll
                    'y': data['pitch'],  # Pitch
                    'z': data['yaw']   # Yaw
                }

                # 3. Salva os ângulos diretamente no arquivo para o visualizador
                with open(angle_file_for_viewer, "w", encoding="utf-8") as f:
                    json.dump(final_angles, f)

                # 4. Imprime no console para depuração
                #    Usamos f-string para uma saída mais legível.
                print(f"Ângulos recebidos -> Roll: {final_angles['x']:.2f}, Pitch: {final_angles['y']:.2f}, Yaw: {final_angles['z']:.2f}")

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Erro ao processar mensagem: {e}. Mensagem recebida: {message}")
                continue

    except Exception as e:
        print(f"Conexão fechada com erro: {e}")
    finally:
        print(f"Cliente {websocket.remote_address} desconectado.")


async def main():
    # Garante que o arquivo de ângulo exista desde o início para o visualizador não fechar
    with open(angle_file_for_viewer, "w", encoding="utf-8") as f:
        json.dump(initial_angle, f)

    async with serve(echo, "0.0.0.0", 8765):
        print("[SERVER] Servidor de orientação iniciado em 0.0.0.0:8765")
        print("Aguardando dados de orientação do robô...")
        await asyncio.Future()  # Mantém o servidor rodando para sempre

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[SERVER] Servidor encerrado.")

