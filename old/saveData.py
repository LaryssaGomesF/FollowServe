import asyncio
from websockets.asyncio.server import serve
import os
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

# Garante que a pasta 'data' exista
os.makedirs("data", exist_ok=True)

# Caminho fixo do arquivo de log
log_file = "data/log.txt"

async def echo(websocket):
    try:
        async for message in websocket:
            print(message)
            # Salva a mensagem no arquivo
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(message + "\n")
    except (ConnectionClosedError, ConnectionClosedOK):
        print("[SERVER] Cliente desconectado, aguardando nova conexão...")
    except Exception as e:
        print(f"[SERVER] Erro inesperado: {e}")

async def main():
    async with serve(echo, "0.0.0.0", 8765):
        print("[SERVER] Servidor WebSocket rodando na porta 8765...")
        await asyncio.Future()  # Mantém o servidor ativo

if __name__ == "__main__":
    asyncio.run(main())
