import paho.mqtt.client as mqtt
import os

BROKER = "localhost"
PORT = 1883
TOPIC = "robot/sensors"
ARQUIVO = "data/dados_mqtt.txt"

# Garante que a pasta existe
os.makedirs(os.path.dirname(ARQUIVO), exist_ok=True)

def on_connect(client, userdata, flags, rc):
    print(f"Conectado ao broker no tópico '{TOPIC}' (Código: {rc})")
    client.subscribe(TOPIC)

import re # Adicione este import no topo do script

def on_message(client, userdata, msg):
    try:
        # Decodifica o payload recebido
        payload = msg.payload.decode("utf-8")
        
        # Encontra todos os padrões que começam com [ e terminam com ]
        # Isso quebra o bloco [..][..][..] em uma lista de strings individuais
        leituras = re.findall(r'\[.*?\]', payload)

        if leituras:
            with open(ARQUIVO, "a") as f:
                for linha in leituras:
                    # Escreve cada leitura individual seguida de uma quebra de linha
                    f.write(linha + "\n")
            
            print(f"Bloco processado: {len(leituras)} linhas salvas.")
            
    except Exception as e:
        print(f"Erro ao processar mensagem: {e}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, 60)
client.loop_forever()