import paho.mqtt.client as mqtt
from datetime import datetime
import ast

BROKER = "localhost"
PORT = 1883
TOPIC = "robot/sensors"
ARQUIVO = "data/dados_mqtt.txt"

def on_connect(client, userdata, flags, rc):
    print("Conectado com código:", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8")

    try:
        # Converte string "[...]" em lista Python
        dados = ast.literal_eval(payload)

        # desempacota
        ts, ax, ay, az, gx, gy, gz, encR, encL = dados

        linha = f"[{ts},{ax},{ay},{az},{gx},{gy},{gz},{encR},{encL}]\n"

        with open(ARQUIVO, "a") as f:
            f.write(linha)

        print("Salvo:", linha.strip())

    except Exception as e:
        print("Erro ao processar mensagem:", e)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, 60)
client.loop_forever()
