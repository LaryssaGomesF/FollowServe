CAMINHO_ARQUIVO = "data/dados_mqtt.txt" 
INTERVALO_ESPERADO = 15  # Agora o intervalo é de 15ms

def verificar_variacao_timestamp():
    valores = []

    try:
        with open(CAMINHO_ARQUIVO, "r", encoding="utf-8") as f:
            for num_linha, linha in enumerate(f, start=1):
                linha = linha.strip()

                if not linha:
                    continue

                try:
                    # Extrai o primeiro valor (timestamp)
                    # Exemplo esperado: [1715000, ...] -> 1715000
                    timestamp = int(linha.strip("[]").split(",")[0])
                    valores.append((num_linha, timestamp))

                except (ValueError, IndexError):
                    print(f"⚠️ Linha {num_linha} com formato inválido: {linha}")

    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {CAMINHO_ARQUIVO}")
        return

    if len(valores) < 2:
        print("⚠️ Dados insuficientes para validar a sequência.")
        return

    erros = []

    for i in range(1, len(valores)):
        linha_ant, tempo_ant = valores[i - 1]
        linha_atual, tempo_atual = valores[i]

        diferenca = tempo_atual - tempo_ant

        # Verifica se a diferença é diferente de 15ms
        if diferenca != INTERVALO_ESPERADO:
            erros.append(
                f"❌ Erro entre linhas {linha_ant} e {linha_atual}: "
                f"Esperado +{INTERVALO_ESPERADO}ms, mas variou {diferenca}ms "
                f"({tempo_ant} -> {tempo_atual})"
            )

    if erros:
        print(f"\nForam encontrados {len(erros)} erro(s) na sequência:\n")
        for erro in erros:
            print(erro)
    else:
        print(f"\n✅ Tudo certo! Todos os dados chegaram com intervalo de {INTERVALO_ESPERADO}ms.")

if __name__ == "__main__":
    verificar_variacao_timestamp()