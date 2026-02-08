# verificar_sequencia.py
# Verifica se o primeiro valor (ignorando a última casa à direita)
# varia de 1 em 1

CAMINHO_ARQUIVO = "data/dados_mqtt.txt"  # ajuste o caminho se necessário


def verificar_variacao_primeiro_valor():
    valores = []

    try:
        with open(CAMINHO_ARQUIVO, "r", encoding="utf-8") as f:
            for num_linha, linha in enumerate(f, start=1):
                linha = linha.strip()

                if not linha:
                    continue

                try:
                    primeiro_valor = int(linha.strip("[]").split(",")[0])

                    # ignora a casa mais à direita
                    valor_sem_ultima_casa = primeiro_valor // 10

                    valores.append(
                        (num_linha, primeiro_valor, valor_sem_ultima_casa)
                    )

                except Exception:
                    print(f"⚠️ Linha {num_linha} inválida: {linha}")

    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {CAMINHO_ARQUIVO}")
        return

    if len(valores) < 2:
        print("⚠️ Poucos dados para verificar.")
        return

    erros = []

    for i in range(1, len(valores)):
        linha_atual, valor_original, valor_atual = valores[i]
        linha_ant, valor_original_ant, valor_ant = valores[i - 1]

        if valor_atual != valor_ant + 1:
            erros.append(
                f"Erro entre linhas {linha_ant} e {linha_atual}: "
                f"{valor_original_ant} ({valor_ant}) -> "
                f"{valor_original} ({valor_atual})"
            )

    if erros:
        print("\n❌ Foram encontrados erros na sequência:\n")
        for erro in erros:
            print(erro)
    else:
        print("\n✅ Tudo certo! Ignorando a última casa, a sequência está correta.")


if __name__ == "__main__":
    verificar_variacao_primeiro_valor()
