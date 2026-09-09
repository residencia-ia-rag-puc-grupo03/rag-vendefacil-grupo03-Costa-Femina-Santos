"""
Etapa 4 - item 3: interface de demonstração (CLI).

Escolhemos CLI entre as três opções aceitas pelo enunciado (CLI, Streamlit,
FastAPI) porque não exige nenhuma dependência nova (o requirements.txt não
tem streamlit/flask/fastapi) e é a opção com menos coisa que pode quebrar ao
vivo no Demo Day - só terminal e Python puro, sem servidor, sem porta, sem
navegador.

Roda com: `python src/demo.py` (a partir da raiz do projeto, com o índice
FAISS já construído e OPENAI_API_KEY configurada no .env).
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from generate import answer_question
from query_index import load_index
from query_analyzer import QueryAnalyzer
from search import FilteredVectorSearch
from hybrid_search import HybridRetriever

BANNER = r"""
============================================================
  VendeFacil - Assistente RAG Corporativo (demo)
============================================================
"""

# Perguntas de exemplo já testadas (uma de cada categoria do benchmark),
# pro roteiro do Demo Day: "leve 2 perguntas de demo já testadas".
PERGUNTAS_DE_EXEMPLO = [
    "Qual é a política de reembolso da empresa?",
    "Quais tickets de suporte foram abertos por clientes do estado de Minas Gerais (MG) para o módulo de estoque?",
    "Qual o salário do funcionário com maior remuneração?",
    "Quem descobriu o Brasil?",
]

REFUSAL_LABELS = {
    "lgpd": "protegido por LGPD",
    "fora_de_escopo": "fora do escopo da VendeFácil",
    "sem_evidencia": "sem evidência suficiente na base",
}


def _print_response(response) -> None:
    print()
    if response.is_refusal:
        label = REFUSAL_LABELS.get(response.refusal_reason, response.refusal_reason)
        print(f"[RECUSADO - {label}]")
        print(response.answer)
        return

    print(f"[confiança: {response.confidence_level}]")
    print(response.answer)
    print()
    print(f"Fontes citadas ({len(response.sources_used)}):")
    for source in response.sources_used:
        print(f"  - {source.filepath}  (chunk_id={source.chunk_id})")
        quotation = source.quotation.replace("\n", " ")
        if len(quotation) > 160:
            quotation = quotation[:160] + "..."
        print(f'    "{quotation}"')


def main() -> None:
    print(BANNER)

    if not config.OPENAI_API_KEY:
        print(
            "ERRO: OPENAI_API_KEY não configurada no .env. Configure antes de rodar a demo "
            "(veja .env.example)."
        )
        sys.exit(1)

    print("Carregando índice FAISS... (isso pode levar alguns segundos)")
    vectorstore = load_index()
    analyzer = QueryAnalyzer(vectorstore)
    filtered_search = FilteredVectorSearch(vectorstore)
    hybrid_retriever = HybridRetriever(vectorstore)

    print("\nPerguntas de exemplo (já testadas):")
    for i, exemplo in enumerate(PERGUNTAS_DE_EXEMPLO, start=1):
        print(f"  {i}. {exemplo}")
    print("\nDigite o número de um exemplo, sua própria pergunta, ou 'sair' para encerrar.\n")

    while True:
        try:
            entrada = input("Pergunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not entrada:
            continue
        if entrada.lower() in ("sair", "exit", "quit"):
            print("Encerrando.")
            break

        if entrada.isdigit() and 1 <= int(entrada) <= len(PERGUNTAS_DE_EXEMPLO):
            pergunta = PERGUNTAS_DE_EXEMPLO[int(entrada) - 1]
            print(f"(usando exemplo {entrada}: {pergunta})")
        else:
            pergunta = entrada

        try:
            response = answer_question(pergunta, vectorstore, analyzer, filtered_search, hybrid_retriever)
            _print_response(response)
        except Exception as error:
            # Nunca deixa a demo travar por causa de uma pergunta ruim - registra
            # o erro e volta pro prompt, em vez de derrubar o programa inteiro.
            print(f"\n[ERRO ao processar a pergunta: {error}]")

        print()


if __name__ == "__main__":
    main()
