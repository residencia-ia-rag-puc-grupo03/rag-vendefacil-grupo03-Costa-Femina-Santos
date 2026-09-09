"""
Consulta estruturada para perguntas de AGREGAÇÃO (MAX/MIN) sobre dados
tabulares (ex.: "qual cliente tem o MAIOR MRR em SP?").

Por que isso existe (Etapa 4 - item 15): confirmamos rodando o benchmark de
verdade (Q10) que busca vetorial (FAISS + embeddings) estruturalmente NÃO
consegue responder esse tipo de pergunta de forma confiável. "Ter o maior
MRR" não é uma propriedade que fica mais "parecida" no espaço de embeddings -
é uma comparação numérica entre TODOS os registros, e a busca por similaridade
só olha os k mais parecidos com a pergunta, não os k com maior valor num
campo. Testamos: nem o gabarito do benchmark acertou o valor real (o dado
real tem um cliente com MRR ainda maior que os dois que apareceram na
comparação gabarito-vs-resposta). Isso não é bug de chunking, k, threshold
ou prompt - é a ferramenta errada pra esse tipo de pergunta.

A solução correta (e o que qualquer pipeline de RAG de produção faz para
perguntas desse tipo) é ROTEAMENTO: detectar a intenção de agregação antes de
acionar a busca vetorial, e responder com uma consulta estruturada direta
(pandas) sobre a fonte original - não com o LLM "adivinhando" a partir de um
recorte de k documentos. Isso elimina qualquer chance de erro nesse tipo de
pergunta, porque não depende de embedding nenhum: é aritmética exata sobre
os dados reais.

Este módulo cobre hoje só customers.csv / campo mrr, que é o caso real do
benchmark. Novas fontes/campos podem ser adicionados no mesmo padrão.
"""
import os
import re
from typing import Optional

import pandas as pd

from query_analyzer import STATE_ALIASES, normalize, contains

_SUPERLATIVE_MAX = ["maior", "mais alto", "maximo", "máximo", "top 1", "melhor"]
_SUPERLATIVE_MIN = ["menor", "mais baixo", "minimo", "mínimo", "pior"]

# Termos reais usados nas perguntas do benchmark e no dado (coluna `mrr` de
# customers.csv) para "receita recorrente mensal".
_MRR_TERMS = ["mrr", "receita recorrente", "receita mensal", "receita recorrente mensal"]


def is_customer_mrr_aggregation(question: str) -> Optional[dict]:
    """
    Detecta se a pergunta é uma agregação MAX/MIN sobre MRR de clientes.
    Retorna {"op": "max"|"min", "state": "SP"|None} ou None se não for esse
    tipo de pergunta - nunca "força" uma detecção incerta.
    """
    query = normalize(question)

    if not any(contains(query, normalize(term)) for term in _MRR_TERMS):
        return None

    if any(contains(query, normalize(term)) for term in _SUPERLATIVE_MAX):
        op = "max"
    elif any(contains(query, normalize(term)) for term in _SUPERLATIVE_MIN):
        op = "min"
    else:
        return None

    state = None
    for state_name, acronym in STATE_ALIASES.items():
        if contains(query, normalize(state_name)) or contains(query, normalize(acronym)):
            state = acronym
            break

    return {"op": op, "state": state}


def answer_customer_mrr_aggregation(spec: dict, customers_csv_path: str) -> Optional[dict]:
    """
    Executa a agregação de verdade sobre o CSV real (não sobre chunks
    recuperados) e devolve os dados da linha vencedora, prontos para montar
    uma citação real (mesmo padrão de SourceEvidence: filepath + trecho).
    Retorna None só se o arquivo não existir ou o filtro não bater com
    nenhuma linha - nunca inventa um resultado.
    """
    if not os.path.exists(customers_csv_path):
        return None

    df = pd.read_csv(customers_csv_path)
    subset = df
    if spec.get("state"):
        subset = subset[subset["state"] == spec["state"]]

    if subset.empty or "mrr" not in subset.columns:
        return None

    idx = subset["mrr"].idxmax() if spec["op"] == "max" else subset["mrr"].idxmin()
    row = subset.loc[idx]

    quotation = (
        f"Cliente {row.get('customer_id','')}: {row.get('company_name','')} "
        f"({row.get('segment','')}), localizado em {row.get('city','')}/{row.get('state','')}, "
        f"produto principal {row.get('main_product','')}, MRR de R$ {row.get('mrr','')}."
    )

    return {
        "customer_id": str(row.get("customer_id", "")),
        "company_name": str(row.get("company_name", "")),
        "state": str(row.get("state", "")),
        "mrr": float(row.get("mrr", 0)),
        "main_product": str(row.get("main_product", "")),
        "quotation": quotation,
    }
