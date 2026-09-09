"""
Script de diagnóstico pontual - NÃO chama nenhuma API (só usa o índice FAISS
local + BM25 local), então pode rodar quantas vezes quiser sem gastar cota.

Objetivo: descobrir por que algumas perguntas do benchmark voltam com
`sem_evidencia` (retrieve() retornando lista vazia de docs), já que a lógica
em generate.retrieve() tem um fallback pra busca híbrida que, em teoria,
nunca deveria devolver zero resultados.

Roda a partir da raiz do projeto:
    python eval\\diagnostico_retrieval.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

from query_index import load_index
from query_analyzer import QueryAnalyzer
from search import FilteredVectorSearch
from hybrid_search import HybridRetriever

PERGUNTAS_TESTE = [
    "Quem é o responsável técnico (Tech Lead) e a gerente de produto (PM) do VendeFácil Estoque?",
    "Quais chamados com prioridade 'Crítica' foram registrados no sistema e qual é o SLA de solução para esse nível?",
    "Listar os logs de erro registrados para o cliente 'CUST008' (Auto Peças Central) no serviço de pagamento (pay).",
]

print("Carregando índice...")
vectorstore = load_index()
analyzer = QueryAnalyzer(vectorstore)
filtered_search = FilteredVectorSearch(vectorstore)
hybrid_retriever = HybridRetriever(vectorstore)

for question in PERGUNTAS_TESTE:
    print("\n" + "=" * 90)
    print(f"PERGUNTA: {question}")

    filters = analyzer.analyze(question).to_dict()
    print(f"Filtros extraídos pelo Query Analyzer: {filters}")

    if filters:
        _, filtered_docs = filtered_search.search(question, k=5, use_filters=True)
        print(f"Resultado da busca FILTRADA: {len(filtered_docs)} chunk(s)")
        for d in filtered_docs[:3]:
            print(f"   - chunk_id={d.metadata.get('chunk_id')} doc_type={d.metadata.get('doc_type')}")
    else:
        print("(nenhum filtro extraído, pulando busca filtrada)")

    fused = hybrid_retriever.hybrid_search(question, k=5)
    print(f"Resultado da busca HÍBRIDA (fallback): {len(fused)} chunk(s)")
    for doc, score in fused[:3]:
        print(f"   - chunk_id={doc.metadata.get('chunk_id')} doc_type={doc.metadata.get('doc_type')} RRF={score:.4f}")

print("\n" + "=" * 90)
print("Se 'busca HÍBRIDA' deu 0 chunk(s) em algum caso acima, o bug está dentro de")
print("hybrid_search() ou na construção do índice/docstore. Se deu >0 mas o benchmark")
print("mesmo assim recusou por sem_evidencia, o bug está em outro ponto do fluxo")
print("(ex.: alguma exceção sendo engolida em algum lugar entre retrieve() e answer_question()).")
