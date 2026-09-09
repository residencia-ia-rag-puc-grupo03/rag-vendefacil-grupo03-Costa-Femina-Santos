"""
Diagnóstico pontual da Q07 - NÃO chama API, só usa o índice local (o novo,
reindexado). Roda a partir da raiz do projeto:
    python eval\\diagnostico_q07.py
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

QUESTION = "Listar os logs de erro registrados para o cliente 'CUST008' (Auto Peças Central) no serviço de pagamento (pay)."

print("Carregando índice...")
vectorstore = load_index()
analyzer = QueryAnalyzer(vectorstore)
filtered_search = FilteredVectorSearch(vectorstore)
hybrid_retriever = HybridRetriever(vectorstore)

filters = analyzer.analyze(QUESTION).to_dict()
print(f"\nFiltros extraídos: {filters}")

_, filtered_docs = filtered_search.search(QUESTION, k=5, use_filters=True)
print(f"\nResultado da busca FILTRADA: {len(filtered_docs)} chunk(s)")
for d in filtered_docs:
    print(f"   - chunk_id={d.metadata.get('chunk_id')} doc_type={d.metadata.get('doc_type')} "
          f"module={d.metadata.get('module')} customer_id={d.metadata.get('customer_id')}")

# Confere DIRETO no docstore (sem busca semântica nenhuma) se o chunk esperado
# existe com o metadado certo - prova dos nove, ignora ranking/similaridade.
print("\nBusca direta no docstore (ignora similaridade, só filtra por metadado):")
todos_docs = list(vectorstore.docstore._dict.values())
match_direto = [
    d for d in todos_docs
    if d.metadata.get("doc_type") == "log"
    and d.metadata.get("customer_id") == "CUST008"
    and d.metadata.get("module") == "pay"
]
print(f"   Chunks com doc_type=log, customer_id=CUST008, module=pay: {len(match_direto)}")
for d in match_direto:
    print(f"   - {d.metadata}")

fused = hybrid_retriever.hybrid_search(QUESTION, k=5)
print(f"\nResultado da busca HÍBRIDA (fallback): {len(fused)} chunk(s)")
for doc, score in fused:
    print(f"   - chunk_id={doc.metadata.get('chunk_id')} doc_type={doc.metadata.get('doc_type')} "
          f"customer_id={doc.metadata.get('customer_id')} RRF={score:.4f}")
