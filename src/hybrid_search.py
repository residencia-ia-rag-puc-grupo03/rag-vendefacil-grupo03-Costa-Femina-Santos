"""
Etapa 2 - item 2: busca híbrida (denso + esparso) com fusão RRF.

Combina a busca densa (embeddings, via FAISS) com a busca esparsa (BM25)
e funde os dois rankings com Reciprocal Rank Fusion (RRF), em vez de somar
os scores das duas buscas (que vêm em escalas incomparáveis).

Este módulo NÃO aplica os filtros de metadados extraídos pelo Query Analyzer -
isso é o item 3 do enunciado (filtro na busca vetorial), que fica por conta
de outro integrante do trio. Aqui a busca roda sobre a base inteira.

Correção (Etapa 4 - item 23): diversificação por doc_type no resultado da
fusão. O índice tem ~87% de chunks `customer` + `sale` (2000 + 3000 de
~5718). Numa busca híbrida SEM filtro, esses registros quase-duplicados
("Cliente CUSTxxxx: ...", "Venda ...") pontuam alto em densa E BM25 e afogam
os chunks narrativos (`ata`, `email`, `product`, `employee` - juntos < 3% do
índice) para fora do top-k, mesmo quando a pergunta é sobre uma reunião ou um
e-mail (causa raiz confirmada de Q02 e Q08 - ver RELATORIO.md). Perguntas que
realmente querem dados de cliente/venda passam pela busca FILTRADA (o Query
Analyzer extrai `customer_id`/`state`) ou pela rota estruturada, nunca só por
aqui - então limitar quantos chunks desses dois tipos entram no top-k da
híbrida bruta é seguro e ataca as duas falhas de uma vez. Nada é descartado:
o excedente vai pro fim da lista e serve de backfill.
"""

import re
import unicodedata
from collections import defaultdict

from rank_bm25 import BM25Okapi

from query_index import load_index

# doc_types que dominam o índice e afogam os chunks narrativos numa busca
# sem filtro. Só estes dois são limitados; todo o resto (ticket, ata, log...)
# fica sem teto, porque perguntas que precisam de vários chunks de um mesmo
# tipo (ex.: "todos os chamados Crítica" - Q06) usam a busca filtrada, não
# esta, e ali o teto não se aplica.
_FLOODING_DOC_TYPES = {"customer", "sale"}
# Máximo de chunks de CADA tipo inundante que podem ocupar o resultado antes
# do backfill. 3 é suficiente pra uma pergunta que de fato precise de um
# punhado de clientes, sem deixar os 2000 registros varrerem o top-k.
_FLOODING_TYPE_CAP = 3


def _tokenize(text: str) -> list[str]:
    """Tokenização simples pro BM25: minúsculo, sem acento, só letras/números."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """
    Mantém, além do índice FAISS já existente, um índice BM25 construído
    sobre o mesmo conjunto de chunks (lidos direto do docstore do FAISS,
    sem reprocessar os arquivos de origem).
    """

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.docs = list(vectorstore.docstore._dict.values())

        self.doc_by_id = {}
        corpus_tokens = []
        self._chunk_ids_in_corpus_order = []
        for doc in self.docs:
            chunk_id = doc.metadata.get("chunk_id")
            self.doc_by_id[chunk_id] = doc
            self._chunk_ids_in_corpus_order.append(chunk_id)
            corpus_tokens.append(_tokenize(doc.page_content))

        self.bm25 = BM25Okapi(corpus_tokens)

    def dense_search(self, query: str, k: int = 10) -> list[str]:
        """Busca densa: ranking por similaridade de embedding no FAISS."""
        results = self.vectorstore.similarity_search(query, k=k)
        return [r.metadata.get("chunk_id") for r in results]

    def sparse_search(self, query: str, k: int = 10) -> list[str]:
        """Busca esparsa: ranking BM25 por sobreposição de termos."""
        scores = self.bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._chunk_ids_in_corpus_order[i] for i in ranked_idx]

    def hybrid_search(self, query: str, k: int = 5, fetch_k: int = 60,
                      rrf_k: int = 60, diversify: bool = True):
        """
        Roda as duas buscas, funde os rankings com RRF e devolve os top-k.

        score_RRF(doc) = soma, para cada recuperador em que o doc aparece,
        de 1 / (rrf_k + posição do doc naquele ranking)

        `fetch_k` (candidatos que cada recuperador traz antes da fusão) foi de
        20 para 60: com 20 candidatos por lado, um chunk narrativo com bom
        casamento lexical (ex.: a ata que cita "Supermercado Boa Compra ...
        Savassi", Q08) nem chegava a ser candidato. `diversify=True` aplica o
        teto por doc_type descrito no cabeçalho do módulo.
        """
        dense_ids = self.dense_search(query, k=fetch_k)
        sparse_ids = self.sparse_search(query, k=fetch_k)

        rrf_scores = defaultdict(float)
        for rank, chunk_id in enumerate(dense_ids, start=1):
            rrf_scores[chunk_id] += 1.0 / (rrf_k + rank)
        for rank, chunk_id in enumerate(sparse_ids, start=1):
            rrf_scores[chunk_id] += 1.0 / (rrf_k + rank)

        ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        if diversify:
            ranked = self._diversify_by_doc_type(ranked)

        fused = ranked[:k]
        return [(self.doc_by_id[chunk_id], score) for chunk_id, score in fused]

    def _diversify_by_doc_type(self, ranked):
        """
        Recebe a lista JÁ fundida por RRF (em ordem decrescente de score) e
        empurra para o fim os chunks de um `doc_type` inundante que passem do
        teto `_FLOODING_TYPE_CAP`, deixando os chunks narrativos que vieram
        logo abaixo subirem. Não remove nada: `mantidos + excedente` preserva
        todos os candidatos, só muda a ordem, então o corte final em `[:k]`
        ainda tem backfill se faltar resultado diverso.
        """
        kept, overflow = [], []
        type_count = defaultdict(int)
        for chunk_id, score in ranked:
            doc = self.doc_by_id.get(chunk_id)
            doc_type = doc.metadata.get("doc_type") if doc is not None else None
            if doc_type in _FLOODING_DOC_TYPES and type_count[doc_type] >= _FLOODING_TYPE_CAP:
                overflow.append((chunk_id, score))
            else:
                kept.append((chunk_id, score))
                type_count[doc_type] += 1
        return kept + overflow


def _print_docs(title: str, docs) -> None:
    print(f"  {title}")
    if not docs:
        print("    (nenhum resultado)")
        return
    for i, doc in enumerate(docs, 1):
        snippet = doc.page_content[:110].replace("\n", " ")
        print(f"    {i}. [{doc.metadata.get('doc_type')}] {snippet}...")


def _print_fused(title: str, fused) -> None:
    print(f"  {title}")
    if not fused:
        print("    (nenhum resultado)")
        return
    for i, (doc, score) in enumerate(fused, 1):
        snippet = doc.page_content[:110].replace("\n", " ")
        print(f"    {i}. (RRF={score:.4f}) [{doc.metadata.get('doc_type')}] {snippet}...")


if __name__ == "__main__":
    vectorstore = load_index()
    retriever = HybridRetriever(vectorstore)

    # Perguntas escolhidas pra evidenciar um caso em que cada recuperador
    # sozinho falha: a 1ª é paráfrase pura (o denso deve ir bem, o BM25 não
    # tem termo literal em comum); a 2ª cita um identificador exato de
    # ticket (o BM25 deve achar de cara, o denso pode não priorizar o
    # código); a 3ª mistura os dois casos.
    TEST_QUESTIONS = [
        "O que a empresa faz quando os dados dos clientes vazam?",
        "ticket TCK-1049",
        "cliente quer parar de usar o módulo de estoque",
    ]

    for question in TEST_QUESTIONS:
        print(f"\nPergunta: {question}")

        dense_docs = [retriever.doc_by_id[cid] for cid in retriever.dense_search(question, k=5)]
        _print_docs("Somente denso (embeddings):", dense_docs)

        sparse_docs = [retriever.doc_by_id[cid] for cid in retriever.sparse_search(question, k=5)]
        _print_docs("Somente esparso (BM25):", sparse_docs)

        fused = retriever.hybrid_search(question, k=5)
        _print_fused("Fusão RRF (híbrido):", fused)
