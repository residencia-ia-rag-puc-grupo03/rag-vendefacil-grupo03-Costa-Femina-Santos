"""
Diagnóstico stage-by-stage de UMA pergunta - por que ela passa ou falha no
pipeline, camada por camada.

Generaliza os `eval/diagnostico_*.py` (que são recortes pontuais de Q07 e do
bug de retrieval vazio) para qualquer pergunta: um id do benchmark (Q01..Q24)
ou um texto livre entre aspas.

NÃO chama nenhuma API por padrão - só usa o índice FAISS local + BM25 local,
então pode rodar quantas vezes quiser sem gastar cota. Passe `--generate`
para de fato chamar o LLM de síntese no fim (aí sim consome cota).

Uso (a partir da raiz do repositório):
    python src/diagnose.py Q08
    python src/diagnose.py "Qual a política de home office da Engenharia?"
    python src/diagnose.py Q08 --generate
    python src/diagnose.py Q08 -k 10

Este arquivo é uma FERRAMENTA DE DESENVOLVIMENTO. Não faz parte do pipeline
nem da avaliação. Ele apenas lê e imprime o que cada estágio produz, para
localizar em qual etapa (1 ingestão/index, 2 recuperação, 3 síntese/guardrail)
está a causa de uma falha - que é exatamente o que a defesa técnica do Demo
Day pergunta.
"""

import argparse
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from domain_keywords import contains_domain_keyword
from generate import answer_question, is_out_of_scope, retrieve
from hybrid_search import HybridRetriever
from lgpd_policy import classify_question, has_only_restricted_docs
from query_analyzer import QueryAnalyzer
from query_index import load_index
from search import FilteredVectorSearch
from structured_query import is_customer_mrr_aggregation

BENCHMARK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "benchmark",
    "questions_and_ground_truth.json",
)


def _rule(char="=", width=90):
    print(char * width)


def _load_benchmark_question(qid: str):
    """Devolve o dict da pergunta do benchmark com esse id, ou None."""
    if not os.path.exists(BENCHMARK_PATH):
        return None
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    wanted = qid.strip().upper()
    for q in data["questions"]:
        if q["id"].upper() == wanted:
            return q
    return None


def _norm_source(path: str) -> str:
    """Mesma normalização do eval: tira o prefixo 'data/' e barra invertida."""
    return path.replace("\\", "/").removeprefix("data/")


def _show_docs(docs, expected_norm=None, limit=8):
    if not docs:
        print("   (nenhum chunk)")
        return
    for i, doc in enumerate(docs[:limit], 1):
        meta = doc.metadata
        src = meta.get("source_file", "")
        hit = ""
        if expected_norm is not None:
            hit = "  <== fonte esperada" if _norm_source(src) in expected_norm else ""
        print(
            f"   {i:>2}. chunk_id={meta.get('chunk_id')}  doc_type={meta.get('doc_type')}  "
            f"module={meta.get('module')}  state={meta.get('state')}  "
            f"customer_id={meta.get('customer_id')}  sensitivity={meta.get('sensitivity')}{hit}"
        )
        snippet = doc.page_content[:140].replace("\n", " ")
        print(f"       {snippet}...")
    if len(docs) > limit:
        print(f"   ... (+{len(docs) - limit} chunks)")


def diagnose(question: str, benchmark_q=None, k: int = 8, do_generate: bool = False):
    print()
    _rule()
    print(f"PERGUNTA: {question}")
    if benchmark_q:
        print(f"  id={benchmark_q['id']}  categoria={benchmark_q['category']}")
        expected = benchmark_q.get("expected_sources", [])
        print(f"  fontes esperadas (gabarito): {expected or '[] (fora de escopo / recusa)'}")
        gt = benchmark_q["ground_truth_answer"]
        print(f"  gabarito: {gt[:200]}{'...' if len(gt) > 200 else ''}")
    _rule()

    expected_norm = None
    if benchmark_q:
        expected_norm = {_norm_source(p) for p in benchmark_q.get("expected_sources", [])}

    print("\nCarregando índice FAISS + BM25 (local, sem API)...")
    vectorstore = load_index()
    analyzer = QueryAnalyzer(vectorstore)
    filtered_search = FilteredVectorSearch(vectorstore)
    hybrid_retriever = HybridRetriever(vectorstore)

    # --- Estágio 3a: classificação determinística (antes de qualquer busca) ---
    print("\n[1] Guardrails determinísticos (rodam ANTES da busca)")
    lgpd_category = classify_question(question)
    print(f"    classify_question() (LGPD)     -> {lgpd_category}")
    agg = is_customer_mrr_aggregation(question)
    print(f"    is_customer_mrr_aggregation()  -> {agg}  "
          f"{'(resolvida por pandas sobre customers.csv, sem busca vetorial)' if agg else ''}")
    has_kw = contains_domain_keyword(question)
    top1 = vectorstore.similarity_search_with_score(question, k=1)[0][1]
    threshold = config.OUT_OF_SCOPE_SCORE_THRESHOLD
    oos = is_out_of_scope(question, vectorstore)
    print(f"    contains_domain_keyword()      -> {has_kw}")
    print(f"    distância L2 do chunk mais próximo -> {top1:.4f}  (threshold OOS = {threshold})")
    print(f"    is_out_of_scope() -> {oos}  "
          f"{'==> RECUSA fora_de_escopo antes de recuperar nada' if oos else ''}")

    # --- Estágio 2: Query Analyzer + as duas buscas ---
    print("\n[2] Query Analyzer (Etapa 2)")
    filters = analyzer.analyze(question).to_dict()
    print(f"    filtros extraídos: {filters or '{} (nenhum -> vai direto pra busca híbrida)'}")

    if filters:
        print("\n[2a] Busca FILTRADA (densa + filtro de metadado)")
        _, filtered_docs = filtered_search.search(question, k=k, use_filters=True)
        print(f"     {len(filtered_docs)} chunk(s):")
        _show_docs(filtered_docs, expected_norm)
        if not filtered_docs:
            print("     -> filtro casou com ZERO chunks: retrieve() cai pro fallback híbrido sem filtro")

    print("\n[2b] Busca HÍBRIDA (densa + BM25 + RRF, sem filtro)")
    fused = hybrid_retriever.hybrid_search(question, k=k)
    _show_docs([doc for doc, _ in fused], expected_norm)

    # --- O que retrieve() realmente entrega pro LLM ---
    print("\n[3] retrieve() - contexto final montado pra síntese")
    docs, filters_used = retrieve(question, vectorstore, analyzer, filtered_search,
                                  hybrid_retriever, k=k)
    print(f"    filtros usados: {filters_used}")
    print(f"    {len(docs)} chunk(s) no contexto:")
    _show_docs(docs, expected_norm)

    only_restricted = has_only_restricted_docs(docs)
    print(f"\n    has_only_restricted_docs(contexto) -> {only_restricted}  "
          f"{'==> dispara segunda camada LGPD (tenta busca ampla; recusa se ainda restrito)' if only_restricted else ''}")

    if expected_norm is not None and docs:
        retrieved_norm = {_norm_source(d.metadata.get("source_file", "")) for d in docs}
        if expected_norm:
            hits = expected_norm & retrieved_norm
            cr = len(hits) / len(expected_norm)
            print(f"\n    Context Relevance (nível de arquivo, como no eval): {cr:.0%}  "
                  f"({len(hits)}/{len(expected_norm)} fontes esperadas presentes)")
            missing = expected_norm - retrieved_norm
            if missing:
                print(f"    fontes esperadas AUSENTES do contexto: {sorted(missing)}")

    # --- Leitura de qual etapa é a suspeita ---
    print("\n[4] Leitura rápida")
    if oos and benchmark_q and benchmark_q.get("expected_sources"):
        print("    -> Pergunta legítima classificada como fora de escopo (Etapa 3): revisar")
        print("       domain_keywords.py e/ou OUT_OF_SCOPE_SCORE_THRESHOLD (src/check_threshold.py).")
    elif expected_norm and docs and not (expected_norm & {_norm_source(d.metadata.get('source_file','')) for d in docs}):
        print("    -> Nenhuma fonte esperada chegou ao contexto: o problema é de RECUPERAÇÃO")
        print("       (Etapa 1 chunking/index OU Etapa 2 filtro/híbrida). Mexer no prompt não resolve.")
    elif only_restricted:
        print("    -> Contexto só com chunks 'restrito': provável filtro doc_type mal extraído")
        print("       (Etapa 2, query_analyzer.py) empurrando pra recusa LGPD indevida.")
    else:
        print("    -> Recuperação trouxe as fontes certas. Se ainda falha no benchmark, a causa")
        print("       provável é SÍNTESE (Etapa 3): cobertura parcial dos key_points, citação")
        print("       incompleta, ou groundedness. Rode com --generate para ver a resposta real.")

    # --- Opcional: chamar o LLM de verdade ---
    if do_generate:
        print("\n[5] Geração real (chama a API - consome cota)")
        try:
            response = answer_question(question, vectorstore, analyzer, filtered_search,
                                       hybrid_retriever)
            print(response.model_dump_json(indent=2))
        except Exception as error:
            print(f"    ERRO na geração: {error}")
    else:
        print("\n[5] (pulei a geração real - passe --generate pra chamar o LLM)")

    _rule()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnóstico stage-by-stage de uma pergunta do pipeline RAG."
    )
    parser.add_argument("question", help="id do benchmark (ex.: Q08) ou o texto da pergunta entre aspas")
    parser.add_argument("-k", type=int, default=8, help="k de recuperação (default 8, igual ao retrieve())")
    parser.add_argument("--generate", action="store_true",
                        help="também chama o LLM de síntese no fim (consome cota de API)")
    args = parser.parse_args()

    raw = args.question.strip()
    benchmark_q = None
    if len(raw) <= 5 and raw.upper().startswith("Q"):
        benchmark_q = _load_benchmark_question(raw)
        if benchmark_q is None:
            print(f"Aviso: '{raw}' não é um id do benchmark - tratando como texto livre.")
            question = raw
        else:
            question = benchmark_q["question"]
    else:
        question = raw

    diagnose(question, benchmark_q=benchmark_q, k=args.k, do_generate=args.generate)


if __name__ == "__main__":
    main()
