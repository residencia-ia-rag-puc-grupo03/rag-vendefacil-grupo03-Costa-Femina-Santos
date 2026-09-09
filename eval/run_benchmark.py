import argparse
import json
import os
import sys
import time
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

from openai import RateLimitError

import config
from generate import answer_question, retrieve, _build_context
from query_index import load_index
from query_analyzer import QueryAnalyzer
from search import FilteredVectorSearch
from hybrid_search import HybridRetriever
from judge_prompt import call_judge

BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "benchmark", "questions_and_ground_truth.json")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
RELATORIO_PATH = os.path.join(PROJECT_ROOT, "RELATORIO.md")


class _CitedDoc:
    """
    Correção (Etapa 4 - item 18): adaptador simples pra reaproveitar
    `_build_context()` (que espera objetos com `.metadata`/`.page_content`,
    como os Document do LangChain) a partir de `response.sources_used`
    (SourceEvidence: filepath, chunk_id, quotation) - ver justificativa
    completa em `_context_for_judge()` logo abaixo.
    """
    def __init__(self, source):
        self.page_content = source.quotation
        self.metadata = {
            "chunk_id": source.chunk_id,
            "source_file": source.filepath,
            "doc_type": None,
        }


def _context_for_judge(response, fallback_docs):
    """
    Correção (Etapa 4 - item 18, refinada após rodar com openrouter/free): o
    contexto passado pro juiz precisa ser o que a resposta REALMENTE usou
    para responder - mas "o que foi usado" é o CHUNK INTEIRO recuperado, não
    só o trecho que o modelo escolheu citar em `quotation`.

    Descoberta rodando de verdade (Q01): o modelo citou só um trecho parcial
    de um chunk de products.json (sem o preço), mas usou o preço real (que
    está no MESMO chunk, só não na citação) na resposta. Como o preço é dado
    real (confirmado em products.json), isso não é alucinação - mas ao
    reconstruir o contexto do juiz só a partir da citação parcial, o juiz
    nunca via o preço no "contexto" dele, e marcava como não sustentado.

    A correção: para cada fonte citada, procura o chunk_id correspondente
    entre os `docs` de verdade retornados por retrieve() (fallback_docs) e
    usa o `page_content` COMPLETO dele. Só cai para o texto da citação
    isolada quando o chunk_id não existe em fallback_docs - que é exatamente
    o caso do `structured_query.py` (Q10), cujo "chunk" nunca veio de uma
    busca vetorial de verdade.
    """
    docs_by_chunk_id = {d.metadata.get("chunk_id"): d for d in fallback_docs}
    matched_docs = []
    for source in response.sources_used:
        real_doc = docs_by_chunk_id.get(source.chunk_id)
        matched_docs.append(real_doc if real_doc is not None else _CitedDoc(source))

    if matched_docs:
        return _build_context(matched_docs)
    return _build_context(fallback_docs)


def _infer_expected_refusal(ground_truth_answer: str):
    """
    Toda ground_truth_answer que espera recusa começa literalmente com um desses
    marcadores (confirmado lendo as 24 perguntas) - evita manter uma lista de IDs
    hardcoded, e trata corretamente Q17 (mesma categoria "Guardrails & LGPD", mas
    com resposta normal esperada).
    """
    text = ground_truth_answer.strip().upper()
    if text.startswith("FORA DO ESCOPO"):
        return True, "fora_de_escopo"
    if text.startswith("RECUSA DE RESPOSTA"):
        return True, "lgpd"
    return False, None


def _normalize_source_path(path: str) -> str:
    """expected_sources vem com prefixo 'data/'; o metadado real gravado na
    ingestão (e ecoado em sources_used) não tem esse prefixo."""
    return path.replace("\\", "/").removeprefix("data/")


def _sources_recall(expected_sources, sources_used):
    if not expected_sources:
        return None
    expected_norm = {_normalize_source_path(p) for p in expected_sources}
    actual_norm = {_normalize_source_path(s.filepath) for s in sources_used}
    return len(expected_norm & actual_norm) / len(expected_norm)


def _context_relevance(expected_sources, docs):
    """
    Métrica determinística da RAG Triad (não é LLM-as-judge).

    ADAPTAÇÃO DOCUMENTADA: o enunciado (Etapa 4) pede comparar chunk_id
    recuperados com chunk_id do gabarito. Conferimos o arquivo oficial
    `benchmark/questions_and_ground_truth.json` (fornecido pelo professor) por
    completo e ele NÃO contém nenhum campo de chunk_id em nenhuma das 24
    perguntas - só `expected_sources` no nível de arquivo. Reportado ao
    professor; enquanto isso, adaptamos a métrica para o nível de arquivo, que
    é o que o gabarito atual sustenta.

    SE o gabarito vier a incluir chunk_id esperado no futuro (ex.: um campo
    "expected_chunk_ids": [...] por pergunta), troque a comparação abaixo por:
        expected_chunks = set(q["expected_chunk_ids"])
        retrieved_chunks = {doc.metadata.get("chunk_id") for doc in docs}
        return len(expected_chunks & retrieved_chunks) / len(expected_chunks)
    """
    if not expected_sources:
        return None
    expected_norm = {_normalize_source_path(p) for p in expected_sources}
    retrieved_norm = {
        _normalize_source_path(doc.metadata.get("source_file", ""))
        for doc in docs
    }
    return len(expected_norm & retrieved_norm) / len(expected_norm)


def _question_score(record):
    """
    Pontuação oficial por questão (0.0 a 1.0), conforme rubrica do enunciado:
      - 0.5: resposta correta (ou recusa correta, em questões de recusa)
      - 0.3: citação aponta o arquivo/chunk certo
      - 0.2: confidence_level / is_refusal coerentes com a resposta dada

    Para questões de recusa (esperada), o enunciado é explícito: "Recusar uma
    pergunta legítima vale zero, igual a errar" - ou seja, recusa correta/incorreta
    é tudo-ou-nada (não existe "citação" numa recusa, já que sources_used deve
    vir vazio por construção do schema). Por isso, aqui: recusa correta = 1.0,
    recusa incorreta = 0.0.

    Para questões respondíveis, a nota é composta:
      - 0.5 * 1 se o juiz considerou a resposta correta (overall_correct)
      - 0.3 * sources_recall (proporcional, não binário - citar 1 de 2 fontes
        esperadas vale metade dos 0.3, não zero)
      - 0.2 * 1 se is_refusal e confidence_level estão coerentes com o que era
        esperado (aqui: is_refusal deveria ser False e confidence_level != "recusado";
        essa parte já é garantida estruturalmente pelo validador Pydantic do
        RAGResponse, então checamos principalmente se o pipeline não caiu numa
        recusa indevida).

    NOTA: o enunciado não detalha a fórmula exata de partial credit para o
    componente de citação nem para o de coerência - esta é uma interpretação
    explícita e documentada, não uma regra literal do PDF. Revise com a dupla /
    professor se quiser um critério diferente.
    """
    if record.get("error"):
        return 0.0

    if record.get("refusal_check") is not None:
        return 1.0 if record["refusal_check"]["correct"] else 0.0

    generation = record.get("generation")
    judge = record.get("judge")
    if generation is None or judge is None:
        return 0.0

    resposta_correta = 0.5 if judge["overall_correct"] else 0.0

    recall = record.get("sources_recall")
    citacao = 0.3 * recall if recall is not None else 0.0

    coerente = (
        generation["is_refusal"] is False
        and generation["confidence_level"] != "recusado"
    )
    coerencia = 0.2 if coerente else 0.0

    return round(resposta_correta + citacao + coerencia, 3)


def _diagnose_failure(record):

    if record.get("error"):
        return "Erro de execução do pipeline (ver campo 'error') - falha técnica, não de qualidade de resposta."

    refusal_check = record.get("refusal_check")
    if refusal_check is not None and not refusal_check["correct"]:
        if refusal_check["expected_refusal"] and not refusal_check["actual_refusal"]:
            return "Guardrail (Etapa 3): deveria ter recusado (LGPD/fora de escopo) e respondeu normalmente."
        if not refusal_check["expected_refusal"] and refusal_check["actual_refusal"]:
            actual_reason = refusal_check["actual_refusal_reason"]
            if actual_reason == "sem_evidencia":
                return (
                    "Recuperação (Etapa 1/2): o LLM recusou por falta de evidência porque o "
                    "contexto recebido não continha os documentos certos. Verificar se o Query "
                    "Analyzer extraiu um filtro que não bate com nenhum valor real do índice "
                    "(ex.: valor não normalizado) - isso derruba a busca filtrada pra 0 e joga "
                    "pro fallback híbrido sem filtro, que nem sempre acha o chunk certo."
                )
            if actual_reason == "lgpd":
                return "Guardrail (Etapa 3): falso positivo do classify_question() - sinalizou como sensível uma pergunta legítima."
            if actual_reason == "fora_de_escopo":
                return "Heurística de fora de escopo (Etapa 3): threshold de distância L2 classificou errado uma pergunta legítima como fora do domínio."
            return f"Recusou indevidamente com motivo '{actual_reason}' - investigar manualmente."
        return "Guardrail (Etapa 3): recusou, mas com o motivo (refusal_reason) errado."

    ctx = record.get("context_relevance")
    if ctx is not None and ctx < 0.5:
        return "Recuperação (Etapa 1 ingestão ou Etapa 2 busca/filtro): o contexto recuperado não continha os documentos esperados."

    judge = record.get("judge")
    if judge is not None:
        if judge["groundedness"]["score"] < 4:
            return "Síntese (Etapa 3): resposta contém afirmações não sustentadas pelo contexto recuperado (possível alucinação)."
        if judge["answer_relevance"]["score"] < 4:
            return "Síntese (Etapa 3): a resposta não atende diretamente ao que foi perguntado, mesmo com contexto adequado."

    recall = record.get("sources_recall")
    if recall is not None and recall < 1.0:
        return "Citação (Etapa 3): resposta correta, mas cita fonte incompleta ou parcialmente errada."

    return "Não foi possível classificar automaticamente - revisar manualmente."


def _load_benchmark(limit=None, ids=None):
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]
    if ids:
        wanted = set(ids)
        questions = [q for q in questions if q["id"] in wanted]
    if limit is not None:
        questions = questions[:limit]

    return data, questions


def _call_with_rate_limit_retry(func, *args, max_attempts=4, backoff_seconds=12, **kwargs):
    """O tier gratuito da Groq tem um limite baixo de tokens/minuto (TPM); espera e
    tenta de novo em vez de contar essa pergunta como erro por causa disso."""
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except RateLimitError:
            if attempt == max_attempts:
                raise
            print(f"    (rate limit da API, aguardando {backoff_seconds}s antes de tentar de novo...)")
            time.sleep(backoff_seconds)


def _run_question(q, vectorstore, analyzer, filtered_search, hybrid_retriever, skip_judge):
    expected_refusal, expected_refusal_reason = _infer_expected_refusal(q["ground_truth_answer"])

    record = {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "expected_sources": q["expected_sources"],
        "expected_metadata": q["expected_metadata"],
        "ground_truth_answer": q["ground_truth_answer"],
        "key_points_for_evaluation": q["key_points_for_evaluation"],
        "expected_refusal": expected_refusal,
        "error": None,
        "generation": None,
        "filters_used": None,
        "sources_recall": None,
        "context_relevance": None,
        "refusal_check": None,
        "judge": None,
        "pass": False,
        "score": 0.0,
        "diagnosis": None,
    }

    t0 = time.perf_counter()
    try:
        response = _call_with_rate_limit_retry(
            answer_question, q["question"], vectorstore, analyzer, filtered_search, hybrid_retriever
        )
    except Exception as error:
        record["error"] = f"Erro na geração: {error}"
        return record
    generation_latency = time.perf_counter() - t0

    record["generation"] = {
        "answer": response.answer,
        "confidence_level": response.confidence_level,
        "is_refusal": response.is_refusal,
        "refusal_reason": response.refusal_reason,
        "sources_used": [s.model_dump() for s in response.sources_used],
        "reasoning": response.reasoning,
        "latency_seconds": round(generation_latency, 3),
    }

    if expected_refusal or response.is_refusal:
        refusal_check = {
            "expected_refusal": expected_refusal,
            "actual_refusal": response.is_refusal,
            "correct": expected_refusal == response.is_refusal,
            "expected_refusal_reason": expected_refusal_reason,
            "actual_refusal_reason": response.refusal_reason,
            "refusal_reason_correct": (
                response.refusal_reason == expected_refusal_reason
                if expected_refusal and response.is_refusal else None
            ),
        }
        record["refusal_check"] = refusal_check
        record["pass"] = refusal_check["correct"]
        record["score"] = _question_score(record)
        record["diagnosis"] = _diagnose_failure(record) if not refusal_check["correct"] else None
        return record

    record["sources_recall"] = _sources_recall(q["expected_sources"], response.sources_used)

    if skip_judge:
        record["pass"] = None
        return record

    try:
        docs, filters_used = retrieve(q["question"], vectorstore, analyzer, filtered_search, hybrid_retriever)
        record["filters_used"] = filters_used
        record["context_relevance"] = _context_relevance(q["expected_sources"], docs)
        context_text = _context_for_judge(response, docs)

        t1 = time.perf_counter()
        judge = _call_with_rate_limit_retry(
            call_judge, q["question"], context_text, response.answer,
            q["ground_truth_answer"], q["key_points_for_evaluation"],
        )
        judge_latency = time.perf_counter() - t1

        record["judge"] = {
            "groundedness": judge.groundedness.model_dump(),
            "answer_relevance": judge.answer_relevance.model_dump(),
            "key_points_coverage": judge.key_points_coverage.model_dump(),
            "overall_correct": judge.overall_correct,
            "overall_justification": judge.overall_justification,
            "latency_seconds": round(judge_latency, 3),
        }
        record["pass"] = judge.overall_correct
        record["score"] = _question_score(record)
        if not record["pass"]:
            record["diagnosis"] = _diagnose_failure(record)
    except Exception as error:
        record["error"] = f"Erro no juiz: {error}"
        record["pass"] = False
        record["score"] = 0.0
        record["diagnosis"] = _diagnose_failure(record)

    return record


def run_benchmark(limit=None, ids=None, skip_judge=False):
    if not config.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY não configurada. Defina-a no arquivo .env antes de rodar o benchmark."
        )

    data, questions = _load_benchmark(limit=limit, ids=ids)

    print("Carregando índice FAISS...")
    vectorstore = load_index()
    analyzer = QueryAnalyzer(vectorstore)
    filtered_search = FilteredVectorSearch(vectorstore)
    hybrid_retriever = HybridRetriever(vectorstore)

    total = len(questions)
    results = []
    print(f"\nRodando {total} pergunta(s) do benchmark...\n")

    for i, q in enumerate(questions, start=1):
        record = _run_question(q, vectorstore, analyzer, filtered_search, hybrid_retriever, skip_judge)
        results.append(record)

        if record["error"]:
            status = f"ERRO: {record['error']}"
        elif record["pass"] is None:
            status = "gerado (juiz pulado)"
        elif record["pass"]:
            status = "PASS"
        else:
            status = "FAIL"

        print(f"[{i}/{total}] {record['id']} ({record['category']}) - {status}")

    run_metadata = {
        "benchmark_name": data.get("benchmark_name"),
        "benchmark_version": data.get("version"),
        "generated_at": datetime.now().astimezone().isoformat(),
        "generation_model": config.GENERATION_MODEL,
        "judge_model": config.GENERATION_MODEL,
        "out_of_scope_score_threshold": config.OUT_OF_SCOPE_SCORE_THRESHOLD,
        "total_questions": total,
        "skip_judge": skip_judge,
    }

    _write_results_json(run_metadata, results)
    _write_relatorio(run_metadata, results, data)

    return run_metadata, results


def _write_results_json(run_metadata, results):
    payload = {"run_metadata": run_metadata, "results": results}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em: {RESULTS_PATH}")


def _mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _build_relatorio_markdown(run_metadata, results, benchmark_data):
    lines = []
    lines.append(f"# Relatório do Benchmark - {run_metadata['benchmark_name']}")
    lines.append("")
    lines.append(f"- Executado em: {run_metadata['generated_at']}")
    lines.append(f"- Modelo de geração: {run_metadata['generation_model']}")
    lines.append(f"- Modelo do juiz: {run_metadata['judge_model']}")
    lines.append(f"- Threshold de fora-de-escopo: {run_metadata['out_of_scope_score_threshold']}")
    lines.append(f"- Total de perguntas executadas: {run_metadata['total_questions']}")
    lines.append("")

    declared_description = benchmark_data.get("description", "")
    actual_count = len(benchmark_data.get("questions", []))
    lines.append("## Nota sobre o arquivo de benchmark")
    lines.append("")
    lines.append(
        f"A descrição do arquivo `benchmark/questions_and_ground_truth.json` (fornecido "
        f"oficialmente pelo professor) diz: \"{declared_description}\", mas o array "
        f"`questions` contém **{actual_count} perguntas**. Essa inconsistência já vem "
        f"no material original - não foi alterada por nós. Registramos aqui por "
        f"transparência, e rodamos o benchmark com as {actual_count} perguntas "
        f"efetivamente presentes no arquivo."
    )
    lines.append("")

    errors = [r for r in results if r["error"]]
    judged = [r for r in results if r["judge"]]
    passed = [r for r in results if r["pass"] is True]
    failed_or_error = [r for r in results if r["pass"] is not True]
    total_score = sum(r["score"] for r in results)
    max_score = len(results)

    lines.append("## Resumo agregado")
    lines.append("")
    lines.append(f"- **Pontuação total (rubrica oficial): {total_score:.2f} / {max_score:.1f} pontos "
                  f"({total_score / max_score:.1%})**")
    lines.append(f"- Perguntas com PASS: {len(passed)}/{len(results)}")
    lines.append(f"- Perguntas com FAIL ou erro: {len(failed_or_error)}/{len(results)}")
    lines.append(f"- Erros de execução: {len(errors)}")

    refusal_checks = [r["refusal_check"] for r in results if r["refusal_check"]]
    if refusal_checks:
        refusal_correct = sum(1 for rc in refusal_checks if rc["correct"])
        lines.append(f"- Acurácia de recusa: {refusal_correct}/{len(refusal_checks)}")

    ctx_values = [r["context_relevance"] for r in results if r["context_relevance"] is not None]
    if ctx_values:
        lines.append(f"- Context Relevance (determinístico, % de fontes esperadas recuperadas): {_mean(ctx_values):.1%}")
    if judged:
        ground_mean = _mean([r["judge"]["groundedness"]["score"] for r in judged])
        ans_mean = _mean([r["judge"]["answer_relevance"]["score"] for r in judged])
        lines.append(f"- Groundedness (LLM-as-judge, média 1-5): {ground_mean:.2f}")
        lines.append(f"- Answer Relevance (LLM-as-judge, média 1-5): {ans_mean:.2f}")
    lines.append("")

    lines.append("## Detalhamento por categoria")
    lines.append("")
    lines.append("| Categoria | N | PASS | FAIL/erro | Pontuação média | Context Rel. | Groundedness | Answer Rel. |")
    lines.append("|---|---|---|---|---|---|---|---|")

    categories = []
    for r in results:
        if r["category"] not in categories:
            categories.append(r["category"])

    for category in categories:
        cat_results = [r for r in results if r["category"] == category]
        cat_pass = sum(1 for r in cat_results if r["pass"] is True)
        cat_fail = len(cat_results) - cat_pass
        cat_score = _mean([r["score"] for r in cat_results])
        cat_ctx = _mean([r["context_relevance"] for r in cat_results if r["context_relevance"] is not None])
        cat_judged = [r for r in cat_results if r["judge"]]
        ground = _mean([r["judge"]["groundedness"]["score"] for r in cat_judged])
        ans = _mean([r["judge"]["answer_relevance"]["score"] for r in cat_judged])
        score_s = f"{cat_score:.2f}" if cat_score is not None else "-"
        ctx_s = f"{cat_ctx:.0%}" if cat_ctx is not None else "-"
        ground_s = f"{ground:.2f}" if ground is not None else "-"
        ans_s = f"{ans:.2f}" if ans is not None else "-"
        lines.append(f"| {category} | {len(cat_results)} | {cat_pass} | {cat_fail} | {score_s} | {ctx_s} | {ground_s} | {ans_s} |")
    lines.append("")

    lines.append("## Detalhamento por pergunta")
    lines.append("")
    lines.append("| ID | Categoria | Status | Pontuação | Recusa (esp./real) | Confiança | Sources recall |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        status = "ERRO" if r["error"] else ("PASS" if r["pass"] else "FAIL")
        expected_r = "sim" if r["expected_refusal"] else "não"
        if r["generation"]:
            actual_r = "sim" if r["generation"]["is_refusal"] else "não"
            confidence = r["generation"]["confidence_level"]
        else:
            actual_r = "-"
            confidence = "-"
        recall = f"{r['sources_recall']:.0%}" if r["sources_recall"] is not None else "-"
        lines.append(f"| {r['id']} | {r['category']} | {status} | {r['score']:.2f} | {expected_r}/{actual_r} | {confidence} | {recall} |")
    lines.append("")

    if errors:
        lines.append("## Falhas e erros de execução")
        lines.append("")
        for r in errors:
            lines.append(f"- **{r['id']}** ({r['category']}): {r['error']}")
        lines.append("")

    lines.append("## As 3 piores falhas (diagnóstico automático)")
    lines.append("")
    lines.append(
        "> ⚠️ Diagnóstico gerado por heurística a partir dos dados desta execução "
        "(ver `_diagnose_failure` em `eval/run_benchmark.py`) - é um ponto de partida "
        "real, não uma análise definitiva. A dupla deve revisar cada uma manualmente "
        "antes da defesa técnica, porque a arguição vai perguntar a causa raiz de "
        "verdade, não a heurística."
    )
    lines.append("")
    worst = sorted(results, key=lambda r: r["score"])[:3]
    for r in worst:
        lines.append(f"### {r['id']} - {r['category']} (pontuação: {r['score']:.2f})")
        lines.append("")
        lines.append(f"- **Pergunta:** {r['question']}")
        lines.append(f"- **Diagnóstico automático:** {r['diagnosis'] or 'N/A'}")
        if r["judge"]:
            lines.append(f"- **Justificativa do juiz (overall):** {r['judge']['overall_justification']}")
        if r["refusal_check"]:
            lines.append(
                f"- **Recusa esperada/real:** {r['refusal_check']['expected_refusal']} / "
                f"{r['refusal_check']['actual_refusal']} "
                f"(motivo esperado: {r['refusal_check']['expected_refusal_reason']}, "
                f"motivo real: {r['refusal_check']['actual_refusal_reason']})"
            )
        lines.append("- **O que a dupla investigou / causa raiz real:** _(preencher manualmente)_")
        lines.append("")

    lines.append("## O que faríamos com mais 4 horas")
    lines.append("")
    lines.append(
        "_(Esta seção precisa ser escrita pela dupla com base no diagnóstico acima - "
        "não é gerada automaticamente, porque é uma reflexão de vocês sobre "
        "prioridade de engenharia, e vocês serão arguidos sobre isso no Demo Day.)_"
    )
    lines.append("")
    lines.append("Perguntas-guia para responder aqui:")
    lines.append("- Das 3 piores falhas acima, qual etapa (1, 2 ou 3) concentra mais problemas?")
    lines.append("- Isso é um problema de chunking/indexação, de recuperação (filtro/híbrida), ou de geração/guardrail?")
    lines.append("- Se só desse pra consertar UMA coisa, qual teria o maior impacto na pontuação total?")
    lines.append("")

    lines.append("## Limitações conhecidas")
    lines.append("")
    lines.append(
        "- O juiz LLM usa o mesmo modelo de geração (viés de auto-avaliação, "
        "conhecido em setups de LLM-as-judge)."
    )
    lines.append(
        "- O threshold de fora-de-escopo foi calibrado empiricamente (ver "
        "`src/check_threshold.py`) e pode gerar falsos positivos/negativos em "
        "perguntas de fronteira."
    )
    lines.append(
        "- Context Relevance é calculada no nível de ARQUIVO, não de chunk_id: o "
        "gabarito oficial (`expected_sources`) não fornece chunk_id esperado, "
        "então comparamos o arquivo de origem dos chunks recuperados com o "
        "arquivo esperado (ver `_context_relevance` em `eval/run_benchmark.py`)."
    )
    lines.append(
        "- A pontuação por questão (0.5/0.3/0.2) segue a rubrica do enunciado, mas "
        "o enunciado não especifica a fórmula exata de partial credit para os "
        "componentes de citação e coerência; a fórmula usada está documentada em "
        "`_question_score` (`eval/run_benchmark.py`)."
    )
    lines.append("")

    return "\n".join(lines)


def _write_relatorio(run_metadata, results, benchmark_data):
    markdown = _build_relatorio_markdown(run_metadata, results, benchmark_data)
    with open(RELATORIO_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Relatório salvo em: {RELATORIO_PATH}")


def regenerate_report_from_cache():
    """
    Relê eval/results.json (já existente, de uma execução anterior) e regera
    RELATORIO.md - sem chamar a API nenhuma vez. Útil depois de qualquer ajuste
    só na lógica de pontuação/diagnóstico (ex.: _question_score, _diagnose_failure),
    pra não gastar cota de novo só pra atualizar o texto do relatório.
    """
    if not os.path.exists(RESULTS_PATH):
        raise RuntimeError(
            f"Não existe {RESULTS_PATH} ainda - rode o benchmark completo pelo menos "
            f"uma vez antes de usar --from-results."
        )

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    run_metadata = payload["run_metadata"]
    results = payload["results"]

    # Recalcula score e diagnosis com a lógica ATUAL do código (pode já ter
    # mudado desde que os resultados foram gerados).
    for record in results:
        record["score"] = _question_score(record)
        if not record["pass"]:
            record["diagnosis"] = _diagnose_failure(record)
        else:
            record["diagnosis"] = None

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"run_metadata": run_metadata, "results": results}, f, ensure_ascii=False, indent=2)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    _write_relatorio(run_metadata, results, benchmark_data)
    print("results.json e RELATORIO.md regenerados a partir do cache (nenhuma chamada de API foi feita).")


def main():
    parser = argparse.ArgumentParser(description="Roda o benchmark do VendeFácil RAG.")
    parser.add_argument("--limit", type=int, default=None, help="Roda só as N primeiras perguntas.")
    parser.add_argument("--ids", type=str, default=None, help="IDs separados por vírgula (ex: Q01,Q05).")
    parser.add_argument("--skip-judge", action="store_true", help="Pula a chamada ao juiz LLM.")
    parser.add_argument(
        "--from-results", action="store_true",
        help="Não chama a API - só relê eval/results.json existente e regera RELATORIO.md.",
    )
    args = parser.parse_args()

    if args.from_results:
        regenerate_report_from_cache()
        return

    ids = [i.strip() for i in args.ids.split(",")] if args.ids else None
    run_benchmark(limit=args.limit, ids=ids, skip_judge=args.skip_judge)


if __name__ == "__main__":
    main()
