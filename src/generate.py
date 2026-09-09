import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import json

from pydantic import ValidationError

import config
from hybrid_search import HybridRetriever
from lgpd_policy import (
    classify_question,
    has_only_restricted_docs,
    mask_pii,
    filter_out_leaked_restricted_docs,
)
from query_analyzer import QueryAnalyzer
from query_index import load_index
from schema import RAGResponse, SourceEvidence
from search import FilteredVectorSearch
from domain_keywords import contains_domain_keyword
from structured_query import is_customer_mrr_aggregation, answer_customer_mrr_aggregation

_CUSTOMERS_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "structured", "customers.csv"
)

SYSTEM_PROMPT = """\
Você é o assistente RAG interno da VendeFácil Tecnologia Ltda.

Responda SOMENTE com base no CONTEXTO fornecido pelo usuário. Nunca use
conhecimento próprio fora do CONTEXTO, mesmo que você "saiba" a resposta.

Sua saída deve ser SOMENTE um objeto JSON válido, sem nenhum texto antes ou
depois, sem markdown, seguindo EXATAMENTE este formato. IMPORTANTE: gere os
campos NESTA ORDEM - "is_refusal", "confidence_level" e "reasoning" são
curtos e vêm PRIMEIRO; "sources_used" (que pode ter várias citações longas)
vem por último. Se a resposta tiver muitas fontes para citar, prefira
resumir mais as citações a deixar de fechar o JSON corretamente - um JSON
incompleto é sempre pior do que uma citação mais curta. ATENÇÃO: se o texto
citado em "quotation" contiver aspas duplas (ex.: "Forçar Reconciliação de
Estoque"), você DEVE escapá-las com barra invertida (\") para manter o JSON
válido - isso é uma causa comum de JSON quebrado, preste atenção especial
nisso:

{
  "is_refusal": false,
  "refusal_reason": null,
  "confidence_level": "alta" | "media" | "baixa",
  "answer": "resposta em linguagem natural, objetiva, baseada só no contexto",
  "reasoning": "explicação breve de como você chegou na resposta a partir do contexto",
  "sources_used": [
    {
      "filepath": "valor idêntico ao source_file do chunk usado",
      "chunk_id": "valor idêntico ao chunk_id do chunk usado",
      "quotation": "trecho LITERAL (copiado, não parafraseado) do chunk, até 500 caracteres"
    }
  ]
}

Regras obrigatórias:
- "sources_used" deve ter pelo menos 1 item sempre que "is_refusal" for false.
- Cada "quotation" deve ser um trecho literal do chunk citado (não invente, não resuma).
- Se o CONTEXTO não tiver informação suficiente para responder com segurança,
  responda com "is_refusal": true, "confidence_level": "recusado",
  "sources_used": [], "refusal_reason": "sem_evidencia". Porém, se o contexto tem uma
  pessoa listada com EXATAMENTE o cargo perguntado (ex.: "Tech Lead", "PM") numa reunião
  ou documento sobre o produto/módulo da pergunta, isso conta como evidência suficiente -
  não recuse só porque a associação pessoa-produto não está escrita numa frase única e
  explícita; a lista de participantes com cargo, no contexto de uma reunião sobre aquele
  produto, já é a evidência.
- Nunca marque "confidence_level" como "recusado" se "is_refusal" for false, e vice-versa.
- Se o chunk relevante tiver MAIS DE UM requisito/regra/item sobre o mesmo assunto da
  pergunta (ex.: uma lista numerada de exigências), inclua TODOS os itens dessa lista na
  resposta, não só os primeiros - não resuma parcialmente algo que está completo no contexto.
- Antes de finalizar a resposta, releia CADA "quotation" que você vai citar em "sources_used"
  e confirme que todo fato relevante dela (números, nomes, causas, códigos de erro, passos,
  quantidades) está mencionado na resposta final - não é permitido citar uma evidência sem
  usar o conteúdo relevante dela.
- Ao listar registros que batem com um critério (ex.: "quais tickets são X"), responda com
  TODOS os registros do CONTEXTO que batem com o critério pedido na pergunta - nunca aplique
  um critério adicional de qualquer campo (status, nível/level, data, prioridade, etc.) que não
  foi explicitamente pedido na pergunta, mesmo que ache que é mais relevante, mais recente, ou
  mais "correto" tecnicamente. Se a pergunta pede "logs de erro" e o contexto tem um log
  correlato de nível WARN sobre o mesmo evento/causa, inclua-o também - não decida sozinho que
  só "ERROR" conta como erro.
- Ao citar um trecho de um chunk, NÃO comece a citação no meio de uma seção que já responde à
  pergunta - se as primeiras linhas do chunk têm uma informação de identificação relevante
  (ex.: nome de responsável, cabeçalho com nome/cargo/contato), inclua essas linhas na citação
  e mencione essa informação na resposta, mesmo que a frase central da resposta esteja mais à
  frente no chunk.
- Quando a pergunta pede para ENUMERAR um conjunto de entidades ("quais são os produtos",
  "quais planos existem", "quais módulos..."), responda com CADA item acompanhado de uma
  breve descrição tirada do contexto (o que o item é / para que serve) - não responda só
  com a lista de nomes ou códigos. Ex.: "VendeFácil Estoque (módulo de controle de
  inventário, curva ABC e alertas de ruptura)", não "VendeFácil Estoque (PROD-ESTOQUE)".
"""


def _build_context(docs) -> str:
    blocks = []
    for doc in docs:
        meta = doc.metadata
        header = (
            f"[chunk_id={meta.get('chunk_id')} | source_file={meta.get('source_file')} | "
            f"doc_type={meta.get('doc_type')}]"
        )
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _refusal(reason: str, message: str) -> RAGResponse:
    return RAGResponse(
        answer=message,
        confidence_level="recusado",
        sources_used=[],
        reasoning=(
            "Recusa aplicada antes da chamada ao LLM, com base na política de "
            "LGPD / disponibilidade de evidência (ver lgpd_policy.py)."
        ),
        is_refusal=True,
        refusal_reason=reason,
    )


def _count_distinct_content_types_mentioned(question: str) -> int:
    """
    Conta quantos TIPOS distintos de documento são citados explicitamente na
    pergunta (ex.: "e-mails, tickets e reuniões" = 3). Usado só para decidir
    o tamanho do complemento de busca - não define filtro nenhum.
    """
    from query_analyzer import normalize, contains

    query = normalize(question)
    groups = [
        ["email", "emails", "e-mail", "e-mails"],
        ["ticket", "tickets", "chamado", "chamados"],
        ["reuniao", "reunioes", "ata", "atas"],
        ["log", "logs"],
    ]
    return sum(1 for group in groups if any(contains(query, normalize(w)) for w in group))


def retrieve(question: str, vectorstore, analyzer: QueryAnalyzer,
             filtered_search: FilteredVectorSearch, hybrid_retriever: HybridRetriever,
             k: int = 8):

    filters = analyzer.analyze(question).to_dict()

    if filters:
        _, docs = filtered_search.search(question, k=k, use_filters=True)
        if docs:
            # Correção (Etapa 4 - item 11): quando o filtro trava num único
            # doc_type (ex.: doc_type=ticket), perguntas que precisam de MAIS
            # de uma fonte (ex.: ticket + política de SLA; e-mail + ticket +
            # ata) nunca recebiam a segunda fonte, porque ela tem um doc_type
            # diferente do filtrado. Complementamos com alguns resultados da
            # busca híbrida SEM filtro (que enxerga todos os tipos), sem
            # remover nada do que o filtro já achou - só adiciona.
            #
            # Correção (Etapa 4 - item 16): quando a pergunta CITA
            # explicitamente 2+ tipos de documento (ex.: "e-mails, tickets e
            # reuniões" - Q08), isso é um sinal forte e explícito de que o
            # usuário quer múltiplas fontes, e não só 3 documentos extras -
            # ampliamos bastante o teto do complemento nesse caso específico.
            #
            # Correção (Etapa 4 - item 21): pendência antiga do Encontro 2 -
            # até aqui só o caso "filtro trava num doc_type" disparava o
            # complemento. Um filtro SEM doc_type (ex.: só customer_id ou só
            # module) que bate certinho mas devolve poucos chunks (< k) fica
            # tão pobre de contexto quanto o caso do item 11, só que nunca
            # passava pelo complemento porque a condição só olhava doc_type.
            # Generalizamos: complementa sempre que o doc_type está travado
            # OU o resultado filtrado já veio abaixo do k pedido.
            if filters.get("doc_type") or len(docs) < k:
                distinct_types = _count_distinct_content_types_mentioned(question)
                # Correção (Etapa 4 - item 25): quando a pergunta cita 2+ tipos
                # de documento (Q08: "e-mails, tickets e reuniões"), a fonte que
                # fecha a resposta costuma estar FUNDO no ranking híbrido cru -
                # a ata específica da Q08 (`2026-01-product_roadmap`) fica no
                # rank ~14, atrás de atas genéricas sobre "supermercado"/"MRR".
                # O pool antigo (k+5=13) e o teto (k+10=18) não a alcançavam.
                # Ampliamos os dois SÓ nesse caso; perguntas de fonte única não
                # mudam. (A 4ª fonte da Q08, `sales_enterprise_feedback` no rank
                # ~29, ainda exige um 2º hop dirigido - ver RELATORIO.md.)
                if distinct_types >= 2:
                    extra_cap = k + 14
                    fused_extra = hybrid_retriever.hybrid_search(question, k=k + 16)
                else:
                    extra_cap = k + 3
                    fused_extra = hybrid_retriever.hybrid_search(question, k=k + 5)
                seen_ids = {d.metadata.get("chunk_id") for d in docs}
                for extra_doc, _score in fused_extra:
                    if len(docs) >= extra_cap:
                        break
                    cid = extra_doc.metadata.get("chunk_id")
                    if cid not in seen_ids:
                        docs.append(extra_doc)
                        seen_ids.add(cid)
            # Correção (Etapa 4 - item 22): achado real da Q08 (ver
            # RELATORIO.md) - o complemento acima busca por similaridade da
            # pergunta inteira, sem saber que um chunk trazido é "restrito" e
            # tem conteúdo de credencial (senha, chave de API, token). Filtra
            # isso ANTES de devolver o contexto, independente da pergunta ter
            # pedido algo sensível ou não - terceira camada de defesa, por
            # conteúdo do chunk, não só por intenção da pergunta.
            return filter_out_leaked_restricted_docs(docs), filters
        # Combinação de filtros não bateu com nenhum chunk (ex.: campos que
        # não coexistem no mesmo documento) - cai para a busca híbrida sem
        # filtro em vez de recusar por falta de evidência.

    fused = hybrid_retriever.hybrid_search(question, k=k)
    docs = [doc for doc, _score in fused]
    return filter_out_leaked_restricted_docs(docs), filters


def is_out_of_scope(question: str, vectorstore, threshold: float = None) -> bool:

    if contains_domain_keyword(question):
        return False

    threshold = threshold if threshold is not None else config.OUT_OF_SCOPE_SCORE_THRESHOLD
    results = vectorstore.similarity_search_with_score(question, k=1)
    if not results:
        return True
    _, score = results[0]
    return score > threshold


def generate_structured_response(question: str, context: str,
                                  max_retries: int = None) -> RAGResponse:
    """
    Chama o LLM pedindo saída no formato RAGResponse e valida com Pydantic.
    Em caso de falha de validação, pede pro próprio modelo corrigir (retry),
    em vez de silenciar o erro com `except: pass`.
    """
    import time

    max_retries = max_retries if max_retries is not None else config.MAX_RETRIES

    user_prompt = f"PERGUNTA: {question}\n\nCONTEXTO:\n{context}"
    last_error = None

    for attempt in range(1, max_retries + 1):
        raw = _call_llm(SYSTEM_PROMPT, user_prompt)

        try:
            data = json.loads(raw)
            return RAGResponse.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as error:
            last_error = error
            # Correção (Etapa 4 - item 20): confirmado rodando com
            # openrouter/free - quando a resposta vem vazia (não é erro de
            # formato, é ausência total de conteúdo), reenviar o MESMO prompt
            # na hora costumava sortear o mesmo modelo problemático de novo.
            # Uma pequena pausa real (não só reformular o prompt) dá chance
            # do roteador escolher outro modelo gratuito na próxima tentativa.
            # Correção (Etapa 4 - item 20, ajustada): pausa progressiva (não
            # fixa) quando a resposta vem vazia - 3 tentativas seguidas com
            # 3s fixos ainda caíam no mesmo problema (Q07, Q08), sinal de que
            # o roteador às vezes demora mais que 3s pra variar de modelo.
            if not raw:
                time.sleep(3 * attempt)
            user_prompt = (
                f"PERGUNTA: {question}\n\nCONTEXTO:\n{context}\n\n"
                f"Sua resposta anterior (tentativa {attempt}) não seguiu o formato "
                f"exigido. Erro de validação: {error}\n"
                f"Responda de novo, SOMENTE com o JSON correto, corrigindo o erro acima."
            )

    raise RuntimeError(
        f"Não foi possível gerar uma resposta válida após {max_retries} tentativas. "
        f"Último erro: {last_error}"
    )


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Isolado numa função própria para ser fácil de trocar de provedor/mocar em teste."""
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
    response = client.chat.completions.create(
        model=config.GENERATION_MODEL,
        temperature=0,
        max_tokens=4096,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content

    # Correção (Etapa 4 - item 17): confirmado rodando com openrouter/free -
    # como o roteador sorteia um modelo gratuito diferente a cada chamada,
    # às vezes o provedor devolve HTTP 200 com o conteúdo da mensagem vazio
    # (None), em vez de erro. Isso não acontecia com um único modelo fixo
    # (Groq). `json.loads(None)` derruba um TypeError que o retry de
    # generate_structured_response não pega (só pega JSONDecodeError e
    # ValidationError), então o pipeline falhava direto em vez de tentar de
    # novo. Devolvendo string vazia aqui, o `json.loads("")` gera um
    # JSONDecodeError normal, que já é tratado pelo retry existente.
    #
    # Correção (Etapa 4 - item 19): também confirmado rodando de verdade -
    # sem `max_tokens` explícito, um juiz gerando uma avaliação longa (várias
    # seções de justificativa) às vezes tinha a saída cortada no meio pelo
    # limite padrão do modelo gratuito sorteado, virando um JSON incompleto
    # (erro tipo "Expecting ',' delimiter"). Um `max_tokens` explícito e
    # generoso reduz a chance disso acontecer.
    return content if content else ""


def _apply_masking(response: RAGResponse) -> RAGResponse:
    masked_sources = [
        SourceEvidence(
            filepath=source.filepath,
            chunk_id=source.chunk_id,
            quotation=mask_pii(source.quotation),
        )
        for source in response.sources_used
    ]
    return RAGResponse(
        answer=mask_pii(response.answer),
        confidence_level=response.confidence_level,
        sources_used=masked_sources,
        reasoning=response.reasoning,
        is_refusal=response.is_refusal,
        refusal_reason=response.refusal_reason,
    )


def answer_question(question: str, vectorstore, analyzer: QueryAnalyzer,
                     filtered_search: FilteredVectorSearch,
                     hybrid_retriever: HybridRetriever) -> RAGResponse:
    """Função principal: pergunta em texto livre -> RAGResponse validado."""

    category = classify_question(question)

    # Correção (Etapa 4 - item 15): perguntas de agregação (MAX/MIN de MRR)
    # são resolvidas por consulta estruturada direta sobre o CSV real, ANTES
    # de qualquer busca vetorial ou chamada de LLM. Ver structured_query.py
    # para a justificativa completa - busca por similaridade não é a
    # ferramenta certa pra "qual é o maior valor de X", e isso foi confirmado
    # rodando o benchmark de verdade (Q10).
    aggregation_spec = is_customer_mrr_aggregation(question)
    if aggregation_spec:
        result = answer_customer_mrr_aggregation(aggregation_spec, _CUSTOMERS_CSV_PATH)
        if result:
            op_word = "maior" if aggregation_spec["op"] == "max" else "menor"
            answer_text = (
                f"O cliente com {op_word} MRR"
                + (f" no estado de {aggregation_spec['state']}" if aggregation_spec["state"] else "")
                + f" é {result['company_name']} ({result['customer_id']}), com MRR de "
                f"R$ {result['mrr']:.2f}, cujo produto principal é {result['main_product']}."
            )
            return RAGResponse(
                answer=answer_text,
                confidence_level="alta",
                is_refusal=False,
                refusal_reason=None,
                sources_used=[
                    SourceEvidence(
                        filepath="structured/customers.csv",
                        chunk_id=f"customer-{result['customer_id']}",
                        quotation=result["quotation"],
                    )
                ],
                reasoning=(
                    "Resolvida por consulta estruturada direta sobre customers.csv "
                    "(agregação MAX/MIN não passa por busca vetorial nem por LLM, "
                    "para garantir exatidão numérica)."
                ),
            )
        # Se a consulta estruturada não achou nenhuma linha (ex.: estado sem
        # nenhum cliente), cai pro fluxo normal em vez de travar aqui.

    if is_out_of_scope(question, vectorstore):
        return _refusal(
            "fora_de_escopo",
            "Essa pergunta não está relacionada à operação da VendeFácil, "
            "então não posso respondê-la com base na minha base de conhecimento.",
        )

    docs, _filters = retrieve(question, vectorstore, analyzer, filtered_search, hybrid_retriever)

    if not docs:
        return _refusal(
            "sem_evidencia",
            "Não encontrei nenhum documento relevante na base para responder a essa pergunta.",
        )

    if category == "recusar":
        return _refusal(
            "lgpd",
            "Não posso fornecer esse dado, pois envolve informação pessoal protegida "
            "pela LGPD (ex.: remuneração individual, CPF, dados bancários, "
            "credenciais ou dados de saúde).",
        )

    if has_only_restricted_docs(docs):
        # Correção (Etapa 4 - item 9): antes de recusar, tenta de novo com
        # busca híbrida SEM filtro de metadados. Motivo real, confirmado no
        # benchmark: um filtro de doc_type mal extraído (ex.: doc_type=
        # employee) podia trazer só chunks de employees.csv (sensitivity=
        # restrito), disparando esta recusa mesmo para perguntas totalmente
        # inofensivas (ex.: "política de home office da Engenharia") cuja
        # resposta certa está em outro documento, não-restrito. Corrigimos
        # a extração de filtros na origem (query_analyzer.py), mas mantemos
        # esta segunda tentativa como rede de segurança para qualquer caso
        # que a heurística de filtro não previu - só recusamos de fato se a
        # busca ampla (sem filtro nenhum) também só achar documentos restritos.
        fused = hybrid_retriever.hybrid_search(question, k=8)
        unfiltered_docs = [doc for doc, _score in fused]
        if unfiltered_docs and not has_only_restricted_docs(unfiltered_docs):
            docs = unfiltered_docs
        else:
            return _refusal(
                "lgpd",
                "Não posso fornecer esse dado, pois envolve informação pessoal protegida "
                "pela LGPD (ex.: remuneração individual, CPF, dados bancários, "
                "credenciais ou dados de saúde).",
            )

    context = _build_context(docs)
    response = generate_structured_response(question, context)

    if category == "mascarar" and not response.is_refusal:
        response = _apply_masking(response)

    return response


if __name__ == "__main__":
    vectorstore = load_index()
    analyzer = QueryAnalyzer(vectorstore)
    filtered_search = FilteredVectorSearch(vectorstore)
    hybrid_retriever = HybridRetriever(vectorstore)

    # >= 2 perguntas por categoria, conforme critério de pronto da Etapa 3.
    TEST_QUESTIONS = [
        # RECUSAR (LGPD)
        "Qual o salário do funcionário com maior remuneração?",
        "Quanto a folha da equipe de suporte custa por pessoa?",
        # MASCARAR
        "Qual o e-mail de contato do cliente CUST001?",
        "Qual o telefone de contato registrado para o cliente CUST014?",
        # RESPONDER
        "Qual é a política de reembolso da empresa?",
        "Como funciona a sincronização de estoque entre lojas?",
        # FORA DE ESCOPO
        "Quem descobriu o Brasil?",
        "Me escreva um poema sobre o outono.",
    ]

    for question in TEST_QUESTIONS:
        print("\n" + "=" * 80)
        print(f"PERGUNTA: {question}")
        try:
            response = answer_question(
                question, vectorstore, analyzer, filtered_search, hybrid_retriever
            )
            print(response.model_dump_json(indent=2, exclude_none=False))
        except Exception as error:
            print(f"ERRO ao gerar resposta: {error}")
