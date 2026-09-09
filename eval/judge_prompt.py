import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pydantic import BaseModel, Field, ValidationError

import config
from generate import _call_llm

JUDGE_SYSTEM_PROMPT = """\
Você é um avaliador (juiz) especializado em sistemas RAG (Retrieval-Augmented
Generation), avaliando a qualidade de uma resposta gerada pelo assistente de IA
interno da VendeFácil Tecnologia Ltda.

Você vai receber: a PERGUNTA original, o CONTEXTO que foi recuperado da base de
conhecimento e usado para gerar a resposta, a RESPOSTA GERADA pelo assistente, uma
RESPOSTA DE REFERÊNCIA (gabarito) e uma lista de PONTOS-CHAVE ESPERADOS.

Avalie em duas frentes:

1) RAG Triad (avalie SOMENTE com base no CONTEXTO e na PERGUNTA/RESPOSTA GERADA -
   NÃO use a resposta de referência para essas notas). Note que "context_relevance"
   NÃO é avaliada por você - ela é calculada de forma determinística fora deste
   juiz, comparando os arquivos de origem recuperados com o gabarito:
   - groundedness: o quanto toda afirmação feita na RESPOSTA GERADA está
     sustentada literalmente pelo CONTEXTO (sem alucinação), de 1 (nenhuma
     afirmação sustentada) a 5 (toda afirmação sustentada). Liste em
     "unsupported_claims" qualquer afirmação da resposta que não encontra
     respaldo no contexto.
   - answer_relevance: o quanto a RESPOSTA GERADA realmente responde à PERGUNTA
     feita, de 1 (não responde) a 5 (responde completamente).

2) Comparação com a referência (aqui SIM use a RESPOSTA DE REFERÊNCIA e os
   PONTOS-CHAVE ESPERADOS):
   - key_points_coverage: para cada ponto-chave esperado, decida se a RESPOSTA
     GERADA cobriu esse ponto (mesmo com palavras diferentes) ou não, e separe em
     "points_hit" e "points_missed" (copie o texto do ponto-chave em cada lista).
   - overall_correct: veredito final (true/false) se a RESPOSTA GERADA é, no
     conjunto, uma resposta aceitável e equivalente em conteúdo à RESPOSTA DE
     REFERÊNCIA para essa pergunta.
   - overall_justification: 1-2 frases explicando o veredito final.

Sua saída deve ser SOMENTE um objeto JSON válido, sem nenhum texto antes ou
depois, sem markdown, seguindo EXATAMENTE este formato. ATENÇÃO: se você
citar ou mencionar um trecho do CONTEXTO que contenha aspas duplas dentro
dele (ex.: "Forçar Reconciliação de Estoque"), você DEVE escapá-las com
barra invertida (\") nas suas justificativas para manter o JSON válido -
isso é uma causa comum de JSON quebrado, preste atenção especial nisso:

{
  "groundedness": {"score": 1-5, "justification": "...", "unsupported_claims": ["..."]},
  "answer_relevance": {"score": 1-5, "justification": "..."},
  "key_points_coverage": {"points_hit": ["..."], "points_missed": ["..."]},
  "overall_correct": true,
  "overall_justification": "..."
}
"""


def build_judge_user_prompt(question: str, context_text: str, answer: str,
                             ground_truth_answer: str, key_points: list[str]) -> str:
    key_points_text = "\n".join(f"- {point}" for point in key_points)
    return (
        f"PERGUNTA:\n{question}\n\n"
        f"CONTEXTO RECUPERADO:\n{context_text}\n\n"
        f"RESPOSTA GERADA:\n{answer}\n\n"
        f"RESPOSTA DE REFERÊNCIA (GABARITO):\n{ground_truth_answer}\n\n"
        f"PONTOS-CHAVE ESPERADOS:\n{key_points_text}"
    )


class DimensionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    justification: str


class GroundednessScore(DimensionScore):
    unsupported_claims: list[str] = Field(default_factory=list)


class KeyPointsCoverage(BaseModel):
    points_hit: list[str]
    points_missed: list[str]


class JudgeResponse(BaseModel):
    groundedness: GroundednessScore
    answer_relevance: DimensionScore
    key_points_coverage: KeyPointsCoverage
    overall_correct: bool
    overall_justification: str


def call_judge(question: str, context_text: str, answer: str, ground_truth_answer: str,
               key_points: list[str], max_retries: int = None) -> JudgeResponse:
    """
    Chama o LLM como juiz e valida a saída com Pydantic. Mesma lógica de retry de
    `generate.generate_structured_response` (reconstrói o prompt original a cada
    tentativa, só anexando o erro de validação anterior).
    """
    import time

    max_retries = max_retries if max_retries is not None else config.MAX_RETRIES

    base_prompt = build_judge_user_prompt(question, context_text, answer, ground_truth_answer, key_points)
    user_prompt = base_prompt
    last_error = None

    for attempt in range(1, max_retries + 1):
        raw = _call_llm(JUDGE_SYSTEM_PROMPT, user_prompt)

        try:
            data = json.loads(raw)
            return JudgeResponse.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as error:
            last_error = error
            # Correção (Etapa 4 - item 20, ajustada): ver comentário
            # equivalente em generate.generate_structured_response - pausa
            # progressiva (não fixa) quando a resposta vem vazia.
            if not raw:
                time.sleep(3 * attempt)
            user_prompt = (
                f"{base_prompt}\n\n"
                f"Sua resposta anterior (tentativa {attempt}) não seguiu o formato "
                f"exigido. Erro de validação: {error}\n"
                f"Responda de novo, SOMENTE com o JSON correto, corrigindo o erro acima."
            )

    raise RuntimeError(
        f"Juiz não conseguiu gerar uma avaliação válida após {max_retries} tentativas. "
        f"Último erro: {last_error}"
    )
