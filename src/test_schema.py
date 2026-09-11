from pydantic import ValidationError

from schema import RAGResponse


def test_case(name: str, data: dict):
    print("\n" + "=" * 70)
    print(name)

    try:
        response = RAGResponse.model_validate(data)

        print("VALIDAÇÃO: OK")
        print(response.model_dump())

    except ValidationError as error:
        print("VALIDAÇÃO: ERRO ESPERADO")
        print(error)


# ============================================================
# 1. Resposta normal válida
# ============================================================

test_case(
    "1 - RESPOSTA NORMAL VÁLIDA",
    {
        "answer": (
            "O reembolso pode ser solicitado conforme "
            "a política interna."
        ),
        "confidence_level": "alta",
        "sources_used": [
            {
                "filepath": "policies/reembolso.pdf",
                "chunk_id": "pdf-reembolso-1",
                "quotation": (
                    "O cliente poderá solicitar o reembolso "
                    "conforme as condições previstas."
                )
            }
        ],
        "reasoning": (
            "A resposta está sustentada diretamente "
            "pelo documento."
        ),
        "is_refusal": False,
        "refusal_reason": None
    }
)


# ============================================================
# 2. Recusa válida
# ============================================================

test_case(
    "2 - RECUSA VÁLIDA",
    {
        "answer": "Não posso fornecer esse dado.",
        "confidence_level": "recusado",
        "sources_used": [],
        "reasoning": (
            "A solicitação envolve dado protegido."
        ),
        "is_refusal": True,
        "refusal_reason": "lgpd"
    }
)


# ============================================================
# 3. Resposta sem evidência - deve falhar
# ============================================================

test_case(
    "3 - SEM EVIDÊNCIA",
    {
        "answer": "O reembolso é permitido.",
        "confidence_level": "alta",
        "sources_used": [],
        "reasoning": "Resposta sem fonte.",
        "is_refusal": False,
        "refusal_reason": None
    }
)


# ============================================================
# 4. Recusa com confiança errada - deve falhar
# ============================================================

test_case(
    "4 - RECUSA COM CONFIANÇA ERRADA",
    {
        "answer": "Não posso fornecer esse dado.",
        "confidence_level": "alta",
        "sources_used": [],
        "reasoning": "Dado protegido.",
        "is_refusal": True,
        "refusal_reason": "lgpd"
    }
)


# ============================================================
# 5. Confidence level inválido - deve falhar
# ============================================================

test_case(
    "5 - CONFIDENCE_LEVEL INVÁLIDO",
    {
        "answer": "Resposta qualquer.",
        "confidence_level": "muito alta",
        "sources_used": [
            {
                "filepath": "manual.md",
                "chunk_id": "manual-1",
                "quotation": "Trecho de evidência."
            }
        ],
        "reasoning": "Teste.",
        "is_refusal": False,
        "refusal_reason": None
    }
)


# ============================================================
# 6. Citação acima de 500 caracteres - deve falhar
# ============================================================

test_case(
    "6 - QUOTATION MAIOR QUE 500",
    {
        "answer": "Resposta de teste.",
        "confidence_level": "alta",
        "sources_used": [
            {
                "filepath": "manual.md",
                "chunk_id": "manual-1",
                "quotation": "A" * 501
            }
        ],
        "reasoning": (
            "Teste do limite máximo da citação."
        ),
        "is_refusal": False,
        "refusal_reason": None
    }
)


# ============================================================
# 7. Recusa sem motivo - deve falhar
# ============================================================

test_case(
    "7 - RECUSA SEM MOTIVO",
    {
        "answer": "Não posso responder.",
        "confidence_level": "recusado",
        "sources_used": [],
        "reasoning": "Teste de consistência.",
        "is_refusal": True,
        "refusal_reason": None
    }
)


# ============================================================
# 8. Resposta normal com refusal_reason - deve falhar
# ============================================================

test_case(
    "8 - RESPOSTA NORMAL COM MOTIVO DE RECUSA",
    {
        "answer": "Resposta baseada no documento.",
        "confidence_level": "alta",
        "sources_used": [
            {
                "filepath": "manual.md",
                "chunk_id": "manual-2",
                "quotation": (
                    "Este é um trecho literal utilizado "
                    "como evidência."
                )
            }
        ],
        "reasoning": "Existe evidência suficiente.",
        "is_refusal": False,
        "refusal_reason": "lgpd"
    }
)