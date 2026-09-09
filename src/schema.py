from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SourceEvidence(BaseModel):
    filepath: str = Field(
        ...,
        description="Arquivo de origem do trecho citado"
    )
    chunk_id: str = Field(
        ...,
        description="Identificador do chunk recuperado"
    )
    quotation: str = Field(
        ...,
        max_length=1200,
        description="Trecho literal que sustenta a resposta"
    )


class RAGResponse(BaseModel):
    answer: str

    confidence_level: Literal[
        "alta",
        "media",
        "baixa",
        "recusado"
    ]

    sources_used: list[SourceEvidence]

    reasoning: str

    is_refusal: bool

    refusal_reason: Literal[
        "lgpd",
        "fora_de_escopo",
        "sem_evidencia"
    ] | None = None

    @model_validator(mode="after")
    def validate_consistency(self):
        """
        Garante a consistência entre recusa,
        confiança, fontes e motivo.
        """

        if self.is_refusal:
            if self.confidence_level != "recusado":
                raise ValueError(
                    "Resposta recusada deve ter confidence_level='recusado'."
                )

            if self.sources_used:
                raise ValueError(
                    "Resposta recusada não pode possuir sources_used."
                )

            if self.refusal_reason is None:
                raise ValueError(
                    "Resposta recusada deve informar refusal_reason."
                )

        else:
            if not self.sources_used:
                raise ValueError(
                    "Resposta não recusada precisa possuir pelo menos uma evidência."
                )

            if self.confidence_level == "recusado":
                raise ValueError(
                    "Resposta não recusada não pode ter confidence_level='recusado'."
                )

            if self.refusal_reason is not None:
                raise ValueError(
                    "Resposta não recusada não deve possuir refusal_reason."
                )

        return self