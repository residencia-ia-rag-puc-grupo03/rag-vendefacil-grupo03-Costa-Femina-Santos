"""
Configuração centralizada do projeto (credenciais e parâmetros).

Lê tudo de variáveis de ambiente (via `.env`, que NUNCA deve ser commitado -
veja `.gitignore` e `.env.example`), conforme exigido na seção 6 do guia do
desafio ("config.py - credenciais e parâmetros centralizados").
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- LLM de síntese (src/generate.py) --------------------------------------
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Roteamento por prefixo do nome do modelo: quando GENERATION_MODEL é um
# Gemini (ex.: "gemini-3.5-flash-lite"), o código usa a GEMINI_API_KEY e o
# endpoint OpenAI-compatível do Google automaticamente - sem precisar duplicar
# a chave em OPENAI_API_KEY nem repetir o base_url no .env. Qualquer outro
# nome de modelo continua usando o par OPENAI_API_KEY / OPENAI_BASE_URL
# (OpenRouter, Groq, OpenAI, ...). Um OPENAI_BASE_URL explícito no .env ainda
# tem prioridade, se alguém quiser forçar outro endpoint.
_IS_GEMINI = GENERATION_MODEL.startswith("gemini")

OPENAI_API_KEY = (os.getenv("GEMINI_API_KEY") if _IS_GEMINI else None) or os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or (
    "https://generativelanguage.googleapis.com/v1beta/openai/" if _IS_GEMINI else None
)

# --- Heurística de "fora de escopo" (src/generate.py) -----------------------
# Distância L2 máxima aceitável entre a pergunta e o chunk mais parecido do
# índice. Precisa ser calibrado empiricamente com perguntas reais (ver
# `if __name__ == "__main__"` em src/generate.py) - o valor abaixo é um
# ponto de partida, não um número validado contra o índice de vocês.
OUT_OF_SCOPE_SCORE_THRESHOLD = float(os.getenv("OUT_OF_SCOPE_SCORE_THRESHOLD", "0.9"))
