"""
Testes de sanidade para a 3ª camada de guardrail em lgpd_policy.py:
`contains_credential_like_content` e `filter_out_leaked_restricted_docs`.

Não depende de índice FAISS, LLM ou API key - testa só as funções puras.
Rodar com: `python src/test_leaked_docs.py` (a partir da raiz do repo).

Contexto: essa camada nasceu do vazamento encontrado na Q24/Q08 (um e-mail
interno com chave da Stripe e segredo JWT reais chegou ao contexto da
síntese). O filtro remove do contexto o chunk `restrito` de comunicação
livre (e-mail/log/ticket) cujo conteúdo parece uma credencial em texto puro.

O caso mais importante coberto aqui é o FALSO POSITIVO que já derrubou a Q17:
`seguranca_lgpd.pdf` *fala sobre* senha/chave de API como assunto, sem conter
credencial nenhuma - e por ser `doc_type="policy"` (fora de e-mail/log/ticket)
NÃO pode ser removido.
"""

from lgpd_policy import (
    contains_credential_like_content,
    filter_out_leaked_restricted_docs,
)


class _FakeDoc:
    def __init__(self, page_content, sensitivity="publico", doc_type="email"):
        self.page_content = page_content
        self.metadata = {"sensitivity": sensitivity, "doc_type": doc_type}


def check(name: str, condition: bool):
    status = "OK" if condition else "FALHOU"
    print(f"[{status}] {name}")
    if not condition:
        raise AssertionError(name)


# --- contains_credential_like_content: deteta credencial em texto puro -----
check(
    "conteudo: chave de API em texto puro -> True",
    contains_credential_like_content("A chave de API da Stripe e sk_live_abc123") is True,
)
check(
    "conteudo: string de conexao postgres -> True",
    contains_credential_like_content("DATABASE_URL=postgres://user:pass@10.0.0.1:5432/prod") is True,
)
check(
    "conteudo: segredo JWT -> True",
    contains_credential_like_content("O segredo JWT usado para assinar os tokens e X9f...") is True,
)
check(
    "conteudo: senha em texto puro -> True",
    contains_credential_like_content("login: admin / senha: 1234abcd") is True,
)
check(
    "conteudo: string vazia -> False",
    contains_credential_like_content("") is False,
)
check(
    "conteudo: texto operacional benigno -> False",
    contains_credential_like_content("Reuniao de planejamento de sprint na quinta as 10h") is False,
)


# --- filter_out_leaked_restricted_docs: remove so o certo -----------------
leaked_email = _FakeDoc(
    "De: infra@vendefacil.com.br\nA chave de API de producao da Stripe e sk_live_51H...",
    sensitivity="restrito",
    doc_type="email",
)
leaked_ticket = _FakeDoc(
    "Cliente colou no chamado: DATABASE_URL=postgres://svc:hunter2@db-prod:5432/vendefacil",
    sensitivity="restrito",
    doc_type="ticket",
)
security_policy = _FakeDoc(
    "Politica de Seguranca: nenhuma senha ou chave de API pode ser enviada por e-mail; "
    "use o cofre de segredos corporativo.",
    sensitivity="restrito",
    doc_type="policy",
)
ordinary_restricted_email = _FakeDoc(
    "Combinamos o reajuste salarial do time de suporte para abril; RH ja foi avisado.",
    sensitivity="restrito",
    doc_type="email",
)
manual_mentioning_token = _FakeDoc(
    "Passo 4: gere um token de acesso no painel do integrador e cole no campo indicado.",
    sensitivity="restrito",
    doc_type="manual",
)
public_with_credential_text = _FakeDoc(
    "FAQ publica: nunca compartilhe sua senha; o suporte nunca vai pedir sua senha.",
    sensitivity="publico",
    doc_type="email",
)

kept = filter_out_leaked_restricted_docs(
    [
        leaked_email,
        leaked_ticket,
        security_policy,
        ordinary_restricted_email,
        manual_mentioning_token,
        public_with_credential_text,
    ]
)

check("filtro: e-mail restrito com chave de API e removido", leaked_email not in kept)
check("filtro: ticket restrito com string postgres e removido", leaked_ticket not in kept)
check(
    "filtro: seguranca_lgpd.pdf (policy) que so MENCIONA chave/senha e mantido (guarda da Q17)",
    security_policy in kept,
)
check(
    "filtro: e-mail restrito comum, sem credencial, e mantido",
    ordinary_restricted_email in kept,
)
check(
    "filtro: manual restrito citando 'token' e mantido (doc_type fora de e-mail/log/ticket)",
    manual_mentioning_token in kept,
)
check(
    "filtro: chunk publico com texto tipo credencial e mantido (so 'restrito' e filtrado)",
    public_with_credential_text in kept,
)
check("filtro: retorna exatamente os 4 chunks mantidos", len(kept) == 4)
check(
    "filtro: preserva a ordem relativa dos chunks mantidos",
    kept == [security_policy, ordinary_restricted_email, manual_mentioning_token, public_with_credential_text],
)
check(
    "filtro: lista sem nenhum vazamento passa intacta",
    filter_out_leaked_restricted_docs([security_policy, ordinary_restricted_email])
    == [security_policy, ordinary_restricted_email],
)
check("filtro: lista vazia -> lista vazia", filter_out_leaked_restricted_docs([]) == [])

print("\nTodos os testes de test_leaked_docs.py passaram.")
