# Histórico do Projeto — como o trabalho foi feito

Narrativa do processo, encontro por encontro: o que foi construído, o que quebrou, como
foi resolvido e quais decisões foram tomadas ao longo do caminho. Feito para **estudar e
apresentar** a parte de "como fizemos".

- **Companheiro de leitura técnico:** `APRESENTACAO.md` (o *sistema* — arquitetura,
  decisões, roteiro dos 7 min, banco de perguntas do mentor).
- **Fonte primária:** `ACOMPANHAMENTO.md` (relatos por encontro) e o histórico de commits.
- Trio: Paula Thamyres (Femina), Élcio Santos Júnior, Letícia Sousa. Divisão do trabalho
  **por item dentro de cada etapa**.

---

## Panorama em 30 segundos

O desafio tem 4 etapas com "critério de pronto" — não se avança sem fechar a anterior.
Construímos um assistente RAG que consulta 6+ formatos de arquivo, filtra por metadado,
responde **sempre com citação literal** e respeita a LGPD (recusa / mascara / responde).

A nota do benchmark evoluiu de **5/24** (primeira rodada, provedor instável) até
**21,27/24 = 88,6%, 20 PASS** (rodada final). A maior parte do salto veio de: (1) trocar
para um modelo estável (Gemini), (2) fechar dois vazamentos/bugs mecânicos, e (3)
diagnosticar cada falha restante estágio por estágio em vez de "mexer no prompt e torcer".

---

## Encontro 1 — 2026-08-26 · Etapa 1 (Ingestão heterogênea)

**Objetivo:** ler os 6 formatos, chunking adaptativo, schema de metadados, indexar em FAISS.

**O que foi feito.** Implementamos os leitores das 6 fontes em `src/loaders.py` com uma
**estratégia de chunking por natureza da fonte** (não uma configuração única): CSV/JSON
tabular → 1 registro = 1 chunk serializado em frase natural; JSONL → 1 ticket = 1 chunk;
Markdown → split por cabeçalho; PDF → split recursivo com overlap; TXT → 1 arquivo = 1
mensagem. Serializamos registro tabular em frase (`"Cliente CUST001: Supermercado Boa
Compra... em Belo Horizonte/MG, plano Enterprise..."`) porque nos primeiros testes de
similaridade os chunks CSV crus (`"CUST001,Boa Compra,MG"`) não recuperavam bem — o
embedding não separa campo de valor numa string crua.

**O que quebrou.** A primeira versão dos loaders foi escrita olhando só o exemplo do
enunciado. Ao testar contra o dataset real:
- `customers.csv` **não tem** coluna `name` nem `signup_date` (o certo é `company_name`, e
  não existe data de cadastro nesse dataset);
- `products.json` e `stores.json` **não são listas na raiz** — são objetos com as chaves
  `"products"` e `"network_stores"`. Ficamos ~20 minutos travados num
  `AttributeError: 'str' object has no attribute 'get'` até perceber que estávamos
  iterando as **chaves do dicionário** em vez da lista de dentro.

Também puxamos `state` e `module` dos tickets para os metadados, prevendo a Etapa 2.

**Resultado.** ~5.715 chunks, com a distribuição por `doc_type` batendo exatamente com a
contagem de linhas de cada arquivo fonte (2000 clientes, 3000 vendas, 450 logs, 75
tickets...). Nenhum chunk sem `doc_type` ou `sensitivity`. Índice FAISS salvo em disco
(`save_local`), recarregado sem reindexar (`load_local`).

**Uso de IA.** Claude comparou o `loaders.py` com os arquivos reais de dados (baixados do
repositório-base) e apontou os 3 bugs de mapeamento de campo. As correções foram revisadas
e testadas contra o dataset real antes de aceitar.

---

## Encontro 2 — 2026-08-28 e 2026-08-31 · Etapa 2 (Busca híbrida + filtros)

Dividido em 3 itens: Query Analyzer (Élcio), aplicar filtros na busca (Letícia + Élcio),
fusão RRF (Paula).

### Query Analyzer — por regra (Élcio)

`src/query_analyzer.py`: recebe a pergunta e devolve filtros estruturados (`doc_type`,
`state`, `module`, `customer_id`, `priority`, `status`). **Escolha por regras, não LLM** —
mais simples, determinístico, sem depender de API. Normalização de caixa/acento/espaço
("São Paulo" = "sao paulo" = "SP"), dicionário de estados (Minas Gerais → MG), sinônimos
(chamado/chamados → ticket), modelo Pydantic `QueryFilters`.

**Decisão importante:** o Query Analyzer carrega os valores que **realmente existem** nos
metadados do índice e só devolve um filtro quando o valor é válido. Teste que provou isso:
*"Mostre os tickets de Wakanda do módulo abacaxi"* → resultado `{doc_type: ticket}` só,
porque "Wakanda" e "abacaxi" não existem nos metadados.

Bug do dia: `ModuleNotFoundError: No module named 'pydantic'` — dependência não instalada
no ambiente.

### Aplicar os filtros na busca vetorial (Letícia, depois Élcio)

Letícia integrou os filtros ao FAISS combinando pré-filtragem lógica com pós-filtragem no
recuperador (`vectorstore.as_retriever(search_kwargs={'filter': ...})`), com fallback
quando a combinação de filtros resulta em conjunto vazio.

**Descoberta na revisão (Élcio, 31/08):** o item 3 aparecia no `ACOMPANHAMENTO.md` como
implementado, **com referência a `src/search.py`**, mas o arquivo **não estava no
repositório**. Élcio implementou `src/search.py` de fato. Dois problemas apareceram:
- **`fetch_k` pequeno + filtro seletivo devolve `[]`** mesmo havendo chunks válidos (o
  FAISS do LangChain filtra *depois* da busca). Solução: `fetch_k` grande na busca filtrada.
- O Query Analyzer devolvia `"VendeFácil Estoque"` mas os tickets usam `"estoque"` nos
  metadados → a busca filtrada zerava. Ajuste: priorizar o valor que casa com os
  metadados reais → `module: estoque`.

Validado com 3 consultas (MG, SP, RJ + módulo estoque), comparando busca com e sem filtro.

### Fusão RRF (Paula)

`src/hybrid_search.py`: busca densa (embeddings via FAISS) + busca esparsa (BM25),
fundindo os dois rankings. Dúvida inicial: como fundir, já que densa e esparsa têm escalas
totalmente diferentes? Seguimos o enunciado e usamos **Reciprocal Rank Fusion** em vez de
somar scores: cada doc ganha `1 / (60 + posição)` em cada ranking, e somamos. Tokenização
própria para o BM25 (minúsculo, sem acento) — sem isso "São Paulo" e "sao paulo" viravam
tokens diferentes.

Escolhemos perguntas de teste que mostram **cada recuperador falhando sozinho**: uma
parafraseada sem termo literal (BM25 fica atrás) e uma citando o código exato de um ticket
(o denso perde para o BM25).

> **Nota de manutenção (31/08):** este bloco ficou salvo no repositório com marcadores de
> conflito de merge (`<<<<<<<`, `=======`, `>>>>>>>`) — o conflito nunca tinha sido
> resolvido, só commitado como estava. Foi limpo depois, sem alterar conteúdo dos relatos.

---

## Encontro 3 — 2026-08-31 · Etapa 3 (Síntese + Pydantic + LGPD)

### Schema Pydantic (Élcio)

`src/schema.py`: `SourceEvidence` (`filepath`, `chunk_id`, `quotation`) e `RAGResponse`
(`answer`, `confidence_level`, `sources_used`, `reasoning`, `is_refusal`, `refusal_reason`).
Campos de valor fechado usam `Literal` (não `str` livre — senão o modelo escreve "Alta",
"ALTA", "muito alta" e o guardrail deixa de ser guardrail). `model_validator` que impõe a
regra de consistência: `is_refusal=True` ⇒ confiança `recusado`, sem fontes, com motivo;
`is_refusal=False` ⇒ ≥ 1 evidência.

`src/test_schema.py`: 5 casos (2 válidos, 3 inválidos propositais — resposta sem
evidência, recusa com confiança "alta", `confidence_level="muito alta"`). Todos deram o
resultado esperado.

**Decisão:** o schema só valida; o retry em caso de saída inválida fica na camada que
chama o LLM (é ela que consegue pedir nova geração). Assim o `schema.py` fica independente
do modelo.

### Citação de evidência + LGPD 3 níveis (Paula)

Claude gerou a primeira versão de `src/lgpd_policy.py` (classificação
recusar/mascarar/responder por regra + mascaramento de e-mail/telefone/CPF/cartão) e
`src/generate.py` (pipeline que liga recuperação → LGPD → LLM → validação Pydantic).

`src/test_lgpd_policy.py`: 18 testes passam **sem chave de API** (classificação, mascaramento,
segunda camada de defesa quando todos os chunks são `sensitivity="restrito"`).

Bug: `config.py` fica na raiz, o script roda de dentro de `src/`, o Python não achava o
módulo. Corrigido adicionando a raiz ao `sys.path` no início do `generate.py`.

**Falha encontrada:** das 8 perguntas de teste (2 recusa LGPD, 2 mascarar, 2 responder, 2
fora de escopo), 6/8 corretas. O detector de "fora de escopo" errava 2:
- **falso positivo:** "sincronização de estoque" recusada por engano;
- **falso negativo:** "quem descobriu o Brasil" passou pro LLM em vez de recusar.

Investigado com `src/check_threshold.py`: **não existe um único valor de limiar** que
separe os dois grupos sem erro nesse conjunto. Decidimos **documentar como limitação
conhecida no README** (com os números reais) em vez de mascarar o problema — o guia
valoriza esse tipo de diagnóstico honesto.

---

## Atividade fora do encontro — 2026-09-01 · pré-Etapa 4

### Correção do detector de "fora de escopo" (Paula)

Antes de seguir para a Etapa 4, resolvemos a pendência do Encontro 3. Criamos
`src/domain_keywords.py` com vocabulário de domínio **tirado de dados reais** (os 5 nomes
de produto em `products.json`, valores reais de `module`/`doc_type` do índice, termos
operacionais dos manuais — nada inventado). Lógica nova em `is_out_of_scope()`: se a
pergunta cita um termo real do domínio, **nunca** é recusada por fora de escopo; só cai na
checagem de distância quando não há nenhum termo de domínio.

| Rodada | Threshold | Acertos | O que mudou |
|---|---|---|---|
| Encontro 3 (no README) | 0,9 | 6/8 | baseline, sem a correção |
| 1ª rodada (keyword, threshold 0,9) | 0,9 | 7/8 | o falso positivo foi corrigido pela keyword |
| 2ª rodada (threshold ajustado) | **0,80** | **8/8** | baixamos o threshold com base na margem real medida |

`OUT_OF_SCOPE_SCORE_THRESHOLD` foi de 0,9 para 0,80 no `.env`.

### Escrita do harness e primeira rodada do benchmark (Paula)

Escrevemos `eval/run_benchmark.py` e `eval/judge_prompt.py` do zero. Antes de rodar,
corrigimos um problema de configuração: o `.env` tinha uma chave da Groq mas
`GENERATION_MODEL=gpt-4o-mini` sem `OPENAI_BASE_URL` → a primeira tentativa deu **401
(Incorrect API key)** porque a chave da Groq estava sendo mandada para a API da OpenAI de
verdade. Ajustamos para o endpoint da Groq + `openai/gpt-oss-120b`.

O `run_benchmark.py` decide se uma pergunta espera recusa olhando o texto do próprio
`ground_truth_answer` (se começa com "RECUSA DE RESPOSTA" ou "FORA DO ESCOPO") — isso trata
certo a Q17, que está na categoria "Guardrails & LGPD" mas espera resposta normal.

**Achado crítico — Q24 (vazamento de credencial):** o pipeline respondeu com a **chave
secreta de produção da Stripe e o segredo JWT reais**, extraídos de um e-mail interno,
quando deveria ter recusado. Causa: `classify_question()` tinha o padrão `r"chave de api"`
(frase exata), mas a pergunta dizia "chave **secreta** de API" — a palavra no meio quebrava
o match. A segunda camada (`has_only_restricted_docs`) também não pegou porque esse e-mail
não estava marcado `sensitivity="restrito"`. Corrigimos o regex para aceitar "chave ... api"
com palavras no meio, mais `segredo`/`jwt` como gatilhos, e validamos rodando
`test_lgpd_policy.py` de novo (18/18, sem regressão).

**Segundo achado:** muitas perguntas normais estavam sendo recusadas por "falta de
evidência". O `QueryAnalyzer` combinava filtros de um jeito que zerava a busca (ex.: a
palavra "cliente" trava `doc_type=customer`, combinado com `module=pay` e
`customer_id=CUST008` não bate com nenhum chunk). Corrigimos o `retrieve()` para cair na
busca híbrida sem filtro quando a busca filtrada retorna zero, em vez de recusar direto.

| Rodada | O que mudou | Resultado |
|---|---|---|
| 1ª | baseline, com o vazamento da Q24 | **5/24** |
| 2ª | fix do `lgpd_policy.py` | **6/24** (Q24 passa a recusar) |
| 3ª | fallback no `retrieve()` | **8/24** (Q11, Q18 param de ser recusadas à toa) |

Decidimos **parar antes de mexer em chunking/embedding** — as falhas restantes já não eram
mais bug mecânico, e sim profundidade/qualidade de recuperação.

---

## Encontro 4 — 2026-09-02 · revisão dos resultados (Paula)

Sem mudança de código. Revisão de `eval/results.json` e `RELATORIO.md`, confirmando que as
16 falhas restantes eram **limitação de recuperação** (k=5, tamanho de chunk, e o embedding
`all-MiniLM-L6-v2` indo mal em agregação — "qual cliente tem o maior MRR" — e lookup exato
por ID), **não bug**. Organizamos a lista de pendências para o grupo decidir prioridade:
melhorar recuperação vs. documentar como limitação conhecida.

---

## Encontro 5 — 2026-09-04 · preparação da apresentação (Paula)

Foco em preparar a apresentação (no Canva). Antes de escrever slides, pedimos uma análise
completa do repositório para confirmar o que faltava: o pipeline inteiro (ingestão, busca,
guardrails, avaliação) estava pronto e testado — **a única peça faltando era a interface
de demonstração**.

Montamos o conteúdo de 15 slides + uma seção grande de perguntas prováveis do professor
por tema.

**Bloqueio do dia:** ao tentar retomar um teste que dependia de LLM (Groq), caímos no
**limite de tokens diário (TPD)**, com o mesmo identificador de organização de tentativas
anteriores. Confirmamos que o limite é **por conta/organização** da Groq, não por chave —
gerar uma chave nova na mesma conta não resolve. O reset é diário. Pausamos os testes
dependentes de LLM.

---

## Encontro 6 — 2026-09-08 · fechar as pendências da Etapa 4 (Paula) — commit `7d83490`

Dia de fechar tudo que faltava:
- **Benchmark completo 24/24 sem erro de execução.**
- `src/demo.py` — interface de demonstração em CLI (o item que faltava do critério de pronto).
- `src/diagnose.py` — diagnóstico local estágio por estágio, sem custo de API por pergunta.
- `src/structured_query.py` — roteamento de perguntas de agregação (maior/menor MRR) via
  consulta direta ao CSV com pandas, em vez de busca vetorial.
- `src/generate.py` — generaliza o complemento de busca híbrida (pendência do Encontro 2).
- **Seções manuais avaliadas do `RELATORIO.md` preenchidas** (causa raiz real das 3 piores
  falhas + "o que faríamos com mais 4 horas"), com a causa confirmada rodando `diagnose.py`.
- Reindexação do `faiss_index/`, ajustes no `eval/`, e seções novas em
  `data/unstructured/policies/` (política interna que faltava, cobre Q21/Q22 — a decidir
  com o professor).

**Baseline oficial (rodada 19:17, `gemini-3.5-flash-lite`):** **20,25 / 24 = 84,4%, 19
PASS, 0 erro de execução.** Context Relevance 89,8%, Groundedness 4,67, Answer Relevance 4,83.

O salto de 8/24 (Groq) para 20,25/24 (Gemini) veio de: modelo estável (o free tier da Groq
era instável e limitado), + as correções acumuladas, + o trabalho deste dia.

---

## Encontro 7 — 2026-09-09 · diagnóstico final e correções (Paula) — commits `6dec87f`, `e460c9e`, `bd5ed00`, `f5cda29`

Restavam 5 falhas: Q01, Q06, Q08, Q10, Q17. Rodamos `src/diagnose.py` em cada uma (sem
gastar cota) para separar falha real de ruído.

**Q02 — o achado mais forte.** A Q02 ("quem é o Tech Lead / PM do VendeFácil Estoque?")
parecia falha de recuperação (Context Relevance 50%). Mas o `diagnose.py` mostrou o chunk
certo (`products.json` / `product-1`) **sendo recuperado sem os nomes** — o serializador de
produto em `loaders.py` descartava os campos `tech_lead` e `product_manager` que estão no
JSON. **Chunk recuperado ≠ resposta disponível.** Incluímos os campos na frase e
reindexamos: **0,00 → PASS 0,85**, determinístico.

**Q08 — a única falha real de pipeline.** Primeira hipótese: mexer no `query_analyzer.py`
para não travar `doc_type: ticket`. Mas as atas do gabarito têm `customer_id=None` — nenhum
filtro as traria. O gargalo real é a **profundidade do complemento** em `retrieve()`.
Medimos o rank das 2 atas na busca híbrida crua: rank ~14 e rank ~29, e o `retrieve()` só
buscava 13 candidatos. Ampliamos o pool para `k+16` e o teto para `k+14` quando a pergunta
cita 2+ tipos de documento. **Context Relevance 50% → 75%.**

**Q01 — regressão de síntese.** Tinha caído de PASS 1,00 para FAIL 0,50. Comparando as
respostas nas duas rodadas: passou de *"VendeFácil Loja (plataforma de e-commerce
omnicanal)"* para *"VendeFácil Loja (PROD-LOJA)"* — só nome e código. O juiz cobra "breve
descrição de cada produto". Regra nova no `SYSTEM_PROMPT`: ao enumerar entidades, dar cada
item **com descrição do contexto**. **Voltou a PASS 1,00.**

**Q06, Q10, Q17 — não são falha de pipeline:**
- **Q06:** recuperação 100% OK (7 tickets `Crítica` + o SLA). O gabarito diz "o ticket é o
  TCK-1005" no singular; há 7. Gabarito sub-especificado + variância de juiz.
- **Q10:** a rota estruturada acha o MRR real em SP (CUST1214 / R$ 3.487,22); o gabarito
  diz CUST008 / R$ 3.100 (nem está no top-6). Gabarito errado.
- **Q17:** o key point "DPO Gabriel Ramos" **está** no `seguranca_lgpd.pdf` recuperado
  (confirmado com `pypdf`). Context Relevance mostra 50% só porque o gabarito lista também
  o `.md`, que decidimos não indexar por ser quase-duplicata. Variância de juiz + artefato
  da métrica.

**Rodada final (10:57, `gemini-3.5-flash-lite`):** **21,27 / 24 = 88,6%, 20 PASS, 0 erro
de execução.** Context Relevance 90,8%, Groundedness 4,74, Answer Relevance 4,95, acurácia
de recusa 5/5.

Também: `config.py` ganhou roteamento por prefixo (`gemini*` → chave e endpoint do Google
automáticos); a interface (`webdemo.py`/`demo.py`) passou a carregar as **24 perguntas do
benchmark** + botão "Rodar todas"; seções manuais do `RELATORIO.md` reescritas para a
rodada final; bloco Encontro 7 no `ACOMPANHAMENTO.md`.

---

## A curva da nota do benchmark

| Quando | Modelo | Nota | O que mudou |
|---|---|---|---|
| 01/09, 1ª rodada | Groq `gpt-oss-120b` | 5 / 24 | baseline, com o vazamento da Q24 |
| 01/09, 2ª | Groq | 6 / 24 | fix do regex de credencial (Q24 recusa) |
| 01/09, 3ª | Groq | 8 / 24 | fallback no `retrieve()` quando o filtro zera |
| **08/09 19:17** | **gemini-3.5-flash-lite** | **20,25 / 24 = 84,4% (19 PASS)** | **baseline oficial** — modelo estável + trabalho do Encontro 6 |
| 09/09 08:50 | gemini-3.5-flash-lite | 19,43 / 24 = 80,9% (17 PASS) | só diversificação da híbrida (Q01/Q17 caíram por ruído de juiz) |
| 09/09 09:08–09:43 | gemini-3.5-flash-lite | 20,70 / 24 = 86,2% (19 PASS) | + fix de ingestão da Q02 |
| **09/09 10:57** | **gemini-3.5-flash-lite** | **21,27 / 24 = 88,6% (20 PASS)** | **rodada final** — + fix da Q01 e complemento mais fundo da Q08 |

O benchmark tem **±1 a 1,5 ponto de ruído de LLM-as-judge** entre rodadas no mesmo modelo.
O sinal real por baixo do ruído: Q02 consertada (ingestão), Q01 consertada (enumeração),
Q08 melhorada (Context Relevance 50% → 75%).

---

## Decisões que definiram o projeto (e quando foram tomadas)

| Decisão | Encontro | Por quê |
|---|---|---|
| Chunking adaptativo (1 estratégia por natureza de fonte) | 1 | um splitter único deixava os chunks CSV crus irrecuperáveis |
| Serializar registro tabular em frase natural | 1 | o embedding não separa campo de valor numa string crua |
| `sensitivity` classificada na ingestão, não na resposta | 1 | barato, determinístico, testável sem API; sustenta o guardrail |
| Query Analyzer por regra + validação contra o índice | 2 | sem custo de API; nunca emite filtro que não casa nada |
| RRF (`k=60`) em vez de somar scores | 2 | densa e BM25 têm escalas incomparáveis |
| `fetch_k` = tamanho do índice na busca filtrada | 2 | o FAISS do LangChain filtra depois da busca |
| Consistência como validador Pydantic, não instrução no prompt | 3 | o modelo desobedece o prompt; o validador não |
| Documentar a limitação do detector de fora de escopo | 3 | não existe threshold único que separe os grupos sem erro |
| Rota estruturada para agregação MAX/MIN de MRR | Etapa 4 (Enc. 6) | similaridade não responde "qual o maior X" — é aritmética |
| Interface sem framework web (só stdlib) | Etapa 4 (Enc. 6) | menos coisa para quebrar ao vivo no Demo Day |
| Registrar Q05/Q06/Q10 como observação, não "consertar" | Etapa 4 (Enc. 7) | `eval/` e `benchmark/` são a régua; não se ajusta para passar |
| Fixar o modelo em Gemini | Etapa 4 (Enc. 7) | Groq free tier era instável e limitado por cota diária |

---

## Bugs memoráveis (bom para "conte uma dificuldade")

1. **`products.json` não era uma lista** — 20 minutos de `AttributeError` iterando as
   chaves do dicionário em vez da lista dentro dele (Encontro 1).
2. **`src/search.py` fantasma** — o `ACOMPANHAMENTO.md` dizia que estava implementado, com
   nome de arquivo e tudo, mas o arquivo não existia no repositório (Encontro 2).
3. **401 da Groq** — chave da Groq sendo mandada para a API da OpenAI porque faltava o
   `OPENAI_BASE_URL` no `.env` (01/09).
4. **Vazamento da Q24** — o assistente respondeu com a chave secreta de produção da Stripe
   e o segredo JWT reais; o regex do guardrail casava "chave de api" exato e a pergunta
   dizia "chave **secreta** de API" (01/09).
5. **Recusa em cadeia** — o Query Analyzer combinava filtros que zeravam a busca e o
   pipeline recusava por "falta de evidência" em vez de tentar sem filtro (01/09).
6. **Cota diária da Groq (TPD)** — descobrimos que o limite é por organização, não por
   chave, então trocar de chave não adianta (Encontro 5).
7. **Q02 parecia recuperação, era ingestão** — o chunk certo era recuperado, mas o
   serializador tinha descartado os campos com a resposta (Encontro 7).
8. **Q01 regrediu por formato** — mesma informação no contexto, mas a resposta encolheu de
   "nome + descrição" para "nome + código" e o juiz reprovou (Encontro 7).

---

## O que ficou de fora (honestidade — o mentor valoriza isso)

- **Q08 não fecha 100%.** A 4ª fonte (`sales_enterprise_feedback`, rank ~29 na híbrida
  crua) precisa de um 2º hop de recuperação dirigido. Está registrado como "o que faríamos
  com mais 4 horas" no `RELATORIO.md`. Com uma query dirigida, esse chunk sobe para o rank 2.
- **Fusão de verdade filtrada × híbrida (reranking)** — hoje só o caso "filtro devolve
  pouco" é tratado; não há RRF entre o ranking filtrado e o híbrido.
- **Ponto de integridade a levar ao professor:** `eval/run_benchmark.py` e `judge_prompt.py`
  foram reescritos pela equipe (é o entregável da Etapa 4); e `data/.../policies/` ganhou
  seções novas que cobrem Q21/Q22 (política interna real, não copiada do gabarito). Rascunho
  da mensagem em `docs/pergunta-professor-integridade.md`.

---

## Uso de IA no processo (disclosure)

O Claude foi usado em todos os encontros — para revisar os loaders contra os dados reais
(Encontro 1), tirar dúvidas sobre a divisão dos itens e revisar a fórmula RRF (Encontro 2),
gerar a primeira versão do `lgpd_policy.py` e do `generate.py` e investigar o detector de
fora de escopo (Encontro 3), escrever o harness do benchmark e diagnosticar o vazamento da
Q24 (01/09), montar o material de apresentação (Encontro 5), fechar as pendências da Etapa
4 (Encontro 6), e diagnosticar as 5 falhas restantes estágio por estágio (Encontro 7).

Em todos os casos, as **decisões de escopo e a execução/conferência** (rodar o benchmark,
testar contra o dataset real, escolher até onde corrigir) foram da equipe. O detalhe de
onde a IA entrou em cada dia está em cada bloco do `ACOMPANHAMENTO.md`.
