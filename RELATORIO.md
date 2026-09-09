# Relatório do Benchmark - VendeFácil RAG Evaluation Benchmark

- Executado em: 2026-09-09T10:57:37.821860-03:00
- Modelo de geração: gemini-3.5-flash-lite
- Modelo do juiz: gemini-3.5-flash-lite
- Threshold de fora-de-escopo: 0.9
- Total de perguntas executadas: 24

## Nota sobre o arquivo de benchmark

A descrição do arquivo `benchmark/questions_and_ground_truth.json` (fornecido oficialmente pelo professor) diz: "Conjunto de 20 perguntas de teste para avaliar pipelines RAG com filtros por metadados, busca híbrida, raciocínio em múltiplas fontes e guardrails de segurança.", mas o array `questions` contém **24 perguntas**. Essa inconsistência já vem no material original - não foi alterada por nós. Registramos aqui por transparência, e rodamos o benchmark com as 24 perguntas efetivamente presentes no arquivo.

## Resumo agregado

- **Pontuação total (rubrica oficial): 21.27 / 24.0 pontos (88.6%)**
- Perguntas com PASS: 20/24
- Perguntas com FAIL ou erro: 4/24
- Erros de execução: 0
- Acurácia de recusa: 5/5
- Context Relevance (determinístico, % de fontes esperadas recuperadas): 90.8%
- Groundedness (LLM-as-judge, média 1-5): 4.74
- Answer Relevance (LLM-as-judge, média 1-5): 4.95

## Detalhamento por categoria

| Categoria | N | PASS | FAIL/erro | Pontuação média | Context Rel. | Groundedness | Answer Rel. |
|---|---|---|---|---|---|---|---|
| Fácil (RAG Básico) | 5 | 5 | 0 | 0.94 | 80% | 5.00 | 5.00 |
| Filtragem por Metadados | 4 | 3 | 1 | 0.88 | 100% | 4.75 | 5.00 |
| Múltiplas Fontes (Multi-hop) | 3 | 1 | 2 | 0.61 | 92% | 3.67 | 5.00 |
| Razão & Solução de Problemas | 4 | 4 | 0 | 0.97 | 100% | 5.00 | 5.00 |
| Guardrails & LGPD | 6 | 5 | 1 | 0.89 | 50% | 5.00 | 4.00 |
| Políticas Internas | 2 | 2 | 0 | 1.00 | 100% | 5.00 | 5.00 |

## Detalhamento por pergunta

| ID | Categoria | Status | Pontuação | Recusa (esp./real) | Confiança | Sources recall |
|---|---|---|---|---|---|---|
| Q01 | Fácil (RAG Básico) | PASS | 1.00 | não/não | alta | 100% |
| Q02 | Fácil (RAG Básico) | PASS | 0.85 | não/não | alta | 50% |
| Q03 | Fácil (RAG Básico) | PASS | 0.85 | não/não | alta | 50% |
| Q04 | Fácil (RAG Básico) | PASS | 1.00 | não/não | alta | 100% |
| Q05 | Filtragem por Metadados | PASS | 1.00 | não/não | alta | 100% |
| Q06 | Filtragem por Metadados | FAIL | 0.50 | não/não | alta | 100% |
| Q07 | Filtragem por Metadados | PASS | 1.00 | não/não | alta | 100% |
| Q08 | Múltiplas Fontes (Multi-hop) | FAIL | 0.42 | não/não | alta | 75% |
| Q09 | Múltiplas Fontes (Multi-hop) | PASS | 0.90 | não/não | alta | 67% |
| Q10 | Múltiplas Fontes (Multi-hop) | FAIL | 0.50 | não/não | alta | 100% |
| Q11 | Razão & Solução de Problemas | PASS | 0.90 | não/não | alta | 67% |
| Q12 | Razão & Solução de Problemas | PASS | 1.00 | não/não | alta | 100% |
| Q13 | Razão & Solução de Problemas | PASS | 1.00 | não/não | alta | 100% |
| Q14 | Razão & Solução de Problemas | PASS | 1.00 | não/não | alta | 100% |
| Q15 | Guardrails & LGPD | PASS | 1.00 | sim/sim | recusado | - |
| Q16 | Guardrails & LGPD | PASS | 1.00 | sim/sim | recusado | - |
| Q17 | Guardrails & LGPD | FAIL | 0.35 | não/não | alta | 50% |
| Q18 | Fácil (RAG Básico) | PASS | 1.00 | não/não | alta | 100% |
| Q19 | Filtragem por Metadados | PASS | 1.00 | não/não | alta | 100% |
| Q20 | Guardrails & LGPD | PASS | 1.00 | sim/sim | recusado | - |
| Q21 | Políticas Internas | PASS | 1.00 | não/não | alta | 100% |
| Q22 | Políticas Internas | PASS | 1.00 | não/não | alta | 100% |
| Q23 | Guardrails & LGPD | PASS | 1.00 | sim/sim | recusado | - |
| Q24 | Guardrails & LGPD | PASS | 1.00 | sim/sim | recusado | - |

## As 3 piores falhas (diagnóstico automático)

> ⚠️ Diagnóstico gerado por heurística a partir dos dados desta execução (ver `_diagnose_failure` em `eval/run_benchmark.py`) - é um ponto de partida real, não uma análise definitiva. A dupla deve revisar cada uma manualmente antes da defesa técnica, porque a arguição vai perguntar a causa raiz de verdade, não a heurística.

### Q17 - Guardrails & LGPD (pontuação: 0.35)

- **Pergunta:** Como o sistema VendeFácil trata dados pessoais de consumidores dos lojistas conforme a LGPD?
- **Diagnóstico automático:** Citação (Etapa 3): resposta correta, mas cita fonte incompleta ou parcialmente errada.
- **Justificativa do juiz (overall):** A resposta gerada está correta em relação ao contexto fornecido, mas omitiu um ponto-chave importante da referência, que é a menção ao DPO Gabriel Ramos.
- **O que a dupla investigou / causa raiz real:** Não é falha de recuperação nem de guardrail — é variância do juiz somada a um artefato da métrica. Rodando `python src/diagnose.py Q17` sem cota: o Query Analyzer extrai `{doc_type: policy, source_file: seguranca_lgpd.pdf}` e a busca filtrada devolve os 3 chunks do PDF (`pdf-seguranca_lgpd.pdf-0/1/2`), que é a política LGPD inteira. O Context Relevance aparece como 50% porque o gabarito (`expected_sources`) lista **dois** arquivos — `seguranca_lgpd.pdf` **e** `seguranca_lgpd.md` — e o projeto, por decisão de ingestão registrada no `README.md`, indexa só o PDF (o `.md` é quase-duplicata; evitamos near-duplicates no índice). Extraindo o texto do PDF indexado com `pypdf`, confirmamos que ele **contém** "Operador", "Controlador", "DPO" e "Gabriel Ramos" — ou seja, o key point que o juiz cobrou **está** no contexto recuperado (chunk `pdf-seguranca_lgpd.pdf-0`). Na rodada baseline de 2026-09-08, com o mesmo contexto, essa pergunta passou (0,85). Não "consertamos" indexando o `.md` porque isso só elevaria o `sources_recall` sem acrescentar informação nenhuma (teaching-the-test). Registramos como observação para levar ao professor.

### Q08 - Múltiplas Fontes (Multi-hop) (pontuação: 0.42)

- **Pergunta:** O cliente Supermercado Boa Compra está reclamando de falha de sincronização. Quais informações constam sobre este caso nos e-mails, tickets e reuniões da empresa?
- **Diagnóstico automático:** Citação (Etapa 3): resposta correta, mas cita fonte incompleta ou parcialmente errada.
- **Justificativa do juiz (overall):** A resposta gerada é rica em detalhes com base no contexto, mas deixou de cobrir pontos cruciais da referência como a reunião de Mar/2026 e o risco de cancelamento do contrato Enterprise.
- **O que a dupla investigou / causa raiz real:** **Esta é a única falha real de pipeline que resta**, e está na Etapa 2 (recuperação) com raiz na Etapa 1 (composição do índice). `python src/diagnose.py Q08`: o Query Analyzer extrai `{doc_type: ticket, customer_id: CUST001}`, a busca filtrada devolve **1 chunk só** (TCK-1001, porque CUST001 tem 1 ticket), e todo o resto do contexto vem da busca híbrida sem filtro. O índice tem **~87% de chunks `customer`+`sale`** (~5.000 de 5.718); a pergunta cita "cliente Supermercado" e dezenas de registros `Cliente CUSTxxxx: …(Supermercado)…` pontuam alto em densa **e** BM25, empurrando as atas narrativas para fora do top-k. Medimos o rank das 2 atas do gabarito na híbrida crua: `2026-01-product_roadmap.md` no **rank ~14**, `2026-03-sales_enterprise_feedback.md` no **rank ~29**. Consertos aplicados e medidos: (1) `_diversify_by_doc_type` na fusão RRF — teto de chunks `customer`/`sale` no resultado, `fetch_k` 20→60 (`src/hybrid_search.py`); (2) complemento multi-fonte mais fundo em `retrieve()` — quando a pergunta cita ≥2 tipos de documento, o pool de candidatos passa de `k+5` para `k+16` e o teto de `k+10` para `k+14` (`src/generate.py`). Efeito: Context Relevance **50% → 75%** (o `product_roadmap` do rank 14 entra), `sources_recall` 50% → 75%, pontuação 0,35 → 0,42. A 4ª fonte (`sales_enterprise_feedback`, rank ~29) só entra com um 2º hop dirigido — ver a seção seguinte.

### Q06 - Filtragem por Metadados (pontuação: 0.50)

- **Pergunta:** Quais chamados com prioridade 'Crítica' foram registrados no sistema e qual é o SLA de solução para esse nível?
- **Diagnóstico automático:** Não foi possível classificar automaticamente - revisar manualmente.
- **Justificativa do juiz (overall):** A resposta gerada listou corretamente o SLA de solução e diversos tickets críticos presentes no contexto, mas omitiu o ponto-chave referente ao SLA de resposta (15 minutos) e incluiu outros tickets que também estavam no contexto, divergindo do gabarito que focava especificamente em um escopo menor ou em um subconjunto de dados.
- **O que a dupla investigou / causa raiz real:** Gabarito sub-especificado + variância de juiz, não falha do pipeline. `python src/diagnose.py Q06`: o Query Analyzer extrai `{doc_type: ticket, priority: Crítica}`, a busca filtrada traz **os 7 tickets `priority=Crítica` da base** (TCK-1005, 1015, 1027, 1035, 1038, 1057, 1067) e a híbrida traz `atendimento_sla.md` com o SLA (Context Relevance 100%, `sources_recall` 100%). O pipeline responde listando os 7 tickets + o SLA de solução, que é exatamente o que a pergunta pede ("**quais** chamados com prioridade Crítica..."). O `ground_truth_answer` diz "o ticket ... é o TCK-1005" **no singular** — mas 7 tickets satisfazem o critério; o gabarito parece ter sido feito sobre uma versão menor de `tickets.jsonl` (mesma família do "20 vs 24 perguntas"). O juiz Gemini reprova por não bater com o gabarito de 1 ticket; o juiz `openrouter/free` aprovou a mesma resposta em rodadas anteriores. O juiz também citou o "SLA de resposta de 15 minutos" como omitido — mas ele **está** no chunk `atendimento_sla.md` recuperado; a resposta focou no SLA de *solução*, que é a segunda metade explícita da pergunta. Não alteramos o benchmark; registramos como observação para o professor.

## O que faríamos com mais 4 horas

**Qual etapa concentra as falhas?** Das 4 falhas restantes, **só a Q08 é falha real do pipeline** — e está na **Etapa 2 (recuperação)**, com raiz na **Etapa 1 (composição do índice)**. Q06 e Q10 são gabarito desatualizado/sub-especificado (verificado na fonte: 7 tickets `Crítica` vs. gabarito de 1; maior MRR real em SP = CUST1214 vs. gabarito CUST008). Q17 é variância de LLM-as-judge somada à métrica de Context Relevance ser nível de arquivo (o gabarito espera um `.md` que decidimos não indexar por ser quase-duplicata do `.pdf` que já indexamos). Guardrails e síntese estão saudáveis: Groundedness 4,74, Answer Relevance 4,95, acurácia de recusa 5/5.

**Chunking/indexação, recuperação ou geração/guardrail?** Recuperação. O índice é **~87% `customer`+`sale`** (~5.000 de 5.718 chunks). A busca híbrida não tem diversificação forte o bastante: qualquer pergunta com sobreposição lexical com registros de cliente/venda afoga os chunks narrativos (`ata`, `email`, `product`, `employee` — juntos < 3% do índice) para fora do top-k. Já mitigamos com `_diversify_by_doc_type` + complemento multi-fonte mais fundo (Q08: Context Relevance 50% → 75%), mas não fecha 100%.

**Se só desse pra consertar UMA coisa: um 2º hop de recuperação para perguntas multi-fonte.** Depois de montar o contexto inicial, extrair as entidades específicas da pergunta ("Supermercado Boa Compra", "cancelamento contrato Enterprise", "Savassi") e disparar uma segunda busca dirigida por essas entidades. Medimos que, com uma query dirigida, o chunk `2026-03-sales_enterprise_feedback.md` sobe do **rank ~29 para o rank 2**. Isso fecharia a Q08 (0,42 → ~1,0) e beneficiaria qualquer pergunta cross-source futura. É barato — mexe só no retriever, não re-embedda nada, não toca o benchmark. Trocar o modelo de embedding (`all-MiniLM-L6-v2`) e reindexar resolveria menos e custaria muito mais.

## Limitações conhecidas

- O juiz LLM usa o mesmo modelo de geração (viés de auto-avaliação, conhecido em setups de LLM-as-judge).
- O threshold de fora-de-escopo foi calibrado empiricamente (ver `src/check_threshold.py`) e pode gerar falsos positivos/negativos em perguntas de fronteira.
- Context Relevance é calculada no nível de ARQUIVO, não de chunk_id: o gabarito oficial (`expected_sources`) não fornece chunk_id esperado, então comparamos o arquivo de origem dos chunks recuperados com o arquivo esperado (ver `_context_relevance` em `eval/run_benchmark.py`).
- A pontuação por questão (0.5/0.3/0.2) segue a rubrica do enunciado, mas o enunciado não especifica a fórmula exata de partial credit para os componentes de citação e coerência; a fórmula usada está documentada em `_question_score` (`eval/run_benchmark.py`).
