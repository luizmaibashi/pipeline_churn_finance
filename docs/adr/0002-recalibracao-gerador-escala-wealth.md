# ADR-0002: Recalibração do Gerador Sintético para Escala de Wealth Management Real

**Data:** 2026-09-10
**Status:** Accepted
**Proposto por:** Luiz Maibashi
**Estende:** [ADR-0001](0001-refatoracao-early-warning-advisor-attrition.md)

---

## 1. CONTEXTO (O Quê?)

Auditoria da refatoração ADR-0001 (2026-09-10) encontrou três defeitos que não quebram nada e por isso sobreviveram a EDA, testes e dois ciclos de commit.

### 1.1 A carteira nunca somou R$ 75 bi

`README.md`, `PROBLEM.md` e o ADR-0001 descrevem "uma gestora com carteira de R$ 75 bi de AuC". O gerador (`generate_synthetic_data`) nunca foi calibrado para isso: sorteia `saldo_bi` de uma única `lognormal(-1.8, 1.3)` sem condicionar por segmento. O dado real produzido:

| Métrica | Valor gerado hoje | O que a narrativa diz |
|---|---|---|
| Soma de AuC (1.200 clientes) | **R$ 487 bi** | R$ 75 bi |
| Cliente mediano | R$ 168 milhões | (implícito: dezenas de milhões) |
| AuC médio "Varejo" (784 clientes) | R$ 443 milhões | `< R$ 100 mil` (PROBLEM.md §4.3) |
| AuC médio "Wealth" (115 clientes) | R$ 318 milhões | `> R$ 500 mil` |

O segmento não tem relação nenhuma com o saldo: "Varejo" tem AuC médio **maior** que "Wealth". A EDA (`reports/eda_clientes_v2_bruto.md`) chegou a tratar um saldo de R$ 442 como anomalia enquanto R$ 4 bi por cliente passava como p99 normal.

Além disso, a refatoração ADR-0001 é toda ancorada em **advisor attrition**: assessor troca de firma e carrega a carteira. Esse fenômeno existe em private banking, RIA e family office. Não existe em varejo, onde o cliente não segue um assessor pessoa física. O enunciado varejista do `PROBLEM.md` é resíduo do escopo genérico anterior ao pivô da ADR-0001.

### 1.2 `pct_carteira_exposta` é binária, não percentual

`auc_exposto = saldo_bi * risco_saida_assessor`, e `risco_saida` é o **evento binário realizado** (`int(rand() < p)`), não a probabilidade. Consequência medida em `carteira_exposta_por_assessor.csv`: 257 assessores com `pct_carteira_exposta = 0.0`, 34 com `1.0`, **nenhum valor intermediário**. Uma coluna chamada "percentual da carteira exposta" que só assume 0% ou 100% é `risco_saida` renomeado. A probabilidade contínua já existe no gerador (`prob_saida_calibrada`, `data_processing/nodes.py`) mas nunca é propagada ao cliente.

O teste `test_auc_exposto_agregado_dentro_da_faixa_de_mercado` passa (12,5% agregado, dentro de [5%, 25%]) porque só checa o agregado, o que mascara a degeneração de cada valor individual.

### 1.3 O "55,7% da importância" não tem artefato

`README.md` e o ADR-0001 §6 afirmam que "as 3 features comportamentais somam 55,7% da importância do modelo". Nenhum CSV prova o número: `pipeline.py` só chama `get_feature_importance` para o modelo v1. O único artefato de importância da v2 é o SHAP (`output/shap/v2/feature_importance_shap.csv`), que dá outra base de cálculo (soma dos \|SHAP\| das 3 features sobre o total, aproximadamente 51,7%).

### 1.4 Os documentos de abril são legado da v1, não contrato

O `PROBLEM.md` v1.0 (2026-04-27) e o `refactoring_blueprint.md` foram escritos antes do pivô da ADR-0001. Eles descrevem um projeto diferente: churn reativo por queda de saldo, escala varejo, deploy Streamlit, objetivo declarado de "elevar o portfólio ao nível corporativo (Software 3.0)". A refatoração dos dias 9 e 10 de setembro (ADR-0001 mais esta ADR) é a direção do projeto. Os documentos de abril não são a régua contra a qual a v2 se mede; são histórico a ser reescrito (`PROBLEM.md` vira v2.0) ou arquivado (`refactoring_blueprint.md`, `refatoracao/`). Onde esta ADR cita o `PROBLEM.md`, é para marcar o que muda nele. Ele deixa de ser o contrato da v2; as definições explicitamente herdadas (a queda de AuC que caracteriza churn e a curva de custo) seguem válidas apenas até a reescrita v2.0 no Bloco 3.

---

## 2. DECISÃO (Por Quê?)

### 2.1 Recalibrar o gerador para uma carteira de private banking / multi-family office

Reposicionar o projeto como uma gestora de patrimônio de porte médio-alto, com **AuC de aproximadamente R$ 75 bi distribuído em aproximadamente 1.200 grupos econômicos** (média de R$ 62,5 milhões por relação). Isso é coerente com:

- a dor de negócio da ADR-0001 (advisor attrition e fuga de carteira);
- a escala de mercado real (ver §5, pesquisa 2026-09-10): private banking no Brasil soma R$ 2,3 trilhões em cerca de 70 mil grupos econômicos, média em torno de R$ 30 milhões por grupo; uma boutique ou braço de private focado na ponta alta fica acima dessa média;
- o critério de entrada de private banking (mínimo de R$ 3 milhões investidos, ANBIMA).

**Distribuição por segmento** (calibrada para somar aproximadamente R$ 75 bi, com cauda de concentração realista onde poucos clientes carregam a maior parte do AuC):

| Segmento | Faixa de AuC | Peso (share de clientes) | Distribuição | AuC agregado alvo |
|---|---|---|---|---|
| `Alta Renda` | R$ 3 M a R$ 15 M | ~45% (540) | lognormal, mediana ~R$ 7 M | ~R$ 4,5 bi |
| `Private` | R$ 15 M a R$ 60 M | ~38% (456) | lognormal, mediana ~R$ 28 M | ~R$ 14,5 bi |
| `Wealth` | R$ 60 M a R$ 250 M | ~13% (156) | lognormal, mediana ~R$ 110 M | ~R$ 19 bi |
| `Family Office` | R$ 250 M a R$ 2 bi | ~4% (48) | lognormal, mediana ~R$ 650 M | ~R$ 37 bi |

Os limites de faixa são alvos de calibração, não cortes rígidos: a cauda de cada lognormal pode ultrapassar levemente a faixa nominal do segmento, o que é o comportamento real de uma carteira.

Ajustes acoplados:

- **Driver de churn por saldo relativo:** o modificador `if saldo < 0.1` (corte fixo de R$ 100 M) vira "AuC no tercil inferior do próprio segmento", que é o sinal de negócio real (cliente pequeno para o segmento onde está tende a ser menos aderente).
- **`meses_cliente` mínimo de 6:** um book de private banking não tem relações de semanas, o ciclo de onboarding e alocação já leva meses. O gerador sorteia a partir de 1; passa a sortear de 6 em diante.
- **Segmentos de wealth:** o código usa `["Varejo", "Alta Renda", "Wealth", "Corporate"]`, herança do enunciado genérico da v1. Passa a `Alta Renda / Private / Wealth / Family Office`, a segmentação real de uma gestora de patrimônio (a partir do piso de R$ 3 M da ANBIMA).
- **Número de assessores realista para a Direção B:** hoje `generate_advisors_data(300)` dá 4 clientes por assessor, o que não é private banking (um private banker cobre de 15 a 40 relações). Passa a ~50 assessores para ~1.200 clientes (média de 24 relações). Isso também dissolve o problema dos 9 assessores sem carteira: com menos assessores, todos recebem book.

### 2.2 `auc_exposto` passa a usar a probabilidade contínua

`auc_exposto = saldo_bi * prob_saida_calibrada` (exposição esperada), em vez de `saldo_bi * risco_saida` (evento realizado). `prob_saida_calibrada` é propagada do assessor para o cliente em `attach_advisor_and_behavioral_features`. `pct_carteira_exposta` na agregação vira uma distribuição contínua entre 0 e o teto da faixa de risco do canal. O agregado de mercado (10% a 15%) é preservado por construção porque `E[risco_saida] = prob_saida_calibrada`.

Teste novo em `test_leakage.py`: rejeita a coluna se ela for degenerada (menos de 10 valores distintos, ou variância próxima de zero), que é o gate que teria pego o defeito 1.2.

### 2.3 `pipeline.py` gera e persiste a importância do GB v2

Nova função `get_feature_importance_v2` em `model_training/nodes.py` (lista de features v2 correta) e `catalog.save("feature_importance_v2", ...)` no `pipeline.py`. O número citado nos docs passa a vir desse CSV, ou é removido se a regeneração mostrar valor diferente. O CSV vira a fonte da frase, não a memória de uma sessão.

### 2.4 Coerência com a v2 (o que esta ADR fecha, o que fica)

A v2 (ADR-0001) foi uma migração parcial. O `api.py` já roda `gb_pipeline_v2.pkl`, mas carrega resíduos da v1 que esta ADR precisa fechar para a recalibração não empilhar uma terceira camada de incoerência:

| Resíduo v1 dentro da v2 | Onde | Decisão desta ADR |
|---|---|---|
| Segmentos `Varejo` / `Corporate` | `api.py:50`, `THRESHOLD_MAP`, `OrdinalEncoder` (2x) | Trocados por `Alta Renda / Private / Wealth / Family Office` |
| Thresholds 0,40 / 0,50 / 0,60 (herança do `PROBLEM.md` v1.0) | `api.py:51-56` | **Recalibrados** sobre a saída de probabilidade do modelo v2, por segmento, numa curva de custo assimétrico (FN pior que FP, Recall alvo ≥ 0,75). Registrado em `reports/thresholds_v2.md`. Não é remapear rótulo, é calibrar. |
| `saldo_bi` (vocabulário v1) contra "AuC" em toda a Linguagem Ubíqua da v2 | 16 arquivos | Renomeado para `auc_milhoes` (ver §7) |
| `_risk_level` calcula `threshold` e usa 0,60 fixo | `api.py:294-300` | Passa a usar o threshold calibrado do segmento |
| `auc_at_risk_MM = saldo_bi * 1000 * 0.012 * prob` (fator 1,2% mágico) | `api.py:386` | Trocado por `auc_milhoes * pct_perda_esperada * prob`, onde `pct_perda_esperada` reflete a definição de churn do `PROBLEM.md` (queda de mais de 30% do AuC), não um fator sem origem |

Fica fora (é o trabalho seguinte, não desta ADR): migração de `app.py` abas 1/2/3, `monitor.py`, `agent.py` e `orchestrator.py` do modelo v1 para o v2.

### 2.5 Estado de conformidade com a v2 (mapa do repositório inteiro)

Varredura de 2026-09-10. A v2 (early-warning comportamental como sinal líder, saldo rebaixado a feature defasada, Direção B como produto descritivo separado, book de wealth, dado sintético honesto) é o norte. O repositório está partido: o núcleo seguiu o pivô, a superfície ficou na v1.

**Segue a v2:** `src/data_processing/nodes.py`, `src/model_training/nodes.py`, `pipeline.py`, `tests/test_leakage.py`, `api.py` (carrega o modelo v2, payload early-warning, endpoint Direção B), `shap_analysis_v2.py`, `app.py` aba 4, `docs/adr/0001` e `0002`, `notebooks/01` (assume dado sintético).

**Atrás da v2 (migrar):**

| Arquivo | Resíduo v1 |
|---|---|
| `app.py` abas 1/2/3 | `load_artifacts()` carrega o modelo v1; formulário de predição sem nenhum input comportamental (`:386-407`); segmentos `Varejo`/`Corporate` (`:386`); "Segmento Varejo, maior taxa histórica de churn (25%+)" (`:437`); `saldo < 0.1` e `saldo * 0.012` hardcoded; header "~R$75bi sob custódia" (`:335`); aba 3 = 100% artefato v1 |
| `monitor.py` | Modelo v1 (`:190`), features v1 hardcoded (`:51`, `:148`). O monitor de drift observa o modelo errado |
| `agent.py` | Segmentos v1 (`:331`, `:349`, `:370`, `:407`); lê só `["cliente_id","segmento","saldo_bi"]`; schema de ferramenta com features v1 (`:377`); string "ROC-AUC 0.93" (`:425`), número que não aparece em nenhuma medição (ROC real do v2 no teste: 0,76) |
| `agent_chat.py` | Pergunta de exemplo "cliente Varejo com retorno de 6%" (`:280`) |
| `shap_analysis.py` (v1) | Ainda existe, referenciado por `api.py:593` e README |
| `README.md` | "19 testes" (são 37); tabela de métricas com números stale ou sem artefato; seção "Métricas de Negócio & ROI Estimado" (contradiz a honestidade do ADR-0001 §5); manda rodar `shap_analysis.py` |
| `AGENTS.md` | "19 testes" / "30 testes" (são 37) |
| `conf/base/parameters.yml` | `n_advisors: 300` |

**Contradiz a v2 (retirar ou reescrever):**

| Item | Por quê |
|---|---|
| `notebooks/02_Evolucao_BigData_PySpark.ipynb` | "Evoluir para alto volume de dados (Big Data)", PySpark, cluster, MLflow. Um book de wealth tem ~1.200 clientes, o problema menos Big Data possível. Existe só para mostrar Spark, aponta na direção oposta da v2. Também tem `execution_count` fora de ordem |
| `refactoring_blueprint.md` + `refatoracao/` | Objetivo declarado: "elevar o portfólio ao nível corporativo (Software 3.0)". Norte da v1 (engenharia como fim), não da v2 (credibilidade de domínio). Vira histórico arquivado |
| `PROBLEM.md` v1.0 | Escala varejo, target reativo, thresholds fixos, Streamlit, "Projeto_.ipynb". Legado inteiro. Reescrito como v2.0 |
| `docs/roadmap_data_science_crispdm.md` | Template genérico que fala de "NPS Predictor". Não é deste projeto. Remover |
| `docker-compose.yml` / `envoy.yaml` / `Dockerfile` | Envoy sidecar mais fila Redis mais workers async para um modelo de 1.200 linhas. Só se sustenta se o projeto assumir "showcase de engenharia" como tese secundária explícita. Decisão em aberto, ver §7 |

O plano de execução deste mapa está no `docs/spec/0002-recalibracao-dados.md`, organizado em blocos 2 a 5.

---

## 3. CONSEQUÊNCIAS

**Positivas:**

- O projeto passa a simular uma carteira que um profissional do mercado reconhece: escala, mix de segmento e concentração de AuC batem com dados públicos da ANBIMA e com benchmarks de wealth management de 2025-2026.
- Os três defeitos de honestidade saem antes de qualquer publicação de portfólio.
- `pct_carteira_exposta` vira uma métrica de fato, utilizável no dashboard da Direção B.
- Toda afirmação numérica nos docs passa a ter CSV de origem.

**Negativas:**

- Regenera todos os artefatos: 14 CSVs, `gb_pipeline.pkl`, `gb_pipeline_v2.pkl`, SHAP v1 e v2. As métricas do README mudam e precisam ser reescritas a partir da nova execução.
- `saldo_bi` é referenciado em 16 arquivos. Renomear para `auc_milhoes` (leitura mais natural: 45,0 em vez de 0,045) é mecânico mas amplo. Decisão registrada em §7.
- A EDA (`reports/eda_clientes_v2_bruto.md`) foi escrita sobre os números antigos e precisa ser regenerada.
- Muda `n_advisors` de 300 para ~50. Os testes da Direção B que dependem de contagem de assessor precisam de revisão.
- Adiciona a calibração de threshold do modelo v2 ao escopo (não estava no plano inicial), porque aplicar threshold v1 sobre modelo v2 já era incoerente antes desta ADR e a recalibração de escala força a decisão.
- O escopo total de conformidade com a v2 (§2.5) é maior do que o Bloco 2 desta ADR: cobre a migração da apresentação (`app.py`, `monitor.py`, `agent.py`), os notebooks e a decisão sobre a stack de infra. O `docs/spec/0002` organiza isso em blocos 2 a 5, executados em sessões separadas.

---

## 4. ALTERNATIVAS DESCARTADAS

| Opção | Por que foi rejeitada |
|---|---|
| Manter a escala varejista do `PROBLEM.md` (Varejo < R$ 100 mil dominante) | 1.200 contas varejo somam de R$ 1 bi a R$ 2 bi, e aí o número "R$ 75 bi" sai da narrativa. Pior: advisor attrition não faz sentido em varejo, e a Direção B (fuga de carteira quando o assessor troca de firma) perde a âncora de negócio. |
| Só trocar o texto "R$ 75 bi" pelo número real (R$ 487 bi) | R$ 406 milhões de AuC médio por cliente não existe em nenhuma gestora real. O problema não é o rótulo, é a distribuição. |
| Só documentar o defeito 1.2 sem corrigir | Uma coluna chamada "percentual" que só vale 0 ou 100 continua enganando quem lê o dashboard. |
| Estender a ADR-0001 com uma §7 em vez de abrir ADR-0002 | A ADR-0001 já tem uma correção pós-implementação (§6). Uma terceira camada de correção dentro do mesmo documento fica ilegível. Recalibração de escala de dado, redefinição de segmento e mudança de threshold é decisão própria. |

---

## 5. PESQUISA DE MERCADO (âncora dos parâmetros, 2026-09-10, WebSearch)

**Private banking no Brasil (ANBIMA):**
- Patrimônio do segmento private: R$ 2,30 trilhões ao fim de 2024 (crescimento de 8,7% no ano). Faz parte de R$ 7,3 trilhões investidos por pessoas físicas no país.
- Em setembro de 2018 o segmento somava R$ 1,05 trilhão em 58.300 grupos econômicos, média de aproximadamente R$ 18 milhões por grupo. Projetando o crescimento para 2024, a média por grupo fica na ordem de R$ 25 a 33 milhões.
- Critério de entrada: mínimo de R$ 3 milhões em ativos financeiros, podendo ser patrimônio familiar, não só individual.

**Advisor movement e attrition (mercado dos EUA, 2025):**
- Wirehouses tiveram attrition líquida de 562 assessores. Só 22,5% dos assessores de wirehouse que se movem permanecem no canal; 26,4% vão para RIA.
- Stickiness do canal RIA: 97,4% dos assessores de origem RIA permanecem no canal. RIA é o único canal com entrada líquida estrutural sustentada.
- Cerca de 57 mil assessores produtores saíram da indústria em 2025 e 53 mil entraram.
- Transições relevantes de 2025: breakaway de R$ 129 bi da Merrill (OpenArc), duas equipes de mais de US$ 6 bi saindo do UBS no fim de 2025.

**Retenção de cliente em wealth management (2025):**
- Firmas reportam retenção anual de clientes entre 92% e 97% (churn anual de 3% a 8%).
- 61% dos clientes trocariam de assessor por perda de confiança; 54% por baixo desempenho (CapIntel Investor Engagement Survey 2025).
- Transferência geracional de patrimônio: 81% dos herdeiros planejam trocar de firma em 1 a 2 anos após receber os ativos. Candidato a feature ou driver de churn futuro (`evento_sucessao`), fora do escopo desta ADR.

**Como a pesquisa entra na calibração:**
- Média alvo de AuC por relação (R$ 62,5 milhões) fica acima da média de mercado (R$ 25 a 33 milhões), coerente com uma gestora skewed para a ponta alta, que é onde o risco de fuga de carteira por advisor attrition dói mais.
- Taxa base de churn por segmento (4% a 10%) fica na faixa dos 3% a 8% de churn anual de wealth, com um pouco de folga porque o target aqui é erosão de AuC (queda acima de 30% por 2 meses), não encerramento de conta.
- Risco base de saída de assessor por canal (`Wirehouse` 0,15 > `Broker-Dealer` 0,11 > `RIA` 0,06) mantém a ordenação que a pesquisa sustenta (wirehouse mais móvel, RIA mais aderente).

Fontes:
- [Patrimônio do private banking avança 8,1% no ano, ANBIMA](https://www.anbima.com.br/pt_br/imprensa/patrimonio-do-private-banking-avanca-8-1-no-ano.htm)
- [Estatísticas de Private, ANBIMA](https://www.anbima.com.br/pt_br/informar/relatorios/varejo-private-e-gestores-de-patrimonio/boletim-de-private-e-varejo/integra.htm)
- [Private banking no Brasil, Safra](https://oespecialista.safra.com.br/private-banking-no-brasil/)
- [Of Myths and Moving: 2025, WealthManagement.com](https://www.wealthmanagement.com/recruiting/of-myths-and-moving-2025)
- [U.S. Wealth Advisor Movement Report 2026, AdvizorPro](https://advizorpro.com/post/us-wealth-advisor-movement-report)
- [Average Client Retention Rate for Financial Advisors, SmartAsset](https://smartasset.com/advisor-resources/financial-advisor-client-retention-rate)
- [How Can RIAs Improve Client Retention in 2026, 11th.com](https://11th.com/blog/educational/how-can-rias-improve-client-retention-in-2026/)

---

## 6. IMPACTO E CRITÉRIO DE SUCESSO (executável, gate da base)

- **`auc_agregado_bate_com_narrativa`:** teste em `test_leakage.py` que soma o AuC do dataset gerado e falha se sair da faixa de R$ 65 bi a R$ 85 bi. O número da narrativa vira asserção.
- **`pct_carteira_exposta_nao_degenerada`:** teste que falha se a coluna tiver menos de 10 valores distintos ou desvio padrão abaixo de 0,02.
- **`ordenacao_auc_por_segmento`:** teste que falha se a mediana de AuC não crescer de `Alta Renda` para `Private` para `Wealth` para `Family Office`.
- **`clientes_por_assessor_plausivel`:** teste que falha se a média de clientes por assessor sair da faixa de 12 a 40 (book de private banking).
- **`feature_importance_v2` versionado:** `output/data/feature_importance_v2.csv` existe após `pipeline.py`, e a soma da importância das 3 features comportamentais está registrada nele.
- **`thresholds_v2` versionado:** `reports/thresholds_v2.md` existe, com o threshold por segmento, a fórmula de custo usada e o Recall/Precision resultante no split de validação. `api.py` importa esses valores, não os hardcoda.
- **Regressão da ADR-0001:** `test_recall_early_warning_vs_baseline_reativo` continua passando (v2 supera v1 em recall). A recalibração de escala não pode inverter o resultado da Direção A.
- **Não regressão de contrato:** os 37 testes atuais continuam passando após ajuste de nomes e perfis de exemplo.

---

## 7. DECISÕES (resolvidas contra a definição da v2)

As três escolhas abaixo não são independentes: cada uma se resolve pela pergunta "o que a v2 (ADR-0001) exige?", não por preferência estética.

1. **Renomear `saldo_bi` para `auc_milhoes`.** A Linguagem Ubíqua da v2 é toda construída em "AuC" (`queda de AuC`, `AuC exposto`, `AuC under Custody`). `saldo_bi` é o único ponto do código que ainda fala v1. O rename alinha o código ao vocabulário da própria v2, além de tornar os valores legíveis (45,0 em vez de 0,045). 16 arquivos, mecânico, os testes pegam regressão.
2. **Recalibrar os thresholds sobre o modelo v2.** A v2 mudou o que o modelo prevê (sinal comportamental antecedente em vez de saldo reativo), logo a distribuição de `P(churn)` mudou. Aplicar os thresholds do `PROBLEM.md` v1.0 (calibrados para o modelo reativo) sobre a saída da v2 não é coerente com a v2. A calibração usa a curva de custo herdada do `PROBLEM.md` §4 sobre um split de validação separado do teste, por segmento, e o resultado fica em `reports/thresholds_v2.md`. Segmento sem suporte estatístico mínimo usa o threshold global e registra o fallback; não se transforma ruído de poucos eventos em regra comercial. Como a curva depende da escala nova de dado, isso só pode ser feito depois da recalibração, dentro deste trabalho, não antes nem depois.
3. **Corrigir o número de assessores, não tratar o sintoma.** Os 9 assessores sem carteira são efeito de `n_advisors=300` para 1.200 clientes (4 relações cada), que não é a razão de um book de private banking. Com ~50 assessores (24 relações cada, faixa real), todos recebem carteira e a Direção B passa a medir risco de fuga sobre books de tamanho plausível. O `pipeline.py` registra quantos assessores ficaram sem book (esperado: zero ou perto disso).

### 7.1 Ainda em aberto (decisão do Luiz, não bloqueia os blocos 2 a 4)

- **A camada de infraestrutura (Docker, Envoy, fila Redis, worker async) fica ou sai?** Ela é do blueprint da v1 e é desproporcional a um modelo de 1.200 linhas. Duas saídas honestas: (a) manter e assumir "showcase de engenharia" como tese secundária explícita no README, com uma frase dizendo que é deliberadamente sobre-construída para demonstrar os padrões; (b) simplificar para uma API síncrona e mover a stack async para um `docs/arquitetura_escala.md` como "como isto escalaria". Bloco 5 executa a escolhida.
- **Os dois notebooks:** `01` reenquadrar para wealth ou aposentar; `02` (Big Data) aposentar ou reescrever como apêndice opcional. Bloco 5.

---

## 8. LINKS RELACIONADOS

- [[0001-refatoracao-early-warning-advisor-attrition]] , a decisão do pivô; esta ADR corrige o que ela não terminou
- [[PROBLEM.md]] , legado da v1; reescrito como v2.0 no Bloco 3, não é referência para esta ADR
- `docs/spec/0002-recalibracao-dados.md` , spec de implementação, blocos 2 a 5
- Gate da base (`.claude/rules/dados.md`): "Critério escrito em ADR precisa aparecer em codigo executavel", "Proporcao reportada sempre com n e intervalo", "Erro implausivel vira nulo"
