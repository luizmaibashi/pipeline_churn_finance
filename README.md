# Pipeline Churn Finance

Um projeto de portfólio sobre early-warning de churn em wealth management. A base inteira é sintética e calibrada para uma gestora fictícia de 1.200 grupos econômicos, com R$ 76,05 bilhões de AuC.

O ponto de partida foi simples: prever churn só pela queda de saldo chega tarde. O pipeline testa sinais de relacionamento, como tempo desde o último contato, queda na cadência e demora na resposta. Em paralelo, mantém uma visão separada da carteira exposta quando um assessor pode sair da firma.

Isso não é uma demonstração de resultado comercial. É uma demonstração de método: como estruturar um problema de dados sintéticos, separar duas perguntas de negócio e não vender uma métrica offline como impacto financeiro.

## O que o projeto entrega

- Um classificador v2 de churn com sinais comportamentais e tratamento de nulos estruturais.
- Uma tabela por assessor com AuC exposto esperado, calculado a partir da probabilidade de saída do assessor.
- API FastAPI, dashboard Streamlit, monitor de drift e agente de dados, todos no modelo v2.
- Artefatos de explicabilidade SHAP do modelo v2.
- 55 testes de contrato, dado, modelo e API.

## Retrato atual do dado sintético

| Indicador | Valor |
|---|---:|
| Relações | 1.200 |
| AuC total | R$ 76,05 bi |
| Mediana de AuC | R$ 16,84 M |
| AuC exposto esperado | R$ 9,30 bi (12,2%) |
| Assessores | 50 |
| Assessores sem book | 0 |

Os quatro segmentos são Alta Renda, Private, Wealth e Family Office. A concentração é proposital: Family Office representa uma parcela pequena das relações e grande parte do AuC.

## Avaliação, sem extrapolar

No split de teste de 240 relações, com 28 eventos de churn, a v2 teve recall de 7,14% contra 0% da baseline reativa. O IC95% bootstrap da diferença foi [0,00; 17,87] p.p., portanto o experimento ainda não prova ganho robusto. O ROC-AUC da v2 no mesmo split foi 0,7209; em validação cruzada de cinco folds, o recall foi 17,14% ± 4,16 p.p.

As três features comportamentais somam 63,2% da importância do Gradient Boosting v2. Esse número está em `output/data/feature_importance_v2.csv`; ele descreve o gerador atual, não um comportamento de clientes reais.

## Como executar

```bash
python pipeline.py
python shap_analysis_v2.py
python -m pytest -q
uvicorn api:app --reload --port 8000
```

O pipeline gera os thresholds por segmento em `output/data/thresholds_v2.csv`. Segmentos com evidência insuficiente recebem o threshold global, marcado como `global_fallback`; a API lê esse arquivo na inicialização e não mantém uma regra paralela no código.

## Estrutura

```text
src/                 geração, limpeza e treino
pipeline.py          orquestração e persistência de artefatos
serving_contract.py  fonte única do contrato de scoring v2 (features, thresholds, regras — ADR-0003)
api.py               serviço de predição
app.py               dashboard Streamlit
monitor.py           monitor de data drift
agent.py             agente de dados sobre a API
agent_chat.py        interface de chat do agente (Streamlit)
tests/               contratos de dado, modelo e API
notebooks/           01: pipeline v2 em pandas/sklearn · 02: apêndice Spark
docs/adr/            decisões de arquitetura
docs/spec/           contrato de implementação
reports/             EDA e thresholds
```

## Camada de infraestrutura (showcase de engenharia)

O `docker-compose.yml` sobe API, worker assíncrono, fila Redis e um sidecar Envoy. Para um modelo de ~1.200 relações isso é **deliberadamente sobre-construído**: existe para demonstrar os padrões de serving distribuído (fila de jobs, worker, proxy), não porque o volume exige. Uma API síncrona resolveria o caso real. Ver ADR-0002 §7.1.

O notebook `02_Evolucao_BigData_PySpark.ipynb` tem a mesma natureza: é um apêndice que mostra o pipeline em Spark/MLflow para o cenário hipotético de a carteira crescer 100x, não a arquitetura recomendada para a escala atual.

## Limites e próximos passos

O dado é sintético. Um teste real exigiria série temporal observada, definição operacional de churn e experimento de retenção. As telas, o monitor e o agente já rodam o modelo v2 (Bloco 4); o que resta é opcional e está listado nos ADRs.

Consulte [PROBLEM.md](PROBLEM.md) e os ADRs para o contrato e as decisões: [ADR-0001](docs/adr/0001-refatoracao-early-warning-advisor-attrition.md) (pivô early-warning), [ADR-0002](docs/adr/0002-recalibracao-gerador-escala-wealth.md) (escala de wealth), [ADR-0003](docs/adr/0003-modulo-serving-contract.md) (contrato de scoring único).
