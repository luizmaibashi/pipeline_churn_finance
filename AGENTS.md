# AGENTS.md: pipeline_churn_finance

Projeto de portfólio fictício sobre early-warning de churn em wealth management. O dado é sintético; métricas offline não devem ser apresentadas como efeito de negócio real.

## Linguagem ubíqua

| Termo | Significado |
|---|---|
| AuC | Assets under Custody, em milhões de reais na coluna `auc_milhoes`. |
| Churn | Erosão de AuC superior a 30%, persistente por dois meses; target simulado no projeto. |
| Early-warning comportamental | Sinal de relacionamento anterior ao churn: recência, cadência e latência de resposta. |
| Advisor attrition | Risco de o assessor deixar a firma, distinto de churn do cliente. |
| AuC exposto | `auc_milhoes × prob_saida_assessor`; agregação descritiva por assessor, fora do modelo de churn. |

## Estado do projeto

- **v2 dado/modelo:** concluída no Bloco 2 da ADR-0002. Gera cerca de R$ 76 bi para 1.200 relações de wealth, com 50 assessores.
- **v2 serving:** `api.py` carrega `gb_pipeline_v2.pkl` e `thresholds_v2.csv`; só aceita segmentos de wealth, AuC de 0 a 2.000 milhões e relações com ao menos seis meses.
- **v2 apresentação:** concluída no Bloco 4. `app.py`, `monitor.py`, `agent.py` e `agent_chat.py` rodam o modelo v2, features de early-warning e segmentos de wealth. `shap_analysis.py` v1 aposentado (só `shap_analysis_v2.py`).
- **v2 serving contract:** `serving_contract.py` (ADR-0003) é a fonte única de features, segmentos, thresholds, regras de risco/fluxo e limiares de fator de risco. `api.py`/`app.py`/`agent.py`/`monitor.py` são consumidores finos.
- **v2 notebooks + infra:** concluído no Bloco 5. `01` reenquadrado para wealth, `02` mantido como apêndice de Spark, stack Docker/Envoy assumida como showcase de engenharia explícito no README. MLOps-lite v1 (`version_manager.py`) removido.
- **evidência:** 55 testes. `feature_importance_v2.csv`, thresholds e SHAP v2 são os artefatos citáveis.
- **limite:** no teste há 28 eventos. A diferença de recall v2 menos v1 tem IC95% que inclui zero; não declarar ganho robusto.

## Regras

- Antes de mudar lógica, leia `PROBLEM.md`, ADR-0001, ADR-0002 e a spec 0002.
- Todo valor fora do domínio de entrada deve gerar erro explícito, nunca clamp silencioso.
- Features só podem usar informação até D-0; ajustar transformadores somente no treino.
- Proporções precisam de `n` e intervalo quando houver inferência.
- Rode `python -m pytest -q` após alterações. Para regenerar artefatos: `python pipeline.py`, `python shap_analysis_v2.py`.
