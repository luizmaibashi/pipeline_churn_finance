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
- **evidência:** 45 testes. `feature_importance_v2.csv`, thresholds e SHAP v2 são os artefatos citáveis.
- **limite:** no teste há 28 eventos. A diferença de recall v2 menos v1 tem IC95% que inclui zero; não declarar ganho robusto.
- **pendente:** Bloco 4 migra `app.py`, monitor e agentes da v1. Bloco 5 decide notebooks e infraestrutura.

## Regras

- Antes de mudar lógica, leia `PROBLEM.md`, ADR-0001, ADR-0002 e a spec 0002.
- Todo valor fora do domínio de entrada deve gerar erro explícito, nunca clamp silencioso.
- Features só podem usar informação até D-0; ajustar transformadores somente no treino.
- Proporções precisam de `n` e intervalo quando houver inferência.
- Rode `python -m pytest -q` após alterações. Para regenerar artefatos: `python pipeline.py`, `python shap_analysis.py`, `python shap_analysis_v2.py`.
