# AGENTS.md — pipeline_churn_finance

Projeto de portfólio fictício: pipeline de previsão de churn para carteira simulada de R$ 75bi de AuC (Assets under Custody), gestora de investimentos de alta renda. Arquitetura de nível corporativo (Kedro-style, FastAPI assíncrono, Envoy, agente de drift) — ver `refactoring_blueprint.md` para o desenho técnico completo e `PROBLEM.md` para o contrato de dados v1.

## Linguagem Ubíqua

| Termo | Significado |
|---|---|
| **AuC** | Assets under Custody — capital do cliente sob custódia da gestora |
| **Churn (v1, reativo)** | `AuC atual < 70% do máximo em 6 meses, por 2 meses consecutivos` — definição contratual em `PROBLEM.md`, mede o efeito já ocorrido |
| **Early-warning comportamental** | Sinal antecedente à queda de AuC, calculado a partir de cadência/qualidade de interação cliente-assessor (frequência de contato, latência de resposta, proxy de sentimento) — não do saldo |
| **Advisor attrition** | Risco do assessor responsável pela conta deixar a firma — causa raiz distinta do churn decidido pelo cliente |
| **AuC exposto** | Fração da carteira de um assessor que tende a migrar junto se ele sair (proxy de risco herdado) — métrica **descritiva** de `carteira_exposta_por_assessor`, não feature do classificador de churn (ver correção §6 do ADR-0001) |
| **Sinal reativo vs. antecedente** | Reativo = mede o efeito já ocorrido; antecedente = mede a causa antes do efeito aparecer |
| **Janela de observação / previsão** | 90 dias de features (D-90 a D-0) → previsão de 30 dias (D+30), contrato anti-leakage em `PROBLEM.md` §3 |

## Estado do projeto

- **v1 (engenharia):** completa — Kedro catalog, API assíncrona (Redis/SQLite dual-mode), Envoy sidecar, agente de IA com fallback offline, 19 testes passando.
- **v1 (dado):** gap conhecido — zero teste de leakage temporal apesar do contrato R-01 a R-05 em `PROBLEM.md` ser rígido.
- **v2 (fechado no modelo/dado, ver [ADR-0001](docs/adr/0001-refatoracao-early-warning-advisor-attrition.md)):** Direção A (early-warning comportamental) é o classificador de churn — supera a baseline v1 (CV 5-fold recall 0,2958±0,0358 vs. 0,0875±0,0306; 3 features comportamentais somam 55,7% da importância). Direção B (risco de saída de assessor / AuC exposto) é agregação separada (`carteira_exposta_por_assessor`), não feature — dashboard de risco de carteira, não preditor de churn de cliente. Dataset 100% sintético, calibrado com parâmetros estatísticos reais de mercado (pesquisa 2026-09-09), 30 testes passando (inclui gate de CI que trava o critério de sucesso do ADR).
- **v2 (apresentação — parcial):** `api.py` migrado para o Pipeline v2 (payload early-warning + nulo estrutural derivado; endpoint `GET /advisors/exposed-portfolio` para a Direção B). Pendente: `app.py` (dashboard), `monitor.py`/`agent.py` (drift ainda sobre features v1), `README.md`. SHAP v2 já existe (`shap_analysis_v2.py`, `output/shap/v2/`).

## Decisões arquiteturais

Ver `docs/adr/` — gerar novo ADR para toda decisão que afete >1 sistema ou trade-off complexo, seguindo `/grill-with-docs` da base de conhecimento.

## Regras herdadas da base de conhecimento

Este projeto é repositório próprio (`.git` ignorado pela base `Base_de_Conhecimento`), mas segue as regras de `.claude/rules/dados.md` da base: sem leakage, sem imputação injustificada sem log, proporção sempre com `n`+IC, critério de ADR precisa virar código executável.
