# ADR-0003: Módulo `serving_contract` — Fonte Única do Contrato de Scoring v2

**Data:** 2026-09-10
**Status:** Accepted
**Proposto por:** Luiz Maibashi
**Estende:** [ADR-0001](0001-refatoracao-early-warning-advisor-attrition.md), [ADR-0002](0002-recalibracao-gerador-escala-wealth.md)

---

## 1. CONTEXTO (O Quê?)

A migração v1→v2 (blocos 2–5 do `docs/spec/0002`) fechou em 2026-09-10. O passe de
elegância (`/simplify`, 4 agentes de revisão) apontou, por unanimidade, um único
achado estrutural: **o contrato de scoring da v2 está copiado à mão em todo lugar
que precisa pontuar um cliente**, e as cópias já divergiram.

### 1.1 O que está duplicado

| Regra | Onde vive hoje |
|---|---|
| `FEATURES_V2_BASE` (11 colunas, ordem canônica) | `src/model_training/nodes.py`, `api.py`, `app.py`, `monitor.py` (remontada de partes), `shap_analysis_v2.py`, `tests/test_model.py` — **6 cópias** |
| Lista dos 4 segmentos de wealth | `api.py` (`SEGMENTOS_VALIDOS`), `agent.py` (`SEGMENTOS_V2`), `app.py` (`SEGMENTOS_V2`), + 4 enums repetidos no `TOOLS_SCHEMA` do agente |
| Carga de `thresholds_v2.csv` → `{segmento: threshold}` | `api._load_threshold_map` (valida cobertura, levanta), `agent._threshold_map` (devolve `{}` no erro), `app.load_thresholds_v2` (sem tratamento) |
| Regra de risco (`prob ≥ thr` → ALTO; `≥ thr·0,6` → MÉDIO) | `api._risk_level`, `agent._risk_from_threshold`, `app.risco_por_threshold` + `app.risk_badge` + `app.gauge_chart` (fator `0,6` inline ~7×) |
| Fluxo operacional (`Wealth`/`Family Office` ou AuC ≥ 250 → revisão humana) | `api._flow`, `agent._flow_v2`, `app.py` (ternário inline) |
| Fator de perda de AuC no churn (`0,30`) | `api.py` (literal solto 2×), `agent.PCT_PERDA_CHURN`, `app.PCT_PERDA_CHURN` |

### 1.2 Divergência já observada (não hipotética)

- Nível de risco: `api.py`/`agent.py` retornam `"ALTO"`; `app.py` retorna `"Alto"`.
- Label de fluxo: `api.py`/`agent.py` usam `"REVISAO_HUMANA (especialista)"`; `app.py` usava `"REVISÃO HUMANA (especialista)"`.
- Segmento sem threshold: `api.py` levanta `HTTPException(503)`; `agent.py` cai em `prob ≥ 0,5`; `app.py` cai em `thr_map.get(seg, 0.5)`. **Três respostas diferentes** para a mesma condição — e a de `app.py`/`agent.py` é justamente a "regra v1 embutida" que a v2 removeu de propósito.
- `monitor.py` monta `FEATURES_V2` como `NUMERIC_FEATURES + ["segmento", flags…]`, com `segmento` no meio da lista, enquanto o modelo foi treinado com `segmento` primeiro. Funciona hoje porque o `ColumnTransformer` seleciona por nome, mas a ordem canônica só existe em `nodes.py`.

### 1.3 Restrições

- **`agent.py` não importa `sklearn` hoje.** Puxar o contrato de `src/model_training/nodes.py` (que importa `sklearn`, `transformers`) arrastaria a árvore de ML inteira para o agente e para qualquer consumidor leve do contrato.
- **`api.py` é infra pesada** (FastAPI, CORS, pydantic, `src.job_queue` que sobe threads). Importar de `api.py` também não serve.
- O label de fluxo tem **duas formas legítimas**: a forma de contrato (`"REVISAO_HUMANA (especialista)"`, o que a API devolve no JSON — mudar isso é breaking change) e a forma de UX (o dashboard quer "Revisão humana (especialista)" legível). São renderizações da **mesma decisão booleana**, não duas regras.
- `nodes.py` monta `FEATURES_V2_BASE = FEATURES_BASE + FEATURES_V2_EXTRA` e usa `FEATURES_V2_EXTRA` / `FEATURES_V2_NUMERICAS_COM_NULO` separadamente no `_build_preprocessing_v2`. Mover a lista final exige decidir a direção da dependência.

---

## 2. DECISÃO (Por Quê?)

Criar **`serving_contract.py` na raiz do projeto** (ao lado de `transformers.py`),
dependência única `pandas`, contendo:

```python
SEGMENTOS_VALIDOS = ["Alta Renda", "Private", "Wealth", "Family Office"]
FEATURES_V2_BASE  = [...]                      # ordem canônica, 11 colunas
PCT_AUC_LOSS_ON_CHURN = 0.30                   # queda de AuC que caracteriza churn (PROBLEM.md v2.0 §2)
RISK_MID_FACTOR       = 0.6                    # fronteira MÉDIO = fração do threshold ALTO
HUMAN_REVIEW_AUC_MM   = 250                    # AuC (R$ mi) acima do qual o fluxo vai a especialista

def load_threshold_map(path=...) -> dict[str, float]   # valida cobertura dos 4 segmentos, levanta se faltar
def risk_level(prob, segmento, thr_map) -> str         # "ALTO" | "MEDIO" | "BAIXO"
def needs_human_review(segmento, auc_milhoes) -> bool   # decisão booleana
def operational_flow(segmento, auc_milhoes) -> str      # forma de contrato: "REVISAO_HUMANA (especialista)" | "AUTO → CRM"
def auc_at_risk_mm(auc_milhoes, prob) -> float          # auc_milhoes * PCT_AUC_LOSS_ON_CHURN * prob
```

**Direção da dependência do `FEATURES_V2_BASE`:** `serving_contract` é a fonte;
`src/model_training/nodes.py` faz `from serving_contract import FEATURES_V2_BASE`
e mantém `FEATURES_BASE` / `FEATURES_V2_EXTRA` locais (são detalhe de treino),
com um `assert FEATURES_BASE + FEATURES_V2_EXTRA == FEATURES_V2_BASE` no import —
falha na hora se as duas visões divergirem, não meses depois num teste que
alguém pulou.

**Forma de contrato vs. forma de UX:** `operational_flow()` (a string que a API
devolve) é a forma canônica e não muda. O dashboard chama `needs_human_review()`
e formata seu próprio rótulo legível — uma renderização, não uma segunda regra.
O nível de risco canônico é `"ALTO"/"MEDIO"/"BAIXO"` (forma da API, sem acento);
`app.py` mapeia para exibição num único ponto.

**Política de segmento ausente:** `load_threshold_map` garante os 4 segmentos ou
levanta. `risk_level` faz `thr_map[segmento]` — como a entrada já é validada
contra `SEGMENTOS_VALIDOS`, um `KeyError` aqui é bug, não caminho normal.
**Nenhum fallback silencioso para `0,5`** em lugar nenhum: os caminhos offline do
agente e do dashboard passam a dizer "rode `python pipeline.py`" em vez de
fabricar um corte.

### Razão principal (adaptada — projeto fictício, sem ROI de negócio)

*"Se não fizermos":* a v2 repete o erro que gerou a própria v2. A auditoria de
2026-09-10 (ADR-0002 §1) encontrou 3 defeitos de honestidade que sobreviveram
porque *nada quebrava* — número num doc, cópia de lista, fator mágico. O contrato
de scoring espalhado em 6 arquivos é a mesma armadilha: a métrica offline
continua boa enquanto o dashboard e a API divergem em silêncio. É exatamente o
*training-serving skew* que `.claude/rules/dados.md` proíbe ("a lógica de feature
engineering vive num lugar só").

*"Se fizermos":* trocar o fator de perda de churn, renomear um segmento ou
reordenar uma feature vira **uma edição em um arquivo**. O gate ML "critério de
ADR precisa aparecer em código executável" passa a ter um alvo real. As rotas da
API viram wrappers finos; o fallback offline do agente chama a mesma função que a
API, não uma reimplementação sincronizada à mão.

---

## 3. CONSEQUÊNCIAS

**Positivas:**

- 6 cópias de `FEATURES_V2_BASE` → 1. Ordem canônica deixa de ser implícita.
- As 3 políticas de "segmento sem threshold" → 1 (levanta, nunca fabrica corte).
- `RISK_MID_FACTOR` e `HUMAN_REVIEW_AUC_MM` viram constantes nomeadas únicas (hoje: literais em ~10 pontos).
- `api.py` deixa de ter `0.30` como literal solto — o "canônico" para de ser o que mais diverge.
- `nodes.py` ganha `assert` de consistência no import (fail-fast).

**Negativas:**

- Toca 7 arquivos + 1 novo (`api.py`, `app.py`, `agent.py`, `agent_chat.py` indireto, `monitor.py`, `src/model_training/nodes.py`, `shap_analysis_v2.py`, `tests/`).
- `nodes.py` passa a importar um módulo da raiz — já faz isso com `transformers`, mas amplia a superfície.
- `serving_contract.py` na raiz (não em `src/`) é uma inconsistência de layout deliberada: `src/` puxa `sklearn`; o contrato precisa ficar leve para o agente. Documentado aqui para não parecer descuido.
- Mudança de comportamento pequena e intencional: `app.py` para de pontuar segmento desconhecido a `0,5` — passa a erro explícito (alinhado à API). Nenhum fluxo real gera segmento fora dos 4 (o `selectbox` só oferece os válidos).

**Timeline:** implementação ~2–3 h (mecânico, guiado por testes). Validação: os 45
testes atuais + 1 novo (`tests/test_serving_contract.py`) verdes.

---

## 4. ALTERNATIVAS DESCARTADAS

| Opção | Vantagem | Por que rejeitada |
|---|---|---|
| **Deixar como está, confiar em revisão** | Zero trabalho | A divergência já aconteceu com revisão ativa. `dados.md` trata isso como falha estrutural, não de disciplina. |
| **Importar o contrato de `src/model_training/nodes.py`** | Sem arquivo novo; `nodes.py` já tem a lista | Arrasta `sklearn` + `transformers` para `agent.py` e qualquer consumidor leve. O agente roda em modo demo sem nada de ML instalado. |
| **Importar de `api.py`** | `api.py` já é "a autoridade" | `api.py` sobe FastAPI, CORS, fila de jobs com threads. Import só para pegar uma lista de strings é absurdo. |
| **`serving_contract.py` dentro de `src/`** | Layout consistente | `src/__init__` e vizinhos podem puxar `sklearn` transitivamente; e `nodes.py` importando de um irmão de `src/` inverte a hierarquia de forma mais confusa que importar da raiz. |
| **Teste-só de igualdade entre as cópias** (sem módulo) | Mínimo de mudança | Detecta drift só se o teste rodar; não elimina as 6 cópias; não resolve as 3 políticas divergentes de threshold. É gate, não conserto. |
| **Só o dashboard adota; API/agente/monitor ficam** | Escopo menor | O dashboard não é o que diverge mais — a API é (2 literais `0.30`, `monitor` com ordem de coluna própria). Meia-consolidação mantém o skew. |

---

## 5. IMPACTO E CRITÉRIO DE SUCESSO (executável)

- **`test_serving_contract.py`:** assere `serving_contract.FEATURES_V2_BASE == src.model_training.nodes.FEATURES_V2_BASE`, `== ` a lista que `api.py` passa a `predict_proba`, e `== ` a de `shap_analysis_v2.py`. Falha se qualquer cópia ressurgir.
- **`test_serving_contract.py`:** `load_threshold_map` sobre um CSV sem um segmento **levanta** (não devolve mapa parcial).
- **Regressão:** `grep -n "0\.30\|PCT_PERDA\|SEGMENTOS_V2\|_risk_from_threshold\|_flow_v2\|risco_por_threshold\|_load_threshold_map" api.py app.py agent.py monitor.py` só retorna `import`/uso de `serving_contract`, nunca definição.
- **Não regressão:** os 45 testes atuais continuam verdes; `AppTest` do `app.py` sem exceção; smoke `TestClient` da `api.py` (`/predict`, `/clients/high-risk`) idêntico ao de hoje.
- **Contrato da API inalterado:** `operational_flow()` devolve exatamente as strings de hoje (`"REVISAO_HUMANA (especialista)"`, `"AUTO → CRM"`); nenhum response schema muda.

**Cenário de regressão a vigiar:** se um futuro candidato de modelo mudar a lista
de features **sem** passar pelo `serving_contract` (ex.: alguém edita só
`nodes.py`), o `assert` no import de `nodes.py` quebra o `pipeline.py` na hora —
que é o comportamento desejado.

---

## 6. REFERÊNCIAS

- [[0001-refatoracao-early-warning-advisor-attrition]], [[0002-recalibracao-gerador-escala-wealth]]
- `.claude/rules/dados.md` — "Paridade treino-serviço (training-serving skew)": a lógica vive num lugar só
- `docs/spec/0002-recalibracao-dados.md` — Bloco 4 (migração da apresentação), de onde o débito nasceu
- Passe de elegância `/simplify` de 2026-09-10 (4 agentes: reuse, simplification, efficiency, altitude — convergência unânime)
- `brain/sessions/SESSAO_10_09_2026_pipeline_churn_v2.md`

---

## ✅ CRITÉRIOS DE ACEITAÇÃO

- [x] Trade-off documentado (forma de contrato vs. UX; direção da dependência; raiz vs. `src/`)
- [x] Alternativas rejeitadas com motivo técnico
- [x] Impacto quantificado (6 cópias → 1; 3 políticas → 1; ~10 literais → constantes)
- [x] Critério de sucesso testável (`test_serving_contract.py` + grep de regressão)
- [x] Cenário de regressão identificado (`assert` no import de `nodes.py`)

---

## 7. SEGUIMENTO (2026-09-10, auditoria de conformidade)

A auditoria pós-fechamento apontou dois resíduos que esta ADR deixou para trás:

1. **`app.py` ainda tinha `thr_map.get(segmento, 0.5)`** — o fallback silencioso que a §2 e a §3 diziam ter removido. Corrigido para `thr_map[segmento]` (o loader já garante os 4 segmentos; `KeyError` aqui é bug, não caminho).

2. **Limiares de fator de risco ainda duplicados** entre `api._recommended_action` e o painel de insights do `app.py` (`dias > 45`, `variação < -0,2`, `retorno < 9`, `resposta > 40`, `auc < 15`, `qtd_produtos == 1`) — a mesma armadilha de cópia que motivou esta ADR, só que fora do contrato de scoring. Fechado agora:
   - `serving_contract.py` ganhou as constantes (`DIAS_SEM_CONTATO_ALERTA`, `QUEDA_CADENCIA_ALERTA`, `TEMPO_RESPOSTA_ALERTA_H`, `RETORNO_12M_BAIXO_PCT`, `AUC_FIDELIZACAO_MIN_MM`, `QTD_PRODUTOS_MONOPRODUTO`) e a função `risk_factors(features) -> list[str]` (códigos canônicos).
   - `api._recommended_action` consome `risk_factors()` e só mapeia código → frase de ação; `app.py` importa as constantes para os alertas do dashboard.
   - `tests/test_serving_contract.py`: `test_risk_factors_dispara_os_codigos_certos` + grep de regressão contra `< 9.0` / `< -0.2` / `> 45` em `api.py`.

Nenhuma mudança de comportamento observável (a saída de `_predict_one` e os insights do dashboard são idênticos; 55 testes verdes, artefatos byte-idênticos).

**Nota de layout (2026-09-10):** a §2 e a §4 desta ADR colocaram `serving_contract.py`
na raiz, com o receio de `src/` arrastar `sklearn`. [ADR-0004](0004-src-layout.md) verificou
que o receio não procede (`src/__init__.py` vazio) e moveu **todo** o código para `src/` —
`serving_contract.py` incluído. A direção da dependência (`serving_contract` é a fonte,
`nodes.py` importa dela) não muda; só o caminho passa a ser `src.serving_contract`.
