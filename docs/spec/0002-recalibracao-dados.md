# Spec 0002: Conformidade do projeto com a v2

**Ancora:** [ADR-0002](../adr/0002-recalibracao-gerador-escala-wealth.md) (ver §2.5, mapa de conformidade)
**Estado:** executado — blocos 2 a 5 fechados em 2026-09-10 (ADR-0002 e ADR-0003 aceitos). Mantido como registro do contrato do diff.
**Norte:** a refatoração de 9 e 10 de setembro (ADR-0001 mais ADR-0002) é a direção. Os documentos de abril (`PROBLEM.md` v1.0, `refactoring_blueprint.md`) são legado da v1.

Este documento é o contrato do diff, organizado em blocos. Cada bloco é uma sessão. Cada mudança tem "antes", "depois" e o teste que prova.

**Ordem de dependência:** Bloco 2 primeiro (regenera o dado e o modelo). Bloco 3 depende do 2 (docs citam números novos). Bloco 4 depende do 2 (telas carregam o modelo e as features novas). Bloco 5 é independente, pode ir a qualquer momento depois do 2.

**Decisão mecânica aprovada para o Bloco 2:** renomear `saldo_bi` para `auc_milhoes` (ADR-0002 §7, decisão 1). Resolvida contra a Linguagem Ubíqua da v2 ("AuC" em todo lugar).

---

# BLOCO 2: dado recalibrado, thresholds v2, coerência do `api.py`

**Objetivo:** o gerador produz uma carteira de wealth de ~R$ 75 bi; `pct_carteira_exposta` deixa de ser binária; o `api.py` para de rodar o modelo v2 com segmentos e thresholds da v1.

## 2.1 Parâmetros (`conf/base/parameters.yml`)

```yaml
n_samples: 1200
n_advisors: 50            # era 300; agora ~24 relações por assessor (book de private banking)
test_size: 0.20
random_state: 42
learning_rate: 0.03
n_estimators: 300
max_depth: 4

# Calibração da carteira sintética (ADR-0002). Alvo: ~R$ 75 bi / ~1.200 grupos
segmentos:
  Alta Renda:    { share: 0.45, auc_mediana_milhoes: 7.0,   auc_sigma_log: 0.55 }
  Private:       { share: 0.38, auc_mediana_milhoes: 28.0,  auc_sigma_log: 0.60 }
  Wealth:        { share: 0.13, auc_mediana_milhoes: 110.0, auc_sigma_log: 0.65 }
  Family Office: { share: 0.04, auc_mediana_milhoes: 650.0, auc_sigma_log: 0.70 }

churn_base_por_segmento:
  Alta Renda: 0.10
  Private: 0.07
  Wealth: 0.05
  Family Office: 0.04

# Curva de custo para calibrar o threshold do modelo v2
threshold_v2:
  custo_fn_sobre_fp: 10       # perder um cliente dói ~10x mais que uma ligação desnecessária
  recall_minimo: 0.75        # alvo de sensibilidade para o caso de busca ativa
```

`share` soma 1,0. O AuC de cada cliente sai de `lognormal(mean=ln(mediana), sigma=sigma_log)`, em milhões de reais, piso R$ 3 M (mínimo de private banking, ANBIMA). Esperado: soma R$ 65 a 85 bi, mediana da carteira ~R$ 20 M, `Family Office` (4% dos clientes) com ~metade do AuC. A calibração fina de `auc_mediana_milhoes` e `sigma_log` é iterativa: rodar, medir a soma, ajustar. Registrar os valores finais aqui.

## 2.2 `src/data_processing/nodes.py`

### `generate_synthetic_data`

| Antes | Depois |
|---|---|
| `segmentos = ["Varejo", "Alta Renda", "Wealth", "Corporate"]`, `seg_prob = [0.65, 0.24, 0.08, 0.03]` | 4 segmentos de wealth, `share` de `parameters` |
| `saldo = np.random.lognormal(-1.8, 1.3, N)` (único, sem segmento) | por segmento: `lognormal(ln(mediana_seg), sigma_seg)`, piso R$ 3 M, coluna `auc_milhoes` |
| `meses_cli = np.random.randint(1, 144, N)` | `np.random.randint(6, 144, N)` (book de wealth não tem relação de semanas) |
| `taxa_base = {"Varejo": 0.18, ...}` | `churn_base_por_segmento` de `parameters` |
| `if saldo[i] < 0.1: mod *= 1.40` (corte fixo R$ 100 M) | `if auc[i] < tercil_inferior[seg]: mod *= 1.40` (relativo ao segmento) |

Os outros modificadores de churn (`retorno < 8`, `freq_cont == 0`, `qtd_prod == 1`, `meses_cli < 12`) ficam iguais.

### `generate_advisors_data`

`n_advisors` 300 para 50 (via `parameters`). Sem mudança de lógica. Com 50 assessores e atribuição ponderada, a chance de assessor sem cliente é ~0.

### `attach_advisor_and_behavioral_features`

| Antes | Depois |
|---|---|
| `risco_saida_assessor = df_advisors["risco_saida"].values[assessor_idx]` (binário) | propaga **também** `prob_saida_assessor = df_advisors["prob_saida_calibrada"].values[assessor_idx]` |
| `out["auc_exposto"] = (out["saldo_bi"] * risco_saida_assessor).round(4)` | `out["auc_exposto"] = (out["auc_milhoes"] * prob_saida_assessor).round(3)` |
| `out["risco_saida_assessor"] = risco_saida_assessor` | mantém a binária (dashboard: "assessor de fato saiu no período") **e** adiciona `out["prob_saida_assessor"]` |

As 3 features comportamentais e o red herring ficam iguais (são função de `churn`, não de `auc`). Revalidar a faixa de correlação mesmo assim.

### `aggregate_carteira_exposta_por_assessor`

Mesmo código, agora sobre `auc_milhoes * prob` (contínuo). `pct_carteira_exposta` vira distribuição contínua. Sem os 9 assessores vazios (efeito de `n_advisors` 300).

### `inject_data_quality_issues` e `clean_clientes_v2_bruto`

Trocar `saldo_bi` por `auc_milhoes` em: erro de escala (`* 1000`), critério de limpeza (`> p99(segmento) * 20`). Lógica conceitual sem mudança.

## 2.3 `src/model_training/nodes.py`

- `FEATURES_BASE`, `FEATURES_V2_BASE`, `FEATURES_AFTER_FE`: `"saldo_bi"` para `"auc_milhoes"`.
- `OrdinalEncoder(categories=[[...]])` em `_build_preprocessing` e `_build_preprocessing_v2`: para `[["Alta Renda", "Private", "Wealth", "Family Office"]]`.
- Nova `get_feature_importance_v2(model)`: mesma lógica de `get_feature_importance`, lista de features da v2. DataFrame ordenado.
- Nova `calibrate_thresholds_v2(model_v2, valid_df, parameters) -> pd.DataFrame`:
  - Split de validação a partir do treino (nunca toca o teste, gate ML "conjunto de teste consultado uma única vez").
  - Por segmento, varre threshold em `[0.05, 0.95]`, escolhe o ponto que minimiza `custo_fn_sobre_fp * FN(t) + FP(t)`, sujeito a `recall >= recall_minimo` quando factível.
  - Retorna `segmento, threshold, recall, precision, fn, fp, n_valid, origem_threshold`.
  - Se o segmento não tiver pelo menos 5 positivos no split de validação, usa o threshold global; `origem_threshold = "global_fallback"`. Caso contrário, `origem_threshold = "segmento"`.
  - GradientBoosting não usa `class_weight`, então não há distorção de balanceamento a recalibrar. Se um candidato com `class_weight` entrar no futuro, calibração Platt/Isotonic antes (nota no código).

## 2.4 `pipeline.py`

- Fase 1.5: `print` do AuC exposto usa `prob`; `print` de assessores sem carteira (esperado: 0).
- Fase 5: `get_feature_importance_v2(gb_final_v2)` e `catalog.save("feature_importance_v2", ...)`.
- Fase 5.5: `calibrate_thresholds_v2(...)` sobre split de validação, salva `reports/thresholds_v2.md` (tabela por segmento + fórmula de custo) e `output/data/thresholds_v2.csv`.
- Fase 6: `print` da soma de importância das 3 features comportamentais, lido do CSV.
- Prints de escala (soma de AuC, mediana por segmento, clientes por assessor).

## 2.5 `conf/base/catalog.yml`

```yaml
feature_importance_v2:
  type: pandas.CSVDataSet
  filepath: output/data/feature_importance_v2.csv
thresholds_v2:
  type: pandas.CSVDataSet
  filepath: output/data/thresholds_v2.csv
```

## 2.6 `api.py` (fechar os resíduos v1 dentro da API que já roda o modelo v2)

| Antes | Depois |
|---|---|
| `SEGMENTOS_VALIDOS = ["Varejo", "Alta Renda", "Wealth", "Corporate"]` | `["Alta Renda", "Private", "Wealth", "Family Office"]` |
| `THRESHOLD_MAP` hardcoded (0,40 / 0,50 / 0,60) | carregado de `output/data/thresholds_v2.csv` na inicialização |
| `_risk_level`: calcula `threshold` e usa 0,60 fixo | usa o threshold calibrado do segmento como corte de "ALTO"; "MEDIO" em `0.6 * threshold` |
| `auc_at_risk_MM = saldo_bi * 1000 * 0.012 * prob` (fator 1,2% sem origem) | `auc_milhoes * PCT_PERDA_CHURN * prob`, `PCT_PERDA_CHURN = 0.30` (queda de AuC que define churn na v2), comentário citando a origem |
| payload `saldo_bi` | `auc_milhoes` |

## 2.7 Testes

**Ajustar:**
- `test_sinal_comportamental_correlaciona_em_faixa_realista`: revalidar as faixas após regeneração; se deslocar, ajustar limites com margem, documentar o valor medido.
- `test_auc_exposto_e_produto_saldo_por_risco`: `auc_milhoes * prob_saida_assessor`.
- `test_model_performance_thresholds`: limites `f1_macro >= 0.55`, `roc_auc >= 0.70` continuam. Se a regeneração ficar abaixo, a calibração ficou ruidosa demais, ajustar, não relaxar o teste.
- Perfis de exemplo (`test_model.py`, `test_features.py`, `test_api.py`): `"saldo_bi": 0.5` para `"auc_milhoes": 120.0`; segmentos novos.

**Novos (`tests/test_leakage.py`, gate do ADR-0002 §6):**

```python
def test_auc_agregado_bate_com_narrativa(dataset_v2):
    _, _, df_v2 = dataset_v2
    total_bi = df_v2["auc_milhoes"].sum() / 1000
    assert 65 <= total_bi <= 85

def test_pct_carteira_exposta_nao_degenerada(dataset_v2):
    _, df_adv, df_v2 = dataset_v2
    col = aggregate_carteira_exposta_por_assessor(df_v2, df_adv)["pct_carteira_exposta"]
    assert col.nunique() >= 10
    assert col.std() >= 0.02

def test_ordenacao_auc_por_segmento(dataset_v2):
    _, _, df_v2 = dataset_v2
    med = df_v2.groupby("segmento")["auc_milhoes"].median()
    assert med["Alta Renda"] < med["Private"] < med["Wealth"] < med["Family Office"]

def test_clientes_por_assessor_plausivel(dataset_v2):
    _, _, df_v2 = dataset_v2
    media = df_v2.groupby("assessor_id").size().mean()
    assert 12 <= media <= 40
```

**Novo (`tests/test_model.py`):**

```python
def test_thresholds_v2_registrados_e_por_segmento():
    assert os.path.exists("reports/thresholds_v2.md")
    txt = open("reports/thresholds_v2.md", encoding="utf-8").read()
    for seg in ["Alta Renda", "Private", "Wealth", "Family Office"]:
        assert seg in txt
```

## 2.8 Regeneração (ordem)

```
python pipeline.py            # CSVs + feature_importance_v2 + thresholds_v2 + 2 pkls
python shap_analysis.py       # SHAP v1
python shap_analysis_v2.py    # SHAP v2
python -m pytest -q           # 37 + 5 novos = 42, verdes
```

Conferir no output: soma de AuC 65 a 85 bi, mediana crescente por segmento, AuC exposto agregado 10 a 15%, clientes por assessor 12 a 40.

## 2.9 Aceitação do Bloco 2

- Os 5 testes novos passam, os 37 antigos continuam passando.
- `test_recall_early_warning_vs_baseline_reativo` continua verde (v2 supera v1). A recalibração não pode inverter a Direção A.
- `output/data/feature_importance_v2.csv` e `reports/thresholds_v2.md` existem e são citáveis.
- `api.py` sobe sem os segmentos `Varejo`/`Corporate` e carrega os thresholds do CSV.

---

# BLOCO 3: docs alinhados à v2

**Depende do Bloco 2** (números novos). **Objetivo:** os documentos param de contar a história da v1.

| Arquivo | Mudança |
|---|---|
| `PROBLEM.md` | **Reescrever como v2.0.** Contexto: gestora de wealth ~R$ 75 bi / ~1.200 grupos. Target: early-warning comportamental (queda de AuC continua como componente, não como único sinal). Segmentos `Alta Renda / Private / Wealth / Family Office` com faixas de AuC de wealth. Thresholds: calibrados por segmento (`reports/thresholds_v2.md`), não fixos no contrato. §8 escopo: PF de private banking, mín. R$ 3 M, sem contas < 6 meses. Direção B (advisor attrition) entra como produto de dado. A v1.0 vira histórico no topo do arquivo ("substituída em 2026-09-10, ver ADR-0001 e ADR-0002"). |
| `README.md` | Narrativa da carteira de wealth. Tabela de métricas dos CSVs regenerados. "19 testes" para 42. Thresholds por segmento de `thresholds_v2.md`. **Remover ou reescrever a seção "Métricas de Negócio & ROI Estimado"**: hoje afirma "+15% retenção", "R$ 500 Milhões", ROI, o que contradiz o ADR-0001 §5 ("projeto fictício sem métrica de negócio real"). Substituir por "o que este projeto demonstra" (método, não ROI inventado). Corrigir "não por artefato de simulação" para uma frase honesta: o dado é sintético e calibrado ao mercado, a demonstração é de método e engenharia, não de um achado transferível. `python shap_analysis.py` para `shap_analysis_v2.py`. |
| `AGENTS.md` | Linguagem Ubíqua (segmentos de wealth). "Estado do projeto" refletindo blocos 2 a 5. Contagem de testes 42. |
| `reports/eda_clientes_v2_bruto.md` | Regenerar todos os números sobre o dataset novo. |
| `docs/adr/0001-*.md` | §7 curto apontando para o ADR-0002. |
| `brain/sessions/frentes/pipeline_churn_finance.md` + `INDEX.md` | Fechar o ciclo: auditoria feita, blocos 2 a 5 executados até onde chegou. |

**Caos Funcional (junto do Bloco 3):**
- `refactoring_blueprint.md` e `refatoracao/` para `docs/historia_v1/`, com um `README.md` de 2 linhas dizendo que é o plano da refatoração de engenharia da v1, concluído, superado pela ADR-0001.
- Remover `docs/roadmap_data_science_crispdm.md` (template de "NPS Predictor", não é deste projeto).

## Aceitação do Bloco 3

- `grep -ri "R\$ 75 bi\|487\|varejo\|corporate" README.md PROBLEM.md AGENTS.md` não retorna nada incoerente com a v2.
- Nenhum número no README sem CSV de origem.
- `PROBLEM.md` v2.0 e ADR-0001/0002 contam a mesma história.

---

# BLOCO 4: migração da apresentação para o modelo v2

**Depende do Bloco 2** (modelo e features). **Objetivo:** as telas param de rodar o modelo v1.

| Arquivo | Mudança |
|---|---|
| `app.py` aba 1 (predição individual) | `load_artifacts()` carrega `gb_pipeline_v2.pkl`. Formulário ganha os 3 inputs comportamentais (`dias_desde_ultimo_contato`, `variacao_freq_contato_3m`, `tempo_resposta_medio_horas`), opcionais (nulo estrutural). Segmentos de wealth. Remover "Segmento Varejo, maior taxa histórica (25%+)". `saldo * 0.012` para a fórmula do `api.py` (`auc_milhoes * 0.30 * prob`). Casos de exemplo na escala de wealth. |
| `app.py` aba 2 (carteira) | Scoring ao vivo com `base_clientes_v2_limpo.csv` e as features v2. |
| `app.py` aba 3 (performance) | Ler os artefatos v2 (`feature_importance_v2.csv`, `cv_scores_v2.csv`, `comparacao_v1_v2.csv`). O pipeline precisa emitir uma matriz de confusão v2 se a aba mostrar uma. |
| `app.py` header | "~R$75bi sob custódia" confirmado pelo dado (Bloco 2 fez a soma bater). |
| `monitor.py` | `NUMERIC_FEATURES` e `FEATURES` para o conjunto v2 (com as 3 comportamentais). Carregar `gb_pipeline_v2.pkl`. O baseline de referência é um sample do próprio dado v2 limpo. |
| `agent.py` | Segmentos de wealth nos schemas de ferramenta (`:331`, `:349`, `:370`, `:407`). Ler as features v2. **Remover a string "ROC-AUC 0.93"** e substituir pelo número real do `comparacao_v1_v2.csv`. Schema da ferramenta de simulação com as features v2. |
| `agent_chat.py` | Perguntas de exemplo reescritas para o vocabulário v2 (early-warning, não "cliente Varejo com retorno de 6%"). |
| `orchestrator.py` | Confirmar que aponta para o `monitor.py` migrado. |
| `shap_analysis.py` (v1) | Aposentar quando a aba 3 e o endpoint de SHAP do `api.py` (`:593`) apontarem para o v2. Remover o arquivo e a referência no README. |

## Aceitação do Bloco 4

- `grep -rn "gb_pipeline.pkl\|\"saldo_bi\"\|Varejo\|Corporate" app.py monitor.py agent.py agent_chat.py` só retorna referências históricas explícitas (comparação v1, se houver).
- Abrir o `app.py` e rodar uma predição na aba 1 usando um input comportamental muda o score.
- `agent.py` não cita nenhum número de performance que não esteja num CSV.

---

# BLOCO 5: notebooks e a stack de infraestrutura

**Independente** (só depende do Bloco 2 para os números). **Objetivo:** o que sobrou do showcase de engenharia da v1 ou vira tese secundária explícita, ou sai.

## 5.1 Notebooks

- `notebooks/01_Pipeline_Pandas_ScikitLearn.ipynb`: reenquadrar de "grande plataforma financeira brasileira" (varejo de massa) para a gestora de wealth. Conteúdo passa a mostrar o pipeline v2. Ou aposentar se o `pipeline.py` mais o README já cobrem o CRISP-DM.
- `notebooks/02_Evolucao_BigData_PySpark.ipynb`: **decisão do Luiz.** Um book de wealth de 1.200 clientes não é um problema de Big Data, esse notebook contradiz a direção. Opções: (a) aposentar; (b) manter como apêndice explícito ("o mesmo pipeline em Spark, para o caso de a carteira crescer 100x"), com o `execution_count` consertado (Restart e Run All).

## 5.2 Stack de infra (Docker, Envoy, fila Redis, worker async)

**Decisão do Luiz (ADR-0002 §7.1).** É do blueprint da v1 e é desproporcional a um modelo de 1.200 linhas. Opções:
- (a) Manter e assumir "showcase de engenharia" como tese secundária: uma seção no README dizendo que a stack async é deliberadamente sobre-construída para demonstrar os padrões (Envoy sidecar, fila, worker), não porque o volume exige.
- (b) Simplificar `api.py` para síncrono, mover a stack para `docs/arquitetura_escala.md` como "como isto escalaria", manter o `docker-compose.yml` mínimo.

## Aceitação do Bloco 5

- Nenhum artefato do repo aponta para uma direção diferente da v2 sem uma frase explícita dizendo por quê.
- `notebooks/02`, se ficar, tem `execution_count` crescente e uma célula de abertura explicando que é apêndice.
