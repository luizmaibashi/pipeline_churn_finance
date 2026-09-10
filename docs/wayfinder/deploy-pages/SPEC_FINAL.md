# SPEC FINAL — Deploy do pipeline_churn_finance em GitHub Pages estático

Compilada de 0001–0005 (0006 corre em paralelo como checklist pré-merge).
Formato `/grill-with-docs`. Próximo: `adr-generator` → `docs/adr/0005-*.md`;
depois `spec-governance` para a implementação.

## 1. O Quê

Publicar o dashboard do projeto como página estática no GitHub Pages — zero
Python em runtime — preservando as 4 abas e o preditor individual interativo.
Arquitetura **híbrida Balde 1 + 2a**:

| Aba | Runtime | Fonte de dado |
|---|---|---|
| 🎯 Predição Individual | **modelo portado em JS** (`infer.mjs`) | input do usuário |
| 📈 Análise da Carteira | snapshot | `base_clientes_v2_limpo.csv` + coluna `prob_churn` pré-computada |
| 🔬 Performance do Modelo | snapshot | `comparacao_v1_v2.csv`, `feature_importance_v2.csv`, `cv_scores_v2.csv`, `thresholds_v2.csv`, `confusion_matrix.csv` |
| 🧭 Carteira Exposta por Assessor | snapshot | `carteira_exposta_por_assessor.csv` |
| SHAP (camada da Tab 1) | snapshot | `client_explanations.csv` (40 KB) |

## 2. Por Quê

- Streamlit Community Cloud dorme após 12h → devolve shell "Zzz" de ~4 KB.
  Portfólio precisa carregar em 200 ms, não em cold start de 40 s.
- O modelo é um `GradientBoostingClassifier` baunilha (300 árvores, `max_depth=4`,
  7.922 nós). Port pra JS mede `max |Δ| = 2e-16` contra `predict_proba` nos 1.200
  clientes reais (provado, `_prova/port_proof.py`, 2026-09-10). 94 KB gzip.
- ONNX (`onnxruntime-web`) traria `.wasm` de vários MB para um modelo de 94 KB —
  pior razão. Grade pré-computada mataria o gauge contínuo. Port direto vence.

## 3. Como

### 3.1 Extração do modelo — `tools/export_model.py`
- Lê `output/models/gb_pipeline_v2.pkl`. **Não usar `fe_params.json`** (defasado
  pra v2: `media_retorno` 11,616 vs 11,514 real).
- Emite `web/model.json`: `imputer_medians` (3), `fe` (`freq_max` 10, `qtd_max` 8,
  `media_retorno` 11,5139…), `encoder.categories` (`['Alta Renda','Private','Wealth','Family Office']`),
  `feature_order` (de `prep.get_feature_names_out()`), `gb.init_raw`
  (de `clf._raw_predict_init`), `gb.lr` (0,03), `trees[300]`
  (`children_left/right`, `feature`, `threshold`, `value`).
- Roda no build (documentado no README) — `model.json` é artefato versionado.

### 3.2 Inferência — `web/infer.mjs` (ES module, sem dependência)
1. Imputa nulo estrutural nas 3 colunas → mediana.
2. FE: `engajamento_score`, `retorno_relativo`, `flag_risco`, `intensidade_rel` —
   `Math.log1p`, `roundHalfEven(x, n)` nos `.round(4)`/`.round(2)`.
3. Encoder: índice do segmento (0–3).
4. Monta `x[15]` na `feature_order`.
5. GB: `raw = init_raw`; por árvore, desce com **`Math.fround(x[f]) <= threshold`**
   (obrigatório — sklearn compara em float32), `raw += lr * leaf_value`.
6. `return 1 / (1 + Math.exp(-raw))`.

### 3.3 Teste de paridade — `tests/paridade/` (gate de CI)
- `gerar_casos.py` → `casos.json`: 1.200 clientes reais + ~1.800 aleatórios
  (seed fixo, faixas fora do treino, 2 flags opcionais) + os 3 perfis nomeados
  da Tab 1, cada um com `prob_esperada` de `predict_proba` (float64).
- `parity.test.mjs` (node) recarrega, roda `infer.mjs`, **falha se `max |Δ| > 1e-9`**.
- `pytest` invoca via subprocess → entra na suíte dos 55 testes.

### 3.4 Build da página — `tools/build_site.py`
- Gera `docs/index.html` + `docs/.nojekyll`.
- Copia os CSVs snapshot para `docs/data/`; adiciona `prob_churn` e `risco` a
  `base_clientes_v2_limpo.csv` (pré-computados uma vez).
- `docs/model.json`, `docs/infer.mjs`.
- Segundo alvo de saída registrado no `pipeline.py` ou script próprio (padrão
  `tech_challenge_fase3`: cópia byte a byte de um HTML gerado).

### 3.5 Identidade visual
- `ui-ux-pro-max --design-system` calibrado para wealth / early-warning
  comportamental. **Não** reusar o dark `#0e1117` + neon do `app.py` (tell de IA)
  nem cair na paleta teal-green dos outros projetos por default.
- Sequência: `artifact-design → dataviz → ui-ux-pro-max → escrita-organica`.
- Dataviz: portar os gráficos Plotly para uma lib que roda estática (Plotly.js
  standalone, ou Observable Plot / Chart.js — decidir na sessão de UI).

### 3.6 Enquadramento honesto
- **Card de destaque antes das abas** (não fixo no scroll): dado sintético,
  gestora fictícia, +7,1 p.p. de recall mas IC95% da diferença inclui zero →
  não é ganho robusto, nenhum número é efeito de negócio. Reusar copy de
  `app.py:814-825` + `README.md §Avaliação`.
- "AuC em risco" (`AuC × 30% × prob`) fica, rotulado como ilustração da mecânica.
- Card + copy nova → `escrita-organica` antes de publicar.

### 3.7 README público
- Bloco "No ar" com a URL, nota no "como reproduzir", árvore de estrutura.
- `docs/adr/0005-execucao-do-modelo-no-browser.md` (via `adr-generator`).
- `docs/spec/deploy-pages.md` (handoff, padrão da frente `portfolio_deploy`).

## 4. Critérios de aceite

1. `tests/paridade/parity.test.mjs` verde com `max |Δ| ≤ 1e-9` sobre ≥ 3.000 casos.
2. `pytest -q` continua verde (56+ testes — o de paridade incluído).
3. `docs/index.html` abre sem console error em servidor HTTP estático ou GitHub
   Pages; as 4 abas renderizam; o preditor responde a mudança de slider
   client-side. `file://` não é ambiente válido para `fetch()` de snapshots.
4. Nenhum `.pkl` em `docs/`. `model.json` ≤ 150 KB gzip (payload transferido);
   o JSON descompactado pode chegar a ~320 KB pela serialização das árvores.
5. Lighthouse: sem cold start, first contentful paint < 1,5 s.
6. Card de enquadramento visível acima da primeira aba.
7. 0006 fechado: alertas Dependabot triados, `requirements.txt` revalidado.

## 5. Fora de escopo

FastAPI, agente de dados, monitor de drift, Redis/worker — ficam como
arquitetura de referência no repo, documentados nos ADRs. `shap_values.csv`
(372 KB) não vai pro bundle.

## 6. Riscos

| Risco | Mitigação |
|---|---|
| Plotly.js standalone é pesado (~3 MB) | avaliar Observable Plot / Chart.js na sessão de UI; import dinâmico se ficar em Plotly |
| Regeneração futura do `.pkl` muda as árvores e o `model.json` fica órfão | `export_model.py` roda no mesmo pipeline; teste de paridade pega divergência |
| `roundHalfEven` mal implementado | os 3 perfis nomeados no `casos.json` pegam regressão de forma legível |
| Re-treino com `scikit-learn` != 1.8.0 muda serialização | pin `==` já existe; 0006 revalida |
