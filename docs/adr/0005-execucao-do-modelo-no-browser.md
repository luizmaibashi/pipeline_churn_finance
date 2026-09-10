# ADR-0005: Execução do modelo no navegador

**Data:** 2026-09-10
**Status:** Accepted
**Proposto por:** Luiz Maibashi
**Contexto:** deploy estático do dashboard no GitHub Pages

## 1. Contexto (o quê)

O dashboard Streamlit depende de runtime Python e sofre cold start no Streamlit
Community Cloud. A peça de portfólio precisa abrir como página estática, sem
expor o `.pkl`, preservando as quatro abas e o preditor individual com gauge
contínuo. O projeto usa dado sintético: as probabilidades demonstram o contrato
de scoring, não uma recomendação comercial nem desempenho em produção.

O `GradientBoostingClassifier` v2 tem 300 árvores e 7.922 nós. A prova em
`docs/wayfinder/deploy-pages/_prova/port_proof.py` reproduziu
`predict_proba` com diferença máxima de `2.2e-16` nos 1.200 clientes. A prova
também mostrou que a comparação de cada split precisa converter a feature para
float32.

## 2. Decisão (por quê)

Adotar arquitetura híbrida Balde 1 + 2a:

- A predição individual executa no browser por `web/infer.mjs`, sem
  dependências. `tools/export_model.py` extrai parâmetros e árvores do `.pkl`
  para `model.json`.
- As abas de carteira, performance e exposição por assessor usam snapshots CSV
  gerados no build. `prob_churn` é pré-calculada apenas para os 1.200 clientes
  publicados.
- A equivalência JS/Python é gate: pelo menos 3.000 casos e `max |delta| <=
  1e-9`, executado também dentro do pytest.

Isso entrega a interação que comunica o modelo sem pagar servidor ou runtime
WASM. Para o portfólio, remove a espera de cold start e permite carregamento de
arquivo estático; o custo é manter o exportador e a paridade a cada retreino.

## 3. Consequências

**Positivas:** página sem Python em runtime; modelo publicado pequeno (cerca de
94 KB gzip); gauge contínuo; tabs analíticas rápidas e reproduzíveis.

**Negativas:** duas implementações do forward pass; `model.json` precisa ser
regenerado junto do `.pkl`; limitações numéricas de JS exigem `Math.fround` nos
splits e round half-even no feature engineering.

## 4. Alternativas descartadas

| Opção | Motivo |
|---|---|
| ONNX com onnxruntime-web | Runtime WASM de vários MB para um modelo de 94 KB gzip. |
| Grade pré-calculada para a aba individual | Onze inputs contínuos tornariam a grade grande e eliminariam o gauge. |
| Manter Streamlit Cloud | Cold start contradiz o objetivo de uma peça de portfólio acessível. |

## 5. Validação e ROI

- **Critério executável:** `tests/paridade/parity.test.mjs` e
  `tests/test_paridade.py` mantêm `max |delta| <= 1e-9`.
- **Artefatos:** `docs/model.json` fica abaixo de 150 KB gzip, com tamanho
  verificado pelo build; `docs/infer.mjs` e `docs/data/*.csv`; nenhum `.pkl` é
  publicado.
- **Aceite de UX:** quatro abas, card de enquadramento acima delas e preditor
  respondendo client-side.
- **Risco de regressão:** regeneração muda as árvores ou parâmetros. O build
  sempre executa o exportador e o gate de paridade detecta divergência.

## 6. Referências

- `docs/wayfinder/deploy-pages/SPEC_FINAL.md`
- `docs/wayfinder/deploy-pages/0001-custo-real-port-gb-js.md`
- `docs/wayfinder/deploy-pages/0002-teste-de-paridade.md`
- `PROBLEM.md`
