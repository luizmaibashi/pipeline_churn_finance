---
tipo: decisao
status: resolvido
criado: 2026-09-10
resolvido: 2026-09-10
---

# Ticket 0005: Estratégia de execução do modelo no browser → ADR-0005

## Estado

0001–0004 fechados. 0003 confirmou: Tab 1 interativa **entra**, com sliders
livres e gauge contínuo. Decisão abaixo é a definitiva — vira ADR-0005.

## Decisão

**Arquitetura híbrida — Balde 1 + 2a:**

1. **Tabs 2, 3, 4 (carteira, performance, carteira exposta) = snapshot puro.**
   O `predict_proba` sobre os 1.200 clientes fixos roda **em build time**; a
   coluna `prob_churn` entra no CSV publicado. Zero modelo no browser para essas
   telas. Gráficos = dataviz sobre CSV.

2. **Tab 1 (Predição Individual) = Opção A, port completo em JS.**
   - `tools/export_model.py` gera `web/model.json` (94 KB gzip, 300 árvores +
     params extraídos do `.pkl`).
   - `web/infer.mjs` — ~40 linhas, `Math.fround` na comparação de threshold,
     `roundHalfEven` no FE.
   - Preserva o gauge contínuo e o `risk_factors()` ao vivo.
   - Teste de paridade obrigatório (0002), tolerância `1e-9`, no CI.

3. **Opção B (ONNX) descartada** — `.wasm` de vários MB para um modelo de 94 KB.
4. **Opção C (grade pré-computada) descartada** — 11 features, várias contínuas;
   discretização mataria o gauge sem economia real (o JSON de árvores é menor que
   uma grade decente seria).

## Saída

Vira `docs/adr/0005-execucao-do-modelo-no-browser.md`. Disparar `adr-generator`
quando 0003/0004 fecharem. `pavc-audit` se a Tab 1 entrar (é lógica de decisão
sobre pessoas replicada num segundo runtime).
