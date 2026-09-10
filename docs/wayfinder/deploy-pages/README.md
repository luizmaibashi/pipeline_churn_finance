# Wayfinder: deploy do pipeline_churn_finance como GitHub Pages estático

**Objetivo:** tirar o projeto do Streamlit (cold start, "Zzz") e colocá-lo no ar
como página estática — zero Python em runtime — seguindo o padrão já usado em
`payflow_inadimplencia` V3, `stable_treasury` (ADR-0016) e `tech_challenge_fase3`.
Balde 2 da política de deploy da base (compute reproduzível no browser).

**Fonte da política:** `../../DECISAO_DEPLOY_PORTFOLIO.md` (base) +
`metodologia/AI_ENGINEERING/13_deploy_portfolio.md` (base).

> **Nota (2026-09-10):** o `stable_treasury` **não** é mais referência — o Luiz
> vai refazê-lo. O padrão port+paridade deste projeto é autocontido (ver 0002).

## TL;DR da análise (0001 + 0002 resolvidos)

- **Port do modelo pra JS é barato.** Provado empiricamente: reimplementação do
  forward pass inteiro bate `predict_proba` com `max |Δ| = 2e-16` nos 1.200
  clientes reais. ~40 linhas de JS, 94 KB gzip, sem `.wasm`, sem `.pkl` público.
- **O gotcha real é float32, não round half-even.** `sklearn.tree` compara
  thresholds em float32; sem `Math.fround` no port, ~0,1% dos inputs pulam pro
  leaf errado (Δ de 0,09 na prob). O teste de paridade é o que pega isso — e
  pegou, nesta sessão.
- **`fe_params.json` está defasado pra v2** — extrair params do `.pkl`.
- **Só a Tab 1 precisa de modelo vivo.** Tabs 2–4 = snapshot; `prob_churn` dos
  1.200 clientes vira coluna do CSV em build time.
- **Recomendação:** híbrido Balde 1 + 2a. Snapshot para 3 abas, port JS + gauge
  contínuo para a Predição Individual. Decisão final depende de 0003 (a Tab 1
  interativa vale o port?) e 0004 (tom do enquadramento).

## Mapa da névoa

O que está nebuloso e por quê não dá pra escrever a spec direto:

1. **Custo real do port do modelo.** O handoff assume que os transformadores
   customizados (`FeatureEngineer`, `StructuralNullImputer`) tornam o port "alto
   custo". Mas o classificador é um `GradientBoostingClassifier` sklearn baunilha
   (300 árvores, `max_depth=4`, ~7.922 nós no total) e os transformadores são
   ~12 linhas de aritmética trivial + um `OrdinalEncoder` de 4 categorias. A
   premissa de custo precisa ser medida antes de escolher a estratégia.
2. **Padrão de paridade.** O `stable_treasury` já resolveu "lógica portada pra JS
   precisa de teste de paridade contra o Python" e tropeçou no `round()`
   half-even. Esse checklist precisa ser extraído antes de portar qualquer coisa.
3. **Escopo da página.** As 4 abas do `app.py` têm necessidades diferentes: só a
   Predição Individual precisa do modelo vivo; as outras 3 são snapshot de CSV.
   Decidir o subconjunto e a identidade visual (o dark theme atual é o clichê de
   IA — memória diz pra rodar `ui-ux-pro-max` por domínio, não repetir paleta).
4. **Enquadramento honesto no deploy.** Recall 7,14% (n=240, 28 churns), IC95%
   da diferença [0,00; 17,87] p.p. inclui zero, dado sintético. Como isso aparece
   na página sem virar letra miúda que ninguém lê — decisão de tom do Luiz.
5. **A decisão A/B/C.** Estratégia de execução do modelo no browser. Depende de
   1 e 2. Vira ADR-0005 do projeto.
6. **Débitos que viajam junto.** 4 vulnerabilidades Dependabot (1 alta) no repo
   público; revalidar `requirements.txt` (já pinado com `==`) nesta máquina.

## Tickets

| # | Tipo | Título | Status |
|---|---|---|---|
| 0001 | pesquisa | Custo real do port do GB v2 para JS | ✅ resolvido |
| 0002 | pesquisa | Teste de paridade JS↔Python — desenho | ✅ resolvido |
| 0003 | grilling | Escopo da página: abas, profundidade, identidade visual | ✅ resolvido |
| 0004 | grilling | Como o enquadramento honesto sobrevive ao deploy | ✅ resolvido |
| 0005 | decisao | Estratégia de execução do modelo no browser → ADR-0005 | ✅ resolvido |
| 0006 | tarefa-simples | Débitos: 4 Dependabot + revalidar requirements.txt | aberto (paralelo) |

**`SPEC_FINAL.md` compilada.** Próximo: `adr-generator` → `docs/adr/0005` →
`spec-governance` para a implementação. 0006 fecha antes do merge.
