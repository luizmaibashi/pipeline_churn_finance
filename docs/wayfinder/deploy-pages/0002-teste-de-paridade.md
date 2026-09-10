---
tipo: pesquisa
status: resolvido
criado: 2026-09-10
resolvido: 2026-09-10
---

# Ticket 0002: Teste de paridade JS↔Python — desenho

> Reescrito 2026-09-10: o handoff mandava usar o `stable_treasury` como
> referência, mas o Luiz vai refazer aquele projeto. Este ticket agora é
> autocontido, ancorado no que a prova de 0001 mediu.

## Bloqueio (resolvido)

Requisito inegociável: lógica portada pra JS precisa de teste de paridade contra
o Python, rodando no CI. Faltava saber **qual** divergência caçar, com que
tolerância e em que formato.

## Resultado

### Divergências JS↔Python a cobrir (por ordem de risco medido)

| Risco | Fonte | Mitigação no port | Mordeu em 0001? |
|---|---|---|---|
| **Alto** | `sklearn.tree` compara threshold em **float32**; JS é float64 | `Math.fround(x[f]) <= threshold` em cada nó | **Sim** — 9.3e-2 na pior linha |
| Médio | `round()` do Python é half-even; `Math.round`/`toFixed` é half-up | implementar `roundHalfEven(x, n)` para os `.round(4)`/`.round(2)` do FE | Não (mas fica na guarda) |
| Baixo | `np.log1p` vs `Math.log1p` | `Math.log1p` (ES2015) — idêntico em float64 | Não |
| Baixo | ordem de operações `(a/b)*(c/d)` | replicar a ordem exata do `FeatureEngineer` | Não |
| Baixo | `None`/`NaN` nas 2 features opcionais | `sem_historico_12m` / `cliente_novo_sem_contato_hist` são derivadas do input, não vêm nulas | Não |

### Formato do harness

- `tools/export_model.py` — lê o `.pkl`, cospe `web/model.json` (params + árvores,
  ~94 KB gzip). Roda no build.
- `web/infer.mjs` — a inferência portada (ES module, sem dependência).
- `tests/paridade/gerar_casos.py` — gera **N = 3.000** perfis: os 1.200 clientes
  reais + ~1.800 aleatórios com seed fixo, cobrindo faixas fora do treino e os
  dois flags opcionais. Salva `casos.json` com `{input, prob_esperada}`
  (de `predict_proba`, float64 full precision).
- `tests/paridade/parity.test.mjs` (node, roda no CI) — recarrega `casos.json` +
  `model.json`, roda `infer.mjs`, **falha se `max |Δ| > 1e-9`**.
- Gate: `pytest` chama o `node` via subprocess num teste, para o `pytest -q`
  local pegar a regressão junto com os outros 55.

### Tolerância

`1e-9` (não igualdade exata: `Math.exp`/`Math.log1p` podem diferir no último
ULP entre engines). Com o cast float32 aplicado, 0001 mediu `1.1e-15` no grid —
folga de 6 ordens de grandeza contra o limite.

### Números de referência (congelar no teste)

Os 3 perfis pré-definidos da Tab 1 (`app.py:509-536`) — Alto / Médio / Baixo
risco — viram casos nomeados no `casos.json` com a prob exata atual, para
detectar regressão de forma legível.
