---
tipo: pesquisa
status: resolvido
criado: 2026-09-10
resolvido: 2026-09-10
---

# Ticket 0001: Custo real do port do GB v2 para JS

## Bloqueio

O handoff classifica o "port completo pra JS" como **alto custo** porque
`skl2onnx` não converte os transformadores customizados. Mas isso confunde dois
problemas:

- **Conversão ONNX dos customizados** — real, mas só importa se a estratégia for
  ONNX (Opção B).
- **Reimplementar a matemática em JS** — que é o que Opção A pede, e a matemática
  aqui é trivial.

Inventário do que precisa ser portado (medido no `.pkl` em 2026-09-10):

| Passo do Pipeline | O que faz | Params a extrair |
|---|---|---|
| `null_imputer` (`StructuralNullImputer`) | `fillna(mediana)` em colunas de nulo estrutural | dict `medianas_` (≤6 floats) |
| `fe` (`FeatureEngineer`) | 4 features derivadas: `engajamento_score`, `retorno_relativo`, `flag_risco`, `intensidade_rel` — aritmética + `np.log1p` + `.round(4)/.round(2)` | `freq_max_`, `qtd_max_`, `media_retorno_` (3 floats) |
| `prep` (`ColumnTransformer` + `OrdinalEncoder`) | codifica `segmento` em 0–3, passa o resto | 1 lista de 4 categorias (`fe_params.json` já tem) |
| `clf` (`GradientBoostingClassifier`) | 300 árvores, `max_depth=4`, `lr=0.03`, `init=DummyClassifier` (prior log-odds), ~7.922 nós totais | árvores + prior inicial |

## Perguntas a responder

1. **Extração das árvores.** `m2cgen` (gera JS puro de um GBM sklearn) cobre o
   `GradientBoostingClassifier` direto? Se sim, é o caminho — resta só o
   preprocessing manual. Se não, escrever o dump de `estimators_` para JSON
   (`children_left/right`, `feature`, `threshold`, `value`) + o loop de inferência
   (soma dos `lr * leaf_value` + prior, sigmoid). ~30 linhas de JS.
2. **Prior inicial.** Como o `init_` (`DummyClassifier`) entra no raw score?
   Para binário é `log(p1/p0)` da base de treino — confirmar extraindo
   `_raw_predict_init` ou `gb.init_.class_prior_`.
3. **`np.log1p` e ordem de operações.** `Math.log1p` existe em JS (ES2015). A
   ordem de `(a/b)*(c/d)` e o `.round(n)` (half-even!) precisam bater — ver 0002.
4. **Tamanho do bundle.** Estimar o JSON das 300 árvores minificado/gzipado.
   Referência: 7.922 nós × ~4 ints ≈ 30–60 KB gzip. Comparar contra o `.wasm` do
   `onnxruntime-web` (multi-MB) — a razão custo/benefício da Opção B.

## Resultado

**Port completo em JS (Opção A) é BAIXO/MÉDIO custo. Provado empiricamente em
2026-09-10** com `_prova/port_proof.py` (reimplementação do forward pass
inteiro em Python "estilo JS", sem sklearn no caminho).

### Paridade medida

| Conjunto | max \|Δ\| vs `predict_proba` |
|---|---|
| 1.200 clientes reais (`base_clientes_v2_limpo.csv`) | **2.2e-16** (epsilon de máquina) |
| grid 2.000 perfis aleatórios com nulos, **sem** cast float32 | 9.3e-02 ⚠️ |
| mesmo grid, **com** cast float32 na comparação de threshold | **1.1e-15** |

### O único gotcha real: float32, não round half-even

`sklearn.tree` faz predição em **float32** (`DTYPE = np.float32`): `X` é convertido
antes de descer a árvore e os thresholds foram aprendidos nessa precisão. Em JS,
a comparação `x[feature] <= threshold` **tem que** ser
`Math.fround(x[feature]) <= threshold`. Sem isso, ~0,1% dos inputs caem do lado
errado de um split perto da fronteira e recebem uma probabilidade grosseiramente
errada (0,09 de diferença absoluta na pior linha do grid — de "baixo risco" para
"alto risco"). O `round()` half-even do FE **não** foi fonte de divergência aqui
(as 3 features derivadas bateram exato), mas o teste de paridade continua
obrigatório — foi ele que pegou o float32.

### Params a extrair (do `.pkl`, NÃO do `fe_params.json`)

`output/models/fe_params.json` está **defasado para a v2**: tem
`media_retorno = 11.616`, o Pipeline real usa `11.514`. Extrair tudo do `.pkl`:

| Origem | Valor |
|---|---|
| `null_imputer.medianas_` | `{retorno_12m_pct: 11.61, dias_desde_ultimo_contato: 14.1, tempo_resposta_medio_horas: 14.0}` |
| `fe.freq_max_ / qtd_max_ / media_retorno_` | `10.0 / 8 / 11.513933…` |
| `prep` → `OrdinalEncoder.categories_[0]` | `['Alta Renda','Private','Wealth','Family Office']` (índice 0–3) |
| ordem do vetor que o `clf` vê | `prep.get_feature_names_out()` — `ordinals__segmento` primeiro, depois 14 `pass__` |
| `clf._raw_predict_init(zeros)` | prior log-odds inicial (escalar) |
| `clf.estimators_.ravel()` | 300 árvores → `children_left/right`, `feature`, `threshold`, `value` |

### Bundle

300 árvores / 7.922 nós + params → JSON **312 KB**, **94 KB gzip**. Sem `.wasm`,
sem `.pkl` público. Opção B (ONNX via `onnxruntime-web`) arrastaria um runtime
`.wasm` de vários MB para rodar isso — **descartada**.

### Inferência em JS (esqueleto, ~40 linhas)

```
fround = Math.fround
imputa nulos estruturais (3 colunas) -> mediana
FE: engajamento_score, retorno_relativo, flag_risco, intensidade_rel
    (usa Math.log1p; arredonda com toEven — ver 0002, mas não mordeu aqui)
encoder: índice de segmento em ['Alta Renda','Private','Wealth','Family Office']
monta x[15] na ordem de get_feature_names_out()
raw = init; para cada árvore: desce com fround(x[f]) <= th; raw += lr * leaf
return 1 / (1 + exp(-raw))
```

### Consequência para 0003/0005

**Só a Tab 1 (Predição Individual) precisa do modelo vivo.** As Tabs 2–4 fazem
`predict_proba` sobre os mesmos 1.200 clientes fixos — isso vira coluna
pré-computada no CSV em build time. O port do modelo é **opcional**, decidido
puramente por "queremos o preditor interativo?" (0003).
