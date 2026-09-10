# EDA — `base_clientes_v2_bruto.csv` (+ `base_assessores_bruto.csv`)

**Data:** 2026-09-09
**Dataset:** saída de `inject_data_quality_issues()` (Etapa 1, `src/data_processing/nodes.py`) — dataset sintético limpo (`base_clientes_v2`) com 6 tipos de sujeira injetados, cada um ancorado numa causa de negócio (ver docstring da função e commit `b80bd67`).
**Objetivo desta EDA:** decidir tratamento por causa, não por reflexo (gate CRISP-DM da base de conhecimento) — nenhuma feature nova deve ser construída sobre o bruto antes deste documento existir.

---

## 1. Duplicatas

- `cliente_id` duplicado: **18** (1,5% da base, N=1218 vs. 1200 originais).
- Linha inteira duplicada: **0** — confirma que a duplicata é do tipo difícil (mesmo cliente, campo levemente divergente), não cópia trivial.
- Amostra:

| cliente_id | saldo_bi | churn |
|---|---|---|
| CLI00102 | 0,1261 | 0 |
| CLI00102 | 0,1286 | 0 |

**Causa de negócio:** onboarding duplo (dois assessores cadastraram o mesmo cliente) ou migração de sistema legado pós-fusão — `saldo_bi` diverge porque a duplicata capturou o saldo em datas de snapshot diferentes.

**Decisão:** deduplicar por `cliente_id`, mantendo a linha com `saldo_bi` mais recente (não existe timestamp real no dataset sintético — na Etapa 3, ao integrar ao pipeline de treino, manter a primeira ocorrência por convenção e documentar a limitação). **Nunca treinar com as duas linhas** — infla `n` artificialmente e pesa o cliente 2x no gradiente.

---

## 2. Colunas constantes / quase constantes

Nenhuma coluna com `nunique() <= 1`. Sem achado.

---

## 3 / 4. Sentinela mascarada e outlier implausível

**Sentinela `-999` em `retorno_12m_pct`:** 68 ocorrências, todas em clientes com `meses_cliente < 12` (104 elegíveis, 65% receberam sentinela — bate com a taxa configurada de 60% ± ruído amostral).

**Causa de negócio:** sistema legado não usa `NULL` — grava `-999` quando o cliente não completou 12 meses de relacionamento e a métrica não pode ser calculada.

**Decisão:** tratar como nulo estrutural, nunca como valor numérico real. Se `-999` entrar num modelo sem tratamento, vira o valor mais extremo da distribuição e distorce qualquer split baseado em árvore — pior que perder a linha.

**Outlier em `saldo_bi`:** 3 casos, todos em `segmento=Varejo`, valores 442,5 / 536,6 / 128,9 (p99 real da coluna é 4,00 — ou seja, são ~100x o teto plausível do próprio segmento).

**Causa de negócio:** erro de casa decimal na digitação manual do assessor (ex.: 0,4425 registrado como 442,5).

**Decisão:** critério relacional, não corte arbitrário — sinalizar `saldo_bi > p99(segmento) * 20` como suspeito de erro de escala (não "outlier alto" genérico, que apagaria clientes Wealth/Corporate legítimos de saldo alto). Correção proposta: dividir por 1000 e revalidar contra a faixa do segmento, não descartar a linha — o cliente existe, só o dado está errado.

---

## 5. Perfil de nulos por coluna

| Coluna | Nulos | % |
|---|---|---|
| `freq_contato_mes` | 176 | 14,4% |
| `dias_desde_ultimo_contato` | 7 | 0,6% |
| `tempo_resposta_medio_horas` | 7 | 0,6% |

Os 7 nulos de `dias_desde_ultimo_contato`/`tempo_resposta_medio_horas` batem exatamente com os 7 clientes com `meses_cliente < 2` — nulo estrutural por design (cliente novo não tem histórico suficiente para a métrica existir). **Não é dado perdido — é dado que ainda não existe.** Decisão: manter nulo, nunca imputar; modelo precisa de estratégia própria para cliente novo (ex.: flag `cliente_novo` + imputação por 0 ou por não incluir a feature nesse caso), não imputação estatística que inventa um comportamento que não ocorreu.

---

## 6. Redundância entre colunas

`dias_desde_ultimo_contato`, `variacao_freq_contato_3m`, `tempo_resposta_medio_horas` — correlação cruzada entre si < 0,07 (medido no dataset limpo, `tests/test_leakage.py`). Não são a mesma coisa disfarçada 3x.

---

## 7. Relação de cada bloco com o alvo

Já coberto pelo checkpoint anti-artefato-de-simulação em `tests/test_leakage.py` (correlação 0,19–0,24 contra `churn`, faixa realista). Sem achado novo aqui.

---

## 8. Códigos de ausência mascarados

Ver item 3/4 — `-999` é o único sentinela do dataset. Nenhum outro código disfarçado (`XNA`, `Unknown`, `9999`) foi injetado nesta rodada.

---

## 9. A nulidade de cada coluna prediz o alvo?

**Teste feito para `freq_contato_mes`** (a única coluna com nulo não-estrutural, portanto a única onde esse teste faz sentido — as outras duas são MNAR por design conhecido, não precisam de teste estatístico para explicar a causa):

| | n | Taxa de churn |
|---|---|---|
| Nulo | 176 | 18,8% |
| Preenchido | 1042 | 20,2% |

Qui-quadrado de independência: **p = 0,72** — sem significância. A nulidade **não** prediz o alvo.

**Causa da nulidade confirmada:** correlação de 0,256 entre `freq_contato_mes` nulo e `anos_de_casa` do assessor vinculado — assessor mais sênior/menos digital registra menos contato no CRM, exatamente a causa de negócio desenhada. É viés real (não é MCAR — Missing Completely At Random), mas é viés **operacional**, não um proxy disfarçado do churn.

**Por que isso importa:** é o oposto do achado do `tech-challenge-fase3-alfabetizacao` (gate #9 da base) — lá a nulidade *era* o vazamento (nulo ⇒ 100% de uma classe). Aqui a nulidade tem causa real e documentada, mas está estatisticamente desconectada do alvo. **Decisão seguinte disso:** imputação é segura aqui — não corre o risco de "aprender" o alvo através do padrão de ausência. Estratégia recomendada: imputar por segmento+canal do assessor (não pela média global, que ignoraria o viés operacional), OU manter como categoria própria (`contato_nao_registrado`) para preservar a informação de que o dado é de baixa confiança sem inventar um valor.

---

## Resumo de decisões (para a Etapa 3 — pipeline de treino sobre o bruto)

| # | Sujeira | Tratamento decidido |
|---|---|---|
| 1 | Duplicata de cliente | Deduplicar por `cliente_id`, manter 1ª ocorrência (documentar limitação de não ter timestamp real) |
| 2 | Nulo `freq_contato_mes` | Imputar por segmento+canal do assessor, ou categoria `contato_nao_registrado` — seguro, nulidade não prediz alvo (p=0,72) |
| 3 | Nulo estrutural (`dias_desde_ultimo_contato`/`tempo_resposta`) | Manter nulo + flag `cliente_novo`; nunca imputar valor inventado |
| 4 | Outlier de escala (`saldo_bi`) | Critério relacional (`> p99(segmento)*20`), corrigir dividindo por 1000, revalidar — não descartar linha |
| 5 | Sentinela `-999` (`retorno_12m_pct`) | Tratar como nulo estrutural, nunca como valor numérico |
| 6 | `canal` sujo | Padronizar string (`R.I.A.`/`ria` → `RIA`) — confirmado que são a mesma categoria, não 3 grupos reais |
