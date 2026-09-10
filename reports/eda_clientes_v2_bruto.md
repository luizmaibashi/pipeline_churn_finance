# EDA: clientes v2 bruto

**Fonte:** execução determinística de `python pipeline.py`, seed 42, 2026-09-10.
**Escopo:** inspeção do dataset bruto antes da limpeza. A população é sintética.

## Foto da carteira limpa de referência

A base v2 contém 1.200 relações, R$ 76,05 bilhões de AuC e 140 eventos de churn (11,7%). A mediana global de AuC é R$ 16,84 milhões. A distribuição segue a ordem esperada de mediana: Alta Renda (R$ 7,19 M), Private (R$ 28,26 M), Wealth (R$ 108,59 M) e Family Office (R$ 544,63 M).

## Sujeiras injetadas

| Tipo | Evidência na execução | Causa simulada | Tratamento |
|---|---:|---|---|
| Duplicata | 18 linhas, base bruta com 1.218 linhas | onboarding duplicado ou migração | manter primeira ocorrência por `cliente_id` |
| Nulo de contato | parte das 175 células nulas | CRM menos digital em relações de assessores seniores | imputar por segmento e canal |
| Nulo estrutural | retorno ou histórico de contato ausente | relação sem histórico suficiente | preservar flag e imputar apenas dentro do pipeline |
| Erro de escala | aproximadamente 0,4% das linhas | casa decimal incorreta | detectar por `p99(segmento) × 20` e dividir por 1.000 |
| Sentinela | `-999` em retorno de cliente sem 12 meses | sistema legado sem `NULL` | converter em nulo estrutural |
| Canal sujo | variantes `R.I.A.` e `ria` | cadastro manual | padronizar para `RIA` |

## Exposição por assessor

A exposição não usa mais o evento binário de saída. O cálculo é `auc_milhoes × prob_saida_assessor`, o que produz distribuição contínua de `pct_carteira_exposta`. O agregado é R$ 9,30 bilhões de R$ 76,05 bilhões, ou 12,2%, com 50 assessores e nenhum sem carteira.

## Decisões de limpeza

1. Deduplicar antes de treinar para não repetir o mesmo cliente no gradiente.
2. Nunca tratar `-999` como retorno real.
3. Não imputar nulo estrutural no dataset inteiro. O `StructuralNullImputer` aprende a mediana apenas no treino e recebe as flags de ausência.
4. Corrigir erro de escala, pois a relação existe e o defeito está na unidade.
5. Tratar `auc_exposto` apenas como produto de dado da Direção B, fora de `FEATURES_V2_BASE`.

## Limites

Esta EDA confirma a coerência interna do gerador, não uma distribuição observada em clientes reais. Os testes em `tests/test_leakage.py` tornam os principais contratos executáveis: escala de AuC, ordenação por segmento, exposição não degenerada e atribuição plausível de clientes por assessor.
