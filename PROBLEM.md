# PROBLEM.md: contrato de pesquisa v2

**Projeto:** Pipeline Churn Finance
**Status:** ativo, dado integralmente sintético
**Versão:** 2.0, 2026-09-10
**Decisões de origem:** ADR-0001 e ADR-0002

## 1. Problema de negócio

O projeto simula uma gestora de wealth management com 1.200 grupos econômicos e R$ 76,05 bilhões de AuC gerado. A pergunta da Direção A é: quais relações apresentam sinais comportamentais de deterioração antes de uma queda relevante de AuC? A pergunta da Direção B é diferente: quanto AuC uma carteira de assessor pode expor se o assessor sair da firma?

As respostas não são transferíveis para uma instituição real. O artefato demonstra contrato de dados, separação de produtos analíticos, avaliação e serving. Não estima ROI, retenção ou receita observada.

## 2. Linguagem e população

| Termo | Definição |
|---|---|
| AuC | Assets under Custody, em `auc_milhoes` no dado e na API. |
| Early-warning comportamental | Recência de contato, variação de cadência e latência de resposta; sinal anterior ao evento. |
| Churn | Queda de AuC acima de 30%, persistente por dois meses; no dado sintético, é o target simulado. |
| AuC exposto | Exposição esperada por saída de assessor: `auc_milhoes × prob_saida_assessor`; não é feature do classificador. |

A população simulada é private banking e multi-family office. Relações com menos de seis meses não entram no escopo. Os segmentos são `Alta Renda`, `Private`, `Wealth` e `Family Office`; cada relação tem de R$ 3 milhões a R$ 2 bilhões de AuC.

| Segmento | n | Mediana de AuC | AuC agregado |
|---|---:|---:|---:|
| Alta Renda | 539 | R$ 7,19 M | R$ 4,47 bi |
| Private | 447 | R$ 28,26 M | R$ 14,94 bi |
| Wealth | 164 | R$ 108,59 M | R$ 21,99 bi |
| Family Office | 50 | R$ 544,63 M | R$ 34,65 bi |

## 3. Contrato temporal e anti-leakage

As features pertencem à janela de observação D-90 a D-0; o score é emitido em D-0; o label é confirmado em D+1 a D+30. É proibido usar `data_encerramento`, `motivo_saida` ou solicitação de resgate pendente como feature. Em backtest, qualquer transformador é ajustado apenas no treino. As regras são testadas em `tests/test_leakage.py`.

## 4. Produtos analíticos

### Direção A: churn do cliente

O modelo v2 recebe segmento, relacionamento, retorno, frequência de contato e três sinais comportamentais. Nulos estruturais são explicitados por flags e imputados dentro do pipeline treinado, nunca no dataset inteiro.

No split de teste de 240 relações, com 28 eventos de churn, a v2 teve recall 7,14% contra 0% da baseline v1; a diferença bootstrap foi +7,21 pontos percentuais, IC95% [0,00; 17,87]. O intervalo inclui zero: o resultado não sustenta uma afirmação robusta de ganho. O ROC-AUC da v2 no mesmo split foi 0,7209. Em CV estratificada de cinco folds, o recall foi 17,14% ± 4,16 p.p.

### Direção B: exposição de carteira do assessor

Cada relação recebe um assessor; a exposição usa a probabilidade calibrada de saída, não o evento binário realizado. O agregado regenerado é R$ 9,30 bilhões, ou 12,2% do AuC. A tabela `carteira_exposta_por_assessor.csv` é descritiva e serve à priorização de retenção de assessor.

## 5. Thresholds e ação

O pipeline seleciona thresholds da v2 pela função `10 × FN + FP`, priorizando falso negativo, com recall alvo de 0,75 quando factível. A seleção ocorre em validação interna, separada do conjunto de teste. Segmentos com menos de cinco eventos positivos usam o threshold global e registram `origem_threshold = global_fallback`.

Os thresholds e suas contagens ficam em `output/data/thresholds_v2.csv` e `reports/thresholds_v2.md`; a API não tem thresholds embutidos. Para cada score, o endpoint retorna classificação, threshold aplicado, AuC em risco (`auc_milhoes × 0,30 × probabilidade`) e fluxo operacional.

## 6. Critérios de aceitação

- AuC simulado entre R$ 65 e R$ 85 bilhões, com mediana crescente por segmento.
- `pct_carteira_exposta` não degenerada e exposição agregada entre 5% e 25%.
- Artefatos `feature_importance_v2.csv`, `thresholds_v2.csv` e SHAP v2 gerados pelo pipeline.
- A v2 supera a baseline v1 em recall no mesmo split, sem declarar ganho de negócio a partir de dado sintético.
- API aceita somente o schema v2 e carrega thresholds persistidos.

## 7. Limites conhecidos

O target e as features são simulados, portanto qualquer métrica mede a coerência do gerador e do pipeline, não desempenho em produção. A baixa quantidade de eventos em `Family Office` e `Wealth` impede threshold próprio estável; por isso há fallback explícito. Avaliar efeito de retenção exigiria dado real, desenho experimental e consentimento operacional, todos fora de escopo.

## 8. Histórico

A versão 1.0, de 2026-04-27, descrevia varejo, thresholds fixos e churn reativo como produto único. Foi substituída pelos ADRs 0001 e 0002; o blueprint de engenharia v1 foi movido para `docs/historia_v1/`.
