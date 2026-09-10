---
tipo: grilling
status: resolvido
criado: 2026-09-10
resolvido: 2026-09-10
---

# Ticket 0004: Como o enquadramento honesto sobrevive ao deploy

## Bloqueio

O projeto se vende como "demonstração de método, não de resultado". Os números
reais:

- Recall v2 = 7,14% vs 0% da baseline (n=240, 28 eventos de churn).
- IC95% bootstrap da diferença = [0,00; 17,87] p.p. — **inclui zero**.
- ROC-AUC v2 = 0,7209; CV 5-fold recall = 17,14% ± 4,16 p.p.
- Base 100% sintética, gestora fictícia.

Está honesto no `README.md §"Avaliação, sem extrapolar"` e no `PROBLEM.md §7`. O
risco no deploy: uma página bonita com um gauge de risco e "AuC em risco: R$ X
mi" **parece** um produto comercial. O disclaimer não pode virar rodapé cinza
que o recrutador não lê.

## Perguntas para o Luiz

1. **Onde o enquadramento aparece.** Banner fixo no topo? Card de destaque antes
   das abas? Texto obrigatório na própria aba de Predição, ao lado do gauge?
2. **Tom.** "Dado sintético — isto demonstra método de estruturação, não
   performance" — essa frase, ou uma versão mais dura ("o IC inclui zero: o
   experimento não prova ganho")?
3. **O número "AuC em risco".** Manter o cálculo `AuC × 30% × prob` na tela (é
   didático) mas rotular explicitamente como ilustração da mecânica, não como
   projeção? Ou tirar da página pública?
4. Passar o texto final pela skill `escrita-organica` antes de publicar (README
   público conta como texto de saída).

## Resultado (Luiz, 2026-09-10)

1. **Card de destaque antes das abas** — bloco visível na entrada da página (não
   fixo no scroll). Conteúdo: dado 100% sintético / gestora fictícia; a v2 supera
   a v1 em recall no mesmo split (+7,1 p.p.) mas com ~28 eventos de churn o IC95%
   da diferença inclui zero — **não é ganho robusto**; nenhum número é efeito de
   negócio observado. É demonstração de método (contrato temporal, separação de
   produtos analíticos, threshold por custo, serving versionado).
2. Reusar o texto que já existe em `app.py:814-825` ("O que este projeto
   demonstra") e no `README.md §"Avaliação, sem extrapolar"` — não reinventar.
3. **"AuC em risco" (`AuC × 30% × prob`)** fica na Tab 1, rotulado explicitamente
   como ilustração da mecânica de priorização, não projeção financeira.
4. Passar o texto final do card + qualquer copy nova da página pela skill
   `escrita-organica` antes de publicar (README público = texto de saída).
