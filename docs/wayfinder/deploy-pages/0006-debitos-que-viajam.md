---
tipo: tarefa-simples
status: aberto
criado: 2026-09-10
---

# Ticket 0006: Débitos que viajam junto com o deploy

## Bloqueio

Nada nebuloso — só falta executar e registrar. Dois débitos abertos na frente
`pipeline_churn_finance` que devem fechar junto com o deploy (o repo público vai
ganhar atenção).

## Tarefas

1. **Vulnerabilidades Dependabot no repo público — não investigadas.**
   Contagem no push de 2026-09-10: **5** (1 alta, 1 moderada, 3 baixas) — o
   handoff dizia 4. `https://github.com/luizmaibashi/pipeline_churn_finance/security/dependabot`
   - `gh api repos/<owner>/pipeline_churn_finance/dependabot/alerts` (ou aba
     Security do GitHub) — listar as 4, identificar pacote/severidade/caminho.
   - Verificar se são dependências de runtime (entram no que roda) ou só de dev
     (`jupyter`, `pytest`). Numa página estática sem Python em runtime, o vetor
     de exposição real muda — registrar isso.
   - Para cada uma: bump possível dentro dos pins `==`? Trava por dependência
     transitiva? (ver regra "Dependência transitiva trava patch de CVE" no
     `AGENTS.md` da base — se travar, precisa de data/gatilho de reavaliação,
     não "registrado, não corrigido").
2. **Revalidar `requirements.txt` nesta máquina.**
   - Já está pinado com `==` (a nota da frente `portfolio_deploy` que dizia "sem
     pin ==" está **stale**). Os pins são de 2026-09-10 nesta máquina.
   - `pip install -r requirements.txt` limpo + rodar os 55 testes + regenerar
     `gb_pipeline_v2.pkl` e conferir se continua byte-idêntico.
   - Furo conhecido (`dados.md`): confirmar que `scikit-learn==1.8.0` do
     requirements bate com a versão com que o `.pkl` foi serializado.

## Resultado

_(preencher ao resolver)_
