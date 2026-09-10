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

## Resultado (2026-09-10)

- Alertas triados: 5 no total. `python-multipart==0.0.27` concentrava 1 alta e
  3 baixas; atualizado para `0.0.31`, versão que cobre todos os ranges afetados.
  É dependência runtime, mas a página Pages não executa Python; o risco permanece
  relevante apenas para a API/FastAPI de referência.
- `python-dotenv==1.0.1` tinha 1 alerta moderado de symlink durante reescrita de
  `.env`; atualizado para `1.2.2`. O projeto não chama `set_key`/`unset_key`,
  portanto a exposição era baixa, mas o upgrade é compatível e elimina o alerta.
- Instalação revalidada nesta máquina e suíte completa verde (58 testes). O
  SHA-256 de `gb_pipeline_v2.pkl` antes e depois do pipeline é
  `3c255abd...877f4bdcf`: regeneração byte-idêntica.
