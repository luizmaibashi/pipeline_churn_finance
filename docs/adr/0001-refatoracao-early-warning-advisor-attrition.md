# ADR-0001: Refatoração — Early-Warning Comportamental + Risco de Saída de Assessor

**Data:** 2026-09-09
**Status:** Accepted
**Proposto por:** Luiz Maibashi

---

## 1. CONTEXTO (O Quê?)

O `pipeline_churn_finance` é um projeto de portfólio fictício (carteira simulada de R$ 75bi AuC), construído em conversa com um amigo do mercado financeiro. A engenharia (Kedro-style, FastAPI assíncrono, Envoy, agente de drift) está madura — 19 testes passando, arquitetura de nível corporativo. O que faltava era ancorar o problema de negócio numa dor **real** de mercado, não só um enunciado genérico de "prever churn".

**Diagnóstico do modelo atual (v1):**
Target definido em `PROBLEM.md` — `AuC atual < 70% do máximo em 6 meses, por 2 meses consecutivos` — é um sinal **reativo**: mede a queda de capital depois que ela já começou. Pesquisa de mercado (2026) confirma que esse é exatamente o padrão que a indústria está tentando superar: *"traditional models only catch churn after a client requests an asset transfer."*

**Achado que motivou o escopo novo:** a literatura de wealth management 2026 aponta duas dores não cobertas pelo modelo v1:
1. Sinal de verdade antecede a queda de AuC — vem de comportamento (queda de frequência de contato, latência de resposta, tom de interação).
2. Uma fatia relevante do churn não é decisão do cliente — é decisão do **assessor**: quando ele muda de firma, carrega carteira consigo (**+US$100bi em AuM transferido em 90 dias** após ondas de saída de assessores em 2023; RIA channel recebeu 32,1% dos advisors "dual" em 2025 — maior fluxo do dataset).

**Linguagem Ubíqua (termos novos):**
- **Early-warning comportamental**: sinal antecedente à queda de AuC, calculado a partir de cadência/qualidade de interação cliente-assessor (não a partir do saldo).
- **Advisor attrition**: risco do assessor responsável pela conta deixar a firma (não confundir com client churn — causa raiz diferente).
- **AuC exposto**: fração da carteira de um assessor que tende a migrar junto se ele sair (proxy de risco herdado, não risco do cliente em si).
- **Sinal reativo vs. antecedente**: reativo = mede o efeito já ocorrido (queda de AuC); antecedente = mede a causa antes do efeito aparecer.

---

## 2. DECISÃO (Por Quê?)

Expandir o escopo do projeto em duas frentes compostas, mantendo a engenharia v1 como base:

**Direção A — Early-warning comportamental**: adicionar features/target baseados em sinal comportamental antecedente (frequência de contato, latência de resposta, proxy de sentimento de interação), complementando — não substituindo — o target de queda de AuC já contratado no `PROBLEM.md`.

**Direção B — Risco de saída de assessor → AuC exposto**: nova entidade `assessor` (hoje inexistente no dataset — schema atual é só por cliente), com target próprio de risco de saída, e uma métrica derivada de AuC exposto por carteira de assessor.

**Razão Principal:** sem isso, o projeto prova competência de engenharia mas não prova que entende o domínio de negócio que afirma resolver — a mesma distância entre "peça construída" e "peça entregue" que a base já documenta como risco recorrente em outros projetos.

"Se não fizermos isso": o portfólio mostra um sistema tecnicamente impressionante resolvendo um problema genérico e desatualizado (churn reativo por queda de saldo).
"Se fizermos": o projeto passa a refletir a dor de 2026 real da indústria — early-warning antecedente + risco estrutural de mobilidade de assessor — com engenharia que já suporta o novo escopo sem reescrever a base.

---

## 3. CONSEQUÊNCIAS

**Positivas:**
- Projeto ganha dois ângulos de negócio defensáveis contra alguém do mercado financeiro real.
- Reaproveita 100% da infraestrutura v1 (Kedro catalog, API, agente, monitor) — o novo escopo é dado + feature + target, não arquitetura nova.
- Fecha o gap de leakage testing (achado no Blind Spot Pass) como efeito colateral, porque os novos targets exigem contrato temporal explícito desde o dia zero.

**Negativas:**
- Escopo multi-sessão (schema novo + 2 targets + geração sintética calibrada + retreino + testes de leakage para ambos) — não é refatoração de tarde.
- Dataset 100% sintético: risco de o gerador produzir correlação "boa demais" e o ganho do modelo virar artefato de simulação, não achado transferível. Mitigado por checkpoint explícito (seção 5).
- Nenhum dataset público real cobre os dois casos — decisão de gerar sintético é definitiva para este projeto (dados reais de advisor movement são proprietários/pagos: AdvizorPro, FINTRX).

---

## 4. ALTERNATIVAS DESCARTADAS

| Opção | Por quê foi rejeitada |
|-------|----------------------|
| C — Manter escopo v1, só robustecer engenharia (Pydantic, testes, visibilidade) | Não resolve a lacuna de negócio identificada; menor valor de portfólio/aprendizado |
| Usar dataset público de churn bancário de varejo (Kaggle, ~10k linhas) diretamente | Não é wealth/private banking, não tem sinal comportamental, não serve pro caso de uso — só referência de estrutura |
| Comprar/obter dado real de advisor movement (AdvizorPro/FINTRX) | Proprietário, pago, fora do escopo de projeto de portfólio fictício |

---

## 5. IMPACTO ROI (critério de sucesso, adaptado — projeto fictício sem métrica de negócio real)

- **Métrica de sucesso técnica:** modelo com sinal antecedente (Direção A) supera em recall/AUC uma baseline reativa equivalente ao target v1, medido sobre o dado sintético — identificador citável a implementar: `recall_early_warning_vs_baseline_reativo` (função/relatório a criar em `reports/`).
- **Métrica de sucesso de negócio:** tradução defensável — cada feature nova tem frase de conexão com a dor de mercado citada nesta ADR (não só tipo/estatística).
- **Checkpoint anti-artefato-de-simulação:** antes de reportar qualquer ganho do modelo A/B como "achado", validar que a correlação sintética não é trivial/determinística demais — inspecionar distribuição de features novas contra ruído razoável, documentar em `reports/`. Mesmo risco já visto em `abracaf_ecossistema` (dado ruim → conclusão errada).
- **Timeline:** multi-sessão; commit após cada etapa fechada (schema, gerador sintético, features, retreino, testes) — trabalho cruza duas máquinas.
- **Risco de regressão:** os 19 testes v1 (`tests/`) devem continuar passando — nenhuma mudança na Direção A/B pode quebrar o contrato de API/serving já existente.

---

## 6. CORREÇÃO PÓS-IMPLEMENTAÇÃO (2026-09-09, auditoria pedida pelo Luiz)

Implementação original tratava `auc_exposto` (Direção B) como **feature** do modelo de churn, junto com o sinal comportamental (Direção A) — a leitura de "A+B compostas" da decisão original foi interpretada como "entram no mesmo classificador".

**Achado da auditoria:** feature importance mostrou `auc_exposto` com 1,3% de peso no modelo — investigação confirmou que não é sinal fraco, é a métrica certa pro objetivo errado. Risco de saída de assessor é quase independente do churn individual do cliente (`corr(risco_saida_assessor, churn) = -0,04`, medido), porque são fenômenos causalmente distintos: um cliente pode sair por insatisfação própria sem o assessor sair, e vice-versa.

**Correção:** `auc_exposto` saiu de `FEATURES_V2_EXTRA` (não entra mais no classificador). Virou `aggregate_carteira_exposta_por_assessor()` — produto de dado separado, agregado por `assessor_id`, respondendo "se ESTE assessor sair, quanto AuC da carteira está exposto" (insumo pra dashboard de risco de carteira, não pra prever churn de cliente).

**Releitura correta de "A+B compostas":** não significa "mesmo modelo, mais features" — significa **dois produtos de dado do mesmo projeto**, cada um respondendo uma pergunta de negócio diferente:
- Direção A → classificador de churn do cliente (usa sinal comportamental)
- Direção B → agregação de exposição de risco por assessor (não é preditiva, é descritiva/priorização)

**Resultado pós-correção:** recall v2 melhorou levemente (CV 5-fold: 0,3125→0,2958, mais estável — desvio caiu de 0,0437 para 0,0358) ao remover o ruído de uma feature sem função ali. As 3 features comportamentais passaram a somar 55,7% da importância do modelo, confirmando que o ganho é genuíno.

---

## 7. LINKS RELACIONADOS

- [[PROBLEM.md]] — contrato de dados v1 (target reativo, ainda vigente como componente)
- [[refactoring_blueprint.md]] — arquitetura de engenharia v1 (Kedro/FastAPI/Envoy), base que este ADR estende
- Pesquisa de mercado (2026-09-09, WebSearch): early-warning comportamental, advisor attrition/AuM transferido, dados de migração RIA/broker-dealer
- Gate relevante (`.claude/rules/dados.md`): GATE ML "Criterio escrito em ADR precisa aparecer em codigo executavel" — `recall_early_warning_vs_baseline_reativo` citado acima precisa existir em `.py` fora de comentário quando implementado
