# PROBLEM.md — Contrato de Pesquisa
# Projeto: Pipeline Churn Finance

> **Status:** `ATIVO`
> **Versão:** 1.0
> **Data:** 2026-04-27
> **Autor:** Luiz Maibashi
> **Revisado por:** —

---

## 1. Declaração do Problema de Negócio

### 1.1 Contexto

Uma gestora de investimentos enfrenta perda silenciosa de AuC (Assets under Custody) causada pelo churn gradual de clientes. O movimento não é abrupto: o cliente não fecha a conta de uma vez — ele **retira capital progressivamente** até que o relacionamento se torna economicamente inviável de manter.

A área comercial (assessores) atua de forma reativa: só percebe o risco quando o dano já é significativo.

### 1.2 Objetivo do Modelo

> **Prever, com 30 dias de antecedência, quais clientes terão uma queda de AuC que caracteriza churn, permitindo que os assessores iniciem ações de retenção antes da perda se concretizar.**

---

## 2. Definição Formal do Evento-Alvo (Target)

### ✅ Definição Canônica de Churn

Um cliente é classificado como **churn = 1** quando:

```
AuC_atual < 70% do AuC_máximo observado nos últimos 6 meses
E essa condição persiste por 2 meses consecutivos
```

**Exemplos concretos:**

| Cenário | AuC Máx. (6m) | AuC Atual | Churn? |
|---------|--------------|-----------|--------|
| Cliente A | R$ 500.000 | R$ 320.000 | ✅ Sim (64%) |
| Cliente B | R$ 200.000 | R$ 145.000 | ✅ Sim (72,5%) |
| Cliente C | R$ 1.000.000 | R$ 750.000 | ❌ Não (75%) |
| Cliente D | R$ 300.000 | R$ 200.000 — apenas 1 mês | ❌ Não (1 mês, não 2) |

### ❌ O que NÃO é churn neste modelo

- Saques pontuais de curto prazo (ex: compra de imóvel)
- Clientes com AuC zerado por encerramento formal da conta (esse evento tem pipeline separado)
- Variações sazonais de AuC por vencimento de CDB sem resgate para outra instituição

---

## 3. Janela Temporal — Anti-Leakage Contract

> **⚠️ GUARDRAIL CRÍTICO:** Esta seção define as regras que NUNCA podem ser violadas. Qualquer feature que viole essas regras invalida o modelo para produção.

### 3.1 Linha do Tempo

```
|<--- JANELA DE OBSERVAÇÃO (90 dias) --->|<--- PREVISÃO (30 dias) --->|
D-90                                    D-0                         D+30
  [Coleta de Features]                 [Scoring]              [Confirmação do Label]
```

### 3.2 Regras Contratuais

| Regra | Descrição |
|-------|-----------|
| **R-01** | Features só podem usar dados **até D-0 (dia do scoring)** |
| **R-02** | O label (churn=1) é calculado **entre D+1 e D+30** |
| **R-03** | **PROIBIDO** usar como feature: `data_encerramento`, `motivo_saida`, `flag_solicitacao_resgate_pendente` ou qualquer variável calculada após D-0 |
| **R-04** | A janela de observação é de **90 dias corridos** antes de D-0 |
| **R-05** | Em backtesting, o modelo nunca pode "ver o futuro": para cada scoring point no passado, só dados anteriores àquela data são usados |

### 3.3 Variáveis de Alta Suspeita (Requerem Validação Temporal)

- `ultimo_contato_assessor`: pode estar correlacionado com a ação de retenção que JÁ aconteceu
- `retorno_relativo_carteira`: verificar se o cálculo inclui dados do mês de referência do label
- `numero_produtos_ativos`: validar se o resgate de um produto é registrado antes ou depois do label ser confirmado

---

## 4. Custo Assimétrico — Configuração do Threshold

### 4.1 Matriz de Custo

| | **Previsto: Churn** | **Previsto: Não-Churn** |
|---|---|---|
| **Real: Churn** | ✅ Verdadeiro Positivo (VP) — Ação de retenção bem direcionada | ❌ **Falso Negativo (FN) — Custo Alto:** AuC perdida sem intervenção |
| **Real: Não-Churn** | ⚠️ Falso Positivo (FP) — Assessor contatou cliente desnecessariamente (custo: tempo) | ✅ Verdadeiro Negativo (VN) |

### 4.2 Decisão de Threshold

> **FN dói mais que FP.** Perder um cliente com R$ 200k de AuC custa dezenas de vezes mais que uma ligação desnecessária.

**Critério de otimização:**
```
Recall (Sensibilidade) ≥ 0.75
Aceitamos até 40% de taxa de Falsos Positivos
Métrica de avaliação principal: F-beta (β=2), que penaliza FN duas vezes mais que FP
```

### 4.3 Threshold por Segmento

| Segmento | AuC | Threshold de Decisão | Fluxo |
|----------|-----|---------------------|-------|
| Varejo | < R$ 100k | P(churn) ≥ 0.40 | Automático → CRM |
| Middle | R$ 100k – R$ 500k | P(churn) ≥ 0.50 | CRM + Notificação ao Assessor |
| Wealth | > R$ 500k | P(churn) ≥ 0.60 | **Revisão obrigatória por Especialista** |

---

## 5. Fluxo Operacional (Como o Modelo Vive em Produção)

### 5.1 SLA e Cadência

```
Toda segunda-feira, 06h00 (antes da abertura do mercado)
  ↓
Pipeline ingere dados de comportamento da semana anterior
  ↓
Modelo gera scores de risco para todos os clientes ativos
  ↓
Output: lista ranqueada por P(churn) + segmento + AuC em risco
  ↓
Integração com CRM: clientes acima do threshold entram na fila do assessor
```

### 5.2 Modelo de Decisão

```
P(churn) >= threshold?
       ↓
    AuC >= R$ 500k?
    ├── SIM → Fila: Revisão por Especialista (não automatizar)
    └── NÃO → Integração automática no CRM → Assessor responsável recebe tarefa
```

### 5.3 Ação de Retenção Sugerida (por Causa-Raiz)

| Causa Prevista | Ação Sugerida |
|---|---|
| Retorno Relativo baixo vs. mercado | Oferecer produto com CDI+ ou portfólio diversificado |
| Falta de contato recente (>60 dias) | Agendar call consultiva com assessor |
| Concentração em produto único | Apresentar relatório de diversificação |
| Queda de movimentação gradual | Convidar para evento/webinar exclusivo |

---

## 6. Critério de Sucesso — O Anti-BS

> **Como saberemos, em 90 dias após o deploy, que o modelo funcionou?**

### 6.1 Métricas de Modelo (Offline)

| Métrica | Meta Mínima |
|---------|-------------|
| F-beta (β=2) | ≥ 0.65 |
| Recall | ≥ 0.75 |
| Precision | ≥ 0.45 |
| AUC-ROC | ≥ 0.80 |
| Lift no decil 1 | ≥ 3x vs. aleatório |

### 6.2 Métrica de Negócio (Online — 90 dias pós-deploy)

```
Grupo de Intervenção: 50% dos clientes alertados recebem ação de retenção
Grupo de Controle:    50% dos clientes alertados NÃO recebem ação (monitorados)

Critério de Sucesso:
  AuC retida no grupo de intervenção ≥ 15% a mais que no grupo de controle

KPI Secundário:
  Taxa de churn efetivo no grupo de intervenção < 20% vs. grupo de controle
```

### 6.3 Critério de Fracasso (Gatilho para Re-treino)

- F-beta no monitoramento semanal cai abaixo de 0.55 por 2 semanas consecutivas
- Data Drift detectado: distribuição do score muda mais de 20% vs. distribuição de treino (KS-Test)
- Taxa de FN aumenta mais de 30% vs. baseline pós-deploy

---

## 7. Restrições e Guardrails Técnicos

### 7.1 Variáveis Proibidas (Hard Rules)

```python
FORBIDDEN_FEATURES = [
    "data_encerramento_conta",
    "motivo_churn",
    "flag_solicitacao_encerramento",
    "data_ultimo_resgate_total",
    # Qualquer variável com sufixo "_futuro" ou "_projetado"
]
```

### 7.2 Requisitos de Reprodutibilidade

- `random_state=42` em todos os modelos e splits
- Versionamento de modelos: `models/v1/`, `models/v2/` etc.
- Artefatos de treino (scaler, encoder, modelo) salvos juntos no mesmo diretório de versão
- `requirements.txt` ou `environment.yml` atualizado a cada versão

### 7.3 Requisitos de Explicabilidade (LGPD)

- **SHAP Values** implementados para todas as previsões
- Para clientes Wealth (revisão humana): o especialista recebe **top-3 motivos** da previsão em linguagem natural
- Exemplo de output: *"Este cliente tem 78% de probabilidade de churn. Principais fatores: (1) Retorno da carteira 23% abaixo do CDI nos últimos 90 dias, (2) Sem contato com assessor há 74 dias, (3) Concentração de 89% em renda fixa com vencimento em 15 dias."*

---

## 8. Escopo e Fora de Escopo

### ✅ Em Escopo

- Clientes Pessoa Física com conta ativa
- AuC mínimo de R$ 5.000 (clientes abaixo desse valor têm dinâmica diferente)
- Dados dos últimos 12 meses de histórico

### ❌ Fora de Escopo (v1.0)

- Clientes Pessoa Jurídica
- Clientes com conta aberta há menos de 6 meses (sem histórico suficiente)
- Previsão de churn por produto específico (escopo de v2.0)
- Integração em tempo real / near-real-time

---

## 9. Referências e Dependências

| Item | Localização |
|------|-------------|
| Notebook principal | `Projeto_.ipynb` |
| Pipeline de features | `src/feature_engineering.py` |
| Modelo treinado (atual) | `output/model.pkl` |
| App Streamlit | `app.py` |
| Monitor de Drift (Fase 2) | `monitor.py` *(a criar)* |
| SHAP Analysis (Fase 1) | *(a implementar no notebook)* |

---

*Este documento é um contrato vivo. Qualquer alteração na definição de target, janela temporal ou threshold deve ser versionada e aprovada antes de entrar em produção.*
