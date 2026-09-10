# Plano de Implementação: Refatoração do Churn Finance (Enterprise Edition)

Este plano descreve a refatoração do repositório `pipeline_churn_finance` para fazer a transição de um pipeline de ML monolítico para um sistema modular, resiliente e auditável de nível corporativo (Software 3.0).

---

## 📈 Tradução das Etapas: Visão Técnica vs. Visão de Negócio

### 📐 Etapa 1: Modularidade Estilo Kedro
*   **Visão Técnica**: Implementação de catálogo de dados (`catalog.yml`), arquivos de parâmetros (`parameters.yml`) e segregação de I/O em nós de execução puros (`nodes.py`).
*   **Tradução para Negócio (Impacto)**: Garante a **Paridade de Treinamento-Serventia** (*Training-Serving Skew*). Isso significa que as previsões mostradas para o assessor no dashboard são calculadas utilizando a mesmíssima lógica matemática com a qual o modelo foi treinado, eliminando desvios na produção. Além disso, se o banco de dados da corretora mudar, basta alterar uma única linha no catálogo de dados, reduzindo custos de manutenção operacional.

### 🛡️ Etapa 2: Blindagem e Suíte de Testes (TDD para IA)
*   **Visão Técnica**: Testes unitários com `pytest` para regras de negócios e transformação de dados, além de contratos de API estritos baseados em tipos `Pydantic`.
*   **Tradução para Negócio (Impacto)**: Atua como uma **blindagem reputacional e de compliance**. Evita que dados corrompidos ou mal formatados de clientes cheguem ao modelo e gerem scores falsos de propensão a churn. A validação estrita da API garante conformidade com regras regulatórias (ex: LGPD) e assegura que assessores não recebam informações incorretas.

### 📡 Etapa 3: Serving Assíncrono com Worker e Proxy Sidecar Envoy
*   **Visão Técnica**: Endpoint com retorno `202 Accepted` de alta performance, fila com Redis/SQLite e processamento em segundo plano por um worker dedicado, tudo isolado e protegido por um proxy Envoy.
*   **Tradução para Negócio (Impacto)**: **Resiliência extrema e escala sob demanda**. Se um assessor decidir subir uma planilha contendo 1.000 clientes de uma única vez para predição, a API responde instantaneamente em 1ms indicando que o processamento começou. Isso evita telas brancas e travamento do sistema (*timeouts*). O Envoy atua como um escudo, impedindo sobrecargas ou ataques virtuais nos servidores principais, gerando economia de custos de infraestrutura em nuvem.

### 🤖 Etapa 4: Operações Agênticas e Monitoramento de Drift
*   **Visão Técnica**: Cálculo estatístico de drift (KS-Test nas features e previsões) integrado a um loop de orquestração agêntica baseado em LLM.
*   **Tradução para Negócio (Impacto)**: **Gestão de risco ativa para R$ 75 bilhões sob gestão (AuC)**. Os modelos de IA perdem eficácia com o tempo conforme o mercado financeiro muda (ex: alta da taxa Selic alterando o retorno das carteiras). O monitor detecta esse envelhecimento silencioso e, imediatamente, o Agente de IA redige um sumário executivo traduzido para a diretoria, indicando a necessidade de re-treinamento e reduzindo o tempo de reação a mudanças de mercado de semanas para minutos.

---

## 🔍 Definições Importantes (Perguntas Respondidas)

1. **Canal de Alerta de Drift**: O orquestrador (`orchestrator.py`) gerará o relatório executivo em Markdown (`_executive_summary.md`) no disco e simulará uma chamada para um Webhook (Slack/Teams). Se o webhook estiver configurado nas variáveis de ambiente, ele fará o disparo real; caso contrário, imprimirá o payload formatado no terminal, enriquecendo o portfólio.
2. **Docker Compose**: O `docker-compose.yml` será desenhado utilizando a especificação agnóstica moderna (sem declaração rígida de versão obsoleta) e documentado usando o padrão moderno do Docker CLI (`docker compose` sem hífen, Compose V2).

---

## 🛠️ Detalhes dos Arquivos a Serem Modificados/Criados

```
├── conf/
│   └── base/
│       ├── catalog.yml               <- [NEW] Catálogo de dados (I/O)
│       └── parameters.yml            <- [NEW] Hiperparâmetros de ML
├── src/
│   ├── data_processing/
│   │   └── nodes.py                  <- [NEW] Funções puras de processamento
│   ├── model_training/
│   │   └── nodes.py                  <- [NEW] Funções puras de treino e métricas
│   └── kedro_runner.py               <- [NEW] Orquestrador do catálogo
├── tests/
│   ├── test_features.py              <- [NEW] Testes de dados e leakage
│   ├── test_model.py                 <- [NEW] Testes do modelo ML
│   └── test_api.py                   <- [NEW] Testes de contratos da API
├── Dockerfile                        <- [NEW] Build da imagem da API/Worker
├── docker-compose.yml                <- [NEW] Orquestração Envoy + Web + Redis + Worker
├── envoy.yaml                        <- [NEW] Configuração de Ingress do Envoy Sidecar
├── worker.py                         <- [NEW] Processamento assíncrono de jobs
├── api.py                            <- [MODIFY] Endpoints assíncronos dual-mode
├── pipeline.py                       <- [MODIFY] Adaptação para DAG Kedro
├── agent.py                          <- [MODIFY] Fallbacks de API offline
└── orchestrator.py                   <- [MODIFY] Correção de bugs e logs formatados
```
