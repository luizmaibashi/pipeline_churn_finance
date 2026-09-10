# 📐 Blueprint de Refatoração: Churn Finance (Enterprise Edition)

Este documento atua como o guia mestre e prompt inicial para o processo de refatoração do repositório `pipeline_churn_finance`. O objetivo é elevar a qualidade do portfólio no GitHub para o nível corporativo/institucional (Software 3.0), utilizando MLOps, modularidade e workflows agênticos.

---

## 🎯 Objetivo do Projeto
Transformar o pipeline de previsão de rotatividade (Churn) do ecossistema financeiro de alta renda (AuC R$ 75bi) em uma solução robusta de engenharia de software e IA. Focaremos em **desacoplamento de dados, APIs assíncronas concorrentes, testes de contrato e operações agênticas de monitoramento**.

---

## 🏗️ Topologia da Arquitetura Alvo

```mermaid
graph TD
    subgraph Camada de Ingestão e Modelagem (Kedro Pattern)
        A[Data Catalog] --> B[Data Processing Pipeline]
        B --> C[Feature Pipeline]
        C --> D[Model Training Pipeline]
    end

    subgraph Camada de Serving & Infra (FastAPI & Docker)
        E[Envoy Proxy Sidecar] --> F[FastAPI Async API]
        F -->|Payload Pydantic| G[Fila de Mensageria / Redis]
        G --> H[Async Inference Worker]
    end

    subgraph Camada de Operações Agênticas (Software 3.0)
        I[Monitor de Drift KS-Test] -->|Drift Detectado| J[Orchestrator Agent]
        J --> K[Agente de Análise Exploratória EDA]
        K --> L[Relatório Executivo Automatizado em Markdown]
    end
```

---

## 🛠️ Roadmap de Execução (Etapas de Refatoração)

### Etapa 1: Desembaraço e Pipelines Modulares (Kedro)
*   **Ação:** Refatorar as lógicas espalhadas em `pipeline.py` e `transformers.py` adotando a estrutura modular do **[[wiki/concepts/Framework_Kedro|Framework Kedro]]**.
*   **Prática:**
    *   Criar o diretório de dados estruturado e mapear as fontes no `catalog.yml`.
    *   Separar o processamento puro em nodes em arquivos de módulo (ex: `data_processing/nodes.py`, `model_training/nodes.py`).
    *   Decouplar inteiramente a lógica de feature engineering para evitar **[[wiki/concepts/Data_Leakage|Data Leakage]]**.

### Etapa 2: Blindagem e Suíte de Testes (TDD para IA)
*   **Ação:** Criar testes de integridade arquitetural em `tests/` antes das grandes mudanças físicas, seguindo o padrão de **[[wiki/concepts/LLM_Detangling#Padrões de Implementação do Detangling|TDD para IA]]**.
*   **Prática:**
    *   Escrever testes unitários com `pytest` cobrindo o comportamento do cálculo de features e integridade do modelo.
    *   Criar validações de payload de API com `Pydantic` definindo as assinaturas e limites rígidos de entrada e saída.

### Etapa 3: Serving de Inferência Assíncrona e Sidecar
*   **Ação:** Refatorar `api.py` para operar sob o padrão de **[[wiki/concepts/APIs_Assincronas|APIs Assíncronas]]**.
*   **Prática:**
    *   FastAPI recebe a requisição de lote/inferência rápida e insere em uma fila em background (Redis/SQS).
    *   Um worker isolado consome a fila e processa a predição sem causar *Thread Starvation* na API de recebimento.
    *   Estruturar o `docker-compose.yml` contendo a API de Serving protegida por um contêiner **[[wiki/concepts/Padrao_Sidecar|Envoy Proxy Sidecar]]** lidando com controle de tráfego e logs de rede.

### Etapa 4: Operações Agênticas e Monitoramento de Drift
*   **Ação:** Acoplar `monitor.py` a um loop agêntico contínuo de auditoria.
*   **Prática:**
    *   O monitor calcula rotineiramente o teste estatístico Kolmogorov-Smirnov (KS-Test) comparando os dados atuais e os de treino.
    *   Se um drift significativo for detectado, um **Agente de IA** (LLM-based) realiza uma análise estatística e escreve autonomamente um diagnóstico detalhado explicando a anomalia macroeconômica/negócio por trás dos desvios, notificando os tomadores de decisão.

---

## 🔍 Referências na Minha Base de Conhecimento
Para orientar o agente da próxima conversa, utilize os seguintes links internos como especificação arquitetural:
*   **Conceito: Débito Técnico em ML:** [[wiki/concepts/Debito_Tecnico_ML]]
*   **Conceito: Code Churn (Métrica de Atrito):** [[wiki/concepts/Code_Churn]]
*   **Conceito: LLM Detangling (Desembaraço):** [[wiki/concepts/LLM_Detangling]]
*   **Conceito: Padrão Sidecar (Envoy/Rust):** [[wiki/concepts/Padrao_Sidecar]]
*   **Conceito: APIs Assíncronas (Mensageria):** [[wiki/concepts/APIs_Assincronas]]
*   **Conceito: Framework Kedro:** [[wiki/concepts/Framework_Kedro]]
*   **Conceito: Linguagem Ubíqua:** [[wiki/concepts/Linguagem_Ubiqua_e_Grill_with_Docs]]
