# Diretrizes de Condução: Metodologia de Refatoração

Este documento estabelece o processo de trabalho, as boas práticas de engenharia de software e os critérios de qualidade para a condução da refatoração do projeto **Churn Finance**.

---

## 🧭 Como Conduziremos a Refatoração

Nossa abordagem seguirá os princípios de **Engenharia de Software 3.0** e **Desenvolvimento Orientado a Testes (TDD) para IA**, divididos em ciclos claros:

### 1. Isolamento e Segurança (Garantia de Não-Regressão)
- **Antes de qualquer alteração estrutural**, criaremos os testes unitários fundamentais na pasta `tests/` cobrindo o comportamento atual do monolito (cálculo de features e carregamento do modelo).
- Isso garantirá que, conforme quebramos o código de `pipeline.py` e `transformers.py` em pedaços modulares, poderemos rodar o `pytest` a qualquer instante para saber se quebramos a lógica analítica do modelo.

### 2. Implementação Modular Incremental (Padrão Kedro)
- Criaremos as pastas e arquivos de configuração primeiro (`conf/base/catalog.yml` e `conf/base/parameters.yml`).
- Criaremos as pastas `src/data_processing` e `src/model_training` e migraremos as funções do monolito para estes diretórios na forma de **Nodes puros** (funções que apenas recebem parâmetros/DataFrames e devolvem outputs, sem I/O direto).
- Implementaremos o `kedro_runner.py` e refatoraremos o `pipeline.py` para chamar este runner.
- Validaremos se o pipeline modular está funcionando rodando `python pipeline.py` e verificando se os testes continuam passando.

### 3. Implementação do Serving Assíncrono (FastAPI + Worker)
- Atualizaremos o `api.py` para introduzir o processamento em lote assíncrono.
- Criaremos o `worker.py` e o banco de dados SQLite local `output/jobs.db` para gerenciar a fila e o estado das tarefas.
- Garantiremos que, localmente (sem Docker/Redis), a API funcione de forma transparente usando as threads de segundo plano do próprio Python (`BackgroundTasks`).
- Em seguida, empacotaremos tudo em containers no `Dockerfile` e no `docker-compose.yml`, introduzindo o Redis como broker em produção e o Envoy Proxy como sidecar de segurança e roteamento na porta `8080`.

### 4. Integração do Loop Agêntico e Auditoria de Drift
- Corrigiremos o bug do orquestrador de drift (`orchestrator.py`), alterando o acesso do dicionário de `n_alertas` para `n_alerts`.
- Refatoraremos o `agent.py` para suportar fallback offline: se o servidor do FastAPI estiver indisponível no momento do teste estatístico semanal, o agente automaticamente lerá as tabelas CSV e metadados diretamente do disco (`output/`) para redigir o resumo executivo sem falhar.

---

## 🛡️ Critérios de Qualidade e Guardrails

*   **Preservação de Contexto e Documentação**: Todos os comentários explicativos relevantes e docstrings devem ser mantidos e adaptados nas funções modularizadas.
*   **Paridade de Métricas**: O modelo final treinado pela estrutura modular Kedro deve atingir no mínimo as mesmas métricas contratuais do `PROBLEM.md` (F1-macro >= 0.55 e ROC-AUC >= 0.70).
*   **Contratos Rígidos na API**: Payloads que não atendam exatamente aos limites do Pydantic (ex: tempo de relacionamento negativo ou saldo negativo) devem ser rejeitados imediatamente com código `422 Unprocessable Entity`.
*   **Sem Efeitos Colaterais nos Nós**: Os arquivos `src/data_processing/nodes.py` e `src/model_training/nodes.py` não devem conter funções que leiam/salvem arquivos diretamente com `pandas.read_csv`, `joblib.dump` ou semelhantes. Todo o I/O deve passar obrigatoriamente pela abstração do `Catalog` no `kedro_runner.py`.

---

## 🚦 Acompanhamento de Progresso

Conforme avançamos, utilizaremos a ferramenta de acompanhamento `task.md` na raiz da pasta de histórico do agente (`C:\Users\Luiz Maibashi\.gemini\antigravity\brain\b3f58387-d7a2-4910-a14c-d0e8091ae3bf/task.md`) para marcar as tarefas como pendentes `[ ]`, em andamento `[/]` ou concluídas `[x]`.
- Cada etapa concluída será validada e documentada com um passo a passo para execução transparente.
- Ao final, geraremos um relatório de encerramento (`walkthrough.md`) com capturas e testes executados.
