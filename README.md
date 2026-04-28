# 🚀 Churn Finance Pipeline — De Modelos a Sistemas em Produção

Bem-vindo ao repositório do **Churn Finance Pipeline**. 

Este projeto foi construído como um laboratório de melhores práticas para resolver um problema clássico: como evoluir de um "modelo de laboratório" (focado apenas em otimizar métricas em um Jupyter Notebook) para um **Sistema de Machine Learning em Produção** (seguro, explicável, monitorado e acessível).

---

## 📖 A Narrativa do Projeto (Por que este repositório existe?)

No mercado financeiro atual, prever quem vai dar churn não é suficiente. As áreas de negócio não consomem modelos `.pkl` nem notebooks. Elas consomem APIs, precisam de explicações claras para fins regulatórios e exigem a garantia de que o modelo não está degradando com o tempo.

Inspirado pelos debates recentes na comunidade de MLOps sobre *"The shift from models to systems"* e na visão de LLMs como *"Reasoning Engines"*, este projeto inverte a lógica tradicional da ciência de dados:

1. **O Contrato Primeiro (`PROBLEM.md`):** Antes de treinar o modelo, defini as regras do jogo. O que é o churn? Quais os *guardrails* para evitar *data leakage*? Como medimos sucesso financeiramente? O foco deixou de ser apenas a "previsão" e passou a ser a "especificação" do problema.
2. **Explicabilidade Sistêmica:** Utilizando SHAP, o sistema traduz a matemática do modelo em explicações gerenciais. Não é apenas *"Risco Alto"*; é *"Risco Alto porque o assessor não faz contato há 70 dias"*.
3. **Escala e Robustez:** O modelo não roda solto. Ele é empacotado em um Model Registry (`version_manager.py`), vigiado semanalmente contra degradação (`monitor.py`), e servido para o mundo externo via uma API REST veloz (`api.py`).
4. **Visão Agêntica (Próximo Passo):** Para coroar a acessibilidade, construí um protótipo de **Data Agent** que consome essas APIs. A ideia é permitir que gestores perguntem em linguagem natural: *"Qual o impacto financeiro salvo no segmento Wealth hoje?"*.

---

## 🛠️ Arquitetura e Estrutura do Repositório

### 1. Rigor de Produção e Contratos
*   📄 `PROBLEM.md` — O documento fundacional que define o conceito de churn, as métricas anti-bs e as regras contra vazamento de dados.
*   🧠 `pipeline.py` & `transformers.py` — O pipeline de treinamento em Gradient Boosting com feature engineering encapsulado, prevenindo *Training-Serving Skew*.
*   🔍 `shap_analysis.py` — Módulo de explicabilidade (SHAP) que gera relatórios em linguagem natural por cliente, *LGPD-ready*.
*   📦 `version_manager.py` — Um *Model Registry* simples que salva métricas, hiperparâmetros e gera *snapshots* do modelo (ex: `v1/`, `v2/`).

### 2. MLOps e Interfaces
*   📡 `api.py` — API REST com FastAPI (com *Swagger UI* e *ReDoc*) servindo 6 endpoints essenciais para predição e monitoramento em lote e tempo real.
*   📊 `app.py` — Dashboard Streamlit com *Dark Mode Premium* para uso direto pelos Assessores de Investimento.
*   🛡️ `monitor.py` — Sistema de *Data Drift* semanal usando testes estatísticos robustos (KS-Test, Chi-Quadrado) para detectar se os dados atuais desviaram da base de treino.

### 3. Visão Agêntica (Data Agent PoC)
*   🤖 `agent.py` & `agent_chat.py` — Um protótipo de Agente LLM (via OpenAI) com chamadas a ferramentas (*function calling*). Ele consome a FastAPI para interpretar resultados de drift, consultar clientes em risco e calcular impactos financeiros através de um chat interativo. *(Requer configuração do `.env`)*.

### 4. Estudo Original
*   📓 `notebooks/` — Cadernos originais detalhando a transição do pandas+scikit-learn tradicional até a migração para **PySpark**.

---

## 🖥️ Como Executar a Esteira Completa

**1. Preparar o ambiente:**
```bash
pip install -r requirements.txt
```

**2. O Fluxo de MLOps (Terminal 1):**
```bash
# Treina o modelo inicial
python pipeline.py

# Gera as explicações SHAP por cliente
python shap_analysis.py

# Empacota o modelo no Model Registry (Cria a versão v1)
python version_manager.py

# Verifica o estado da base atual (Data Drift)
python monitor.py
```

**3. Subir os Serviços:**
```bash
# Terminal 2 - Iniciar a FastAPI (Backend)
uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 3 - Iniciar o Dashboard (Para os assessores)
streamlit run app.py

# Terminal 4 - Iniciar o Data Agent (Para a diretoria)
streamlit run agent_chat.py
```

*(Dica: acesse a documentação interativa da API em `http://localhost:8000/docs` após subir a FastAPI).*

---

## 🔮 O Futuro do Projeto

Este projeto demonstra a infraestrutura mínima para um sistema inteligente. O próximo horizonte envolve validar a **Ação Recomendada** no mundo real.
Criaremos um *Feedback Loop* onde as interações do CRM com os clientes sinalizados em risco voltarão para alimentar e avaliar se a intervenção de fato impediu o churn, retroalimentando as iterações futuras do modelo e fechando o ciclo de inteligência de negócios.
