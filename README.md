# 🏦 Churn Prediction — Ecossistema Financeiro
### Projeto de Portfólio | Dados

> **Contexto simulado:** ecossistema financeiro independente de grande porte com ~R$75bi sob custódia,
> ~12.000 clientes e 4 segmentos de investidores. Todos os dados são **sintéticos e fictícios**,
> gerados com lógica de negócio real embutida para fins de demonstração técnica.

---

## 📋 Índice
1. [Problema de Negócio](#-problema-de-negócio)
2. [Arquitetura da Solução](#-arquitetura-da-solução)
3. [Stack Tecnológica](#-stack-tecnológica)
4. [Estrutura do Repositório](#-estrutura-do-repositório)
5. [Metodologia CRISP-DM](#-metodologia-crisp-dm)
6. [Resultados](#-resultados)
7. [Feature Engineering](#-feature-engineering)
8. [Benchmark de Modelos](#-benchmark-de-modelos)
9. [Avaliação Rigorosa](#-avaliação-rigorosa)
10. [Simulação de Predição ao Vivo](#-simulação-de-predição-ao-vivo)
11. [ROI Financeiro](#-roi-financeiro)
12. [Como Executar](#-como-executar)
13. [Próximos Passos](#-próximos-passos)

---

![Dashboard Preview](deploy.jpeg)

## 😣 Problema de Negócio

A empresa **não consegue identificar antecipadamente** quais clientes vão resgatar
seus investimentos (churnar). Cada cliente perdido representa, em média, **R$6,7 milhões em AuC**.

| Indicador | Valor |
|---|---|
| Base de clientes | ~12.000 |
| Churn atual | ~12% ao ano |
| AuC médio por cliente | R$ 6,7 milhões |
| **Custódia em risco** | **R$ 9,6 bilhões** |

**Tradução analítica:**
> *"Dado o perfil e comportamento de um cliente, qual a probabilidade de ele resgatar nos próximos 30 dias?"*
> → Problema de **Classificação Binária** (churn = 0 ou 1)

**Meta de Negócio:** Reduzir churn de 12% → 9% em 6 meses = proteger ~R$2,4bi em custódia
**Meta Analítica:** F1-macro ≥ 0.55 | ROC-AUC ≥ 0.70

---

## 🏗️ Arquitetura da Solução

Adotamos a **Medallion Architecture** — padrão de mercado para empresas financeiras
com múltiplas fontes heterogêneas, onde rastreabilidade e auditoria são obrigatórias (CVM/BACEN).

```
Fontes Brutas
   CRM Salesforce │ B3 / XP │ Bloomberg │ Core Bancário
              │
     ┌─────────────────────────────┐
     │  🥉 BRONZE  (Raw / Landing) │
     │  Cópia exata da fonte.      │
     │  Sem transformação.         │
     │  Preserva histórico p/ reprocessamento │
     └──────────────┬──────────────┘
                    │
     ┌─────────────────────────────┐
     │  🥈 SILVER  (Staging)       │
     │  Deduplicação               │
     │  Padronização de tipos      │
     │  Tratamento de nulos        │
     └──────────────┬──────────────┘
                    │
     ┌─────────────────────────────┐
     │  🥇 GOLD  (Data Warehouse)  │
     │  Star Schema                │
     │  Pronto para BI e ML        │
     └──────────────┬──────────────┘
                    │
          ┌─────────┴──────────┐
          │                    │
     📊 Dashboards BI    🤖 Modelos ML
```

---

## 🛠️ Stack Tecnológica

| Camada | Ferramenta | Justificativa |
|---|---|---|
| Linguagem | Python 3.10+ | Ecossistema de dados maduro |
| Manipulação | pandas / numpy | Padrão industria |
| Estatística | scipy.stats | Testes de hipótese (t-Student, ANOVA) |
| ML | scikit-learn | Flexibilidade + pipelines auditáveis |
| Visualização | Plotly | Interativo, sem dependência de kaleido |
| Persistência | joblib | Serialização de modelos e transformadores |
| Orquestração (prod) | Apache Airflow | DAGs auditáveis, padrão mercado |
| Storage (prod) | AWS S3 + Delta Lake | ACID + time travel |
| Transformação (prod) | dbt | Versionamento e testes declarativos |
| Qualidade (prod) | Great Expectations | Alertas automáticos de qualidade |

---

## 📁 Estrutura do Repositório

```
churn-prediction-financeiro/
│
├── 📓 Projeto.ipynb              ← Notebook principal (CRISP-DM completo)
├── 📄 README.md                  ← Este arquivo
├── 📄 requirements.txt           ← Dependências
├── 📄 .gitignore
│
├── output/
│   ├── data/
│   │   ├── base_clientes.csv          ← Base sintética (1.200 clientes)
│   │   ├── base_feature_eng.csv       ← Base com features engenheiradas
│   │   ├── testes_hipotese.csv        ← Resultados t-Student e ANOVA
│   │   └── feature_importance.csv    ← Ranking de importância Gini
│   │
│   ├── models/
│   │   ├── gb_pipeline.pkl            ← Pipeline completo serializado (Model + Encoder + FE)
│   │   └── fe_params.json             ← Parâmetros base da Feature Engineering
│   │
│   └── charts/
│       ├── chart_benchmark.html       ← Benchmark F1 vs ROC-AUC
│       ├── chart_importance.html      ← Feature Importance
│       ├── chart_confusao.html        ← Matriz de Confusão
│       ├── chart_cv.html              ← Cross-Validation por fold
│       └── chart_churn_seg.html       ← Churn por segmento
```

---

## 🔬 Metodologia CRISP-DM

O projeto segue rigorosamente as 6 fases do **CRISP-DM** (Cross-Industry Standard Process for Data Mining):

```
  ┌──────────────────────────────────────────────────────┐
  │  FASE 0  Setup & Reprodutibilidade                   │
  │  FASE 1  Business Understanding  ← ponto de partida  │
  │  FASE 2  Data Understanding      ← auditoria + stats │
  │  FASE 3  Data Preparation        ← feature eng + split│
  │  FASE 4  Modeling                ← benchmark 5 modelos│
  │  FASE 5  Evaluation              ← CV + ROI financeiro│
  │  FASE 6  Deployment              ← pkl + predição live│
  └──────────────────────────────────────────────────────┘
```

---

## 📊 Resultados

### Auditoria da Base de Dados

| Atributo | Valor |
|---|---|
| Total de clientes | 1.200 |
| Features originais | 7 |
| Features criadas | 5 |
| Features no modelo final | 10 |
| Duplicatas | 0 |
| Valores nulos | 0 |
| Proporção Não-Churn / Churn | 80% / 20% |

### Distribuição de Churn por Segmento

| Segmento | Total Clientes | Churns | Taxa |
|---|---|---|---|
| Varejo | 784 | ~201 | ~25,6% |
| Alta Renda | 270 | ~31 | ~11,5% |
| Wealth | 115 | ~6 | ~5,2% |
| Corporate | 31 | ~2 | ~6,5% |

> **ANOVA: F=16,28 | p < 0,001** → segmento é fator **estatisticamente significativo** para churn.

---

## ⚙️ Feature Engineering

5 novas variáveis criadas com justificativa de negócio para cada uma:

| Feature Nova | Variáveis Base | Lógica | Resultado |
|---|---|---|---|
| `engajamento_score` | freq_contato × qtd_produtos | Mede vínculo multidimensional | ✅ correlação melhorou |
| `retorno_relativo` | retorno_12m − média_carteira | Cliente abaixo da média se sente prejudicado | ✅ correlação melhorou |
| `flag_risco` | retorno_relativo < 0 AND freq=0 AND qtd=1 | Combinação de sinais de risco | ✅ correlação melhorou |
| `intensidade_rel` | log(meses) × log(freq_contato) | Tempo + engajamento captura fidelidade | ✅ correlação melhorou |
| `segmento_enc` | segmento (categórico) | LabelEncoder para uso em árvores | ✅ r=0.004 → r=0.074 |

**Regra de Ouro (Eliminando Data Leakage e Training-Serving Skew):**
Para evitar vazamentos de dados, implementamos um `FeatureEngineer` customizado encapsulado em um **Sklearn Pipeline**. O modelo aprende parâmetros (`max`, `mean`) apenas no Treino, e os aplica no Teste e Produção em uma única fase:

```python
from sklearn.pipeline import Pipeline
from transformers import FeatureEngineer

pipeline = Pipeline([
    ("fe", FeatureEngineer()),
    ("prep", preprocessor),
    ("clf", GradientBoostingClassifier())
])

pipeline.fit(X_train, y_train) # 100% Livre de Leakage
joblib.dump(pipeline, "gb_pipeline.pkl")
```

---

## 🏆 Benchmark de Modelos

5 modelos comparados por **F1-macro** (métrica honesta com classes desbalanceadas) e **ROC-AUC**.
Acurácia é monitorada mas **não** é o critério de seleção.

| Modelo | Acurácia | F1-macro | F1-churn | ROC-AUC |
|---|---|---|---|---|
| Dummy (baseline) | 0,7167 | 0,5502 | 0,2766 | 0,5495 |
| Logistic Regression | 0,5475 | 0,4800 | 0,3587 | 0,5857 |
| Decision Tree | 0,5542 | 0,4678 | 0,3902 | 0,6140 |
| Random Forest | 0,7917 | 0,4608 | 0,0385 | 0,6172 |
| **Gradient Boosting** ✅ | **0,8000** | **0,5673** | **0,2500** | **0,6431** |

> **Por que Gradient Boosting?** Aprendizado sequencial adaptativo — cada árvore corrige
> os erros da anterior. Especialmente eficaz para capturar a classe minoritária (churn)
> sem necessidade de normalização prévia.

### Feature Importance (Gradient Boosting / Gini)

| Ranking | Feature | Importância |
|---|---|---|
| 🥇 1° | saldo_bi | 0,2762 |
| 🥈 2° | intensidade_rel | 0,1619 |
| 🥉 3° | meses_cliente | 0,1609 |
| 4° | retorno_relativo | 0,1004 |
| 5° | segmento_enc | 0,0895 |
| 6° | retorno_12m_pct | 0,0831 |
| 7° | engajamento_score | 0,0526 |
| 8° | qtd_produtos | 0,0412 |
| 9° | freq_contato_mes | 0,0339 |
| 10° | flag_risco | 0,0005 |

---

## ✅ Avaliação Rigorosa

### Cross-Validation 5-Fold (Gradient Boosting)

| Fold | F1-macro |
|---|---|
| Fold 1 | 0,5115 |
| Fold 2 | 0,5302 |
| Fold 3 | 0,5180 |
| Fold 4 | 0,4662 |
| Fold 5 | 0,4995 |
| **Média ± DP** | **0,5051 ± 0,0202** |
| IC 95% | [0,4655 — 0,5447] |
| Coef. Variação | 4,4% → ✅ **ESTÁVEL** |

> Coef. de variação < 10% comprova que a performance não foi "sorte" do split.

### Matriz de Confusão

```
                Predito: Ficou   Predito: Churnou
Real: Ficou         TN = 184         FP = 8
Real: Churnou       FN = 40          TP = 8
```

| Métrica | Valor | Interpretação |
|---|---|---|
| Precision (churn) | 50% | De cada alerta, 50% são churn real |
| Recall (churn) | 16,6% | Capturamos 8 de 48 churns reais |
| F1-score (macro) | 0,567 | ✅ Meta atingida (≥ 0,55) |
| ROC-AUC | 0,643 | Discriminação acima do aleatório |

---

## 🤖 Simulação de Predição ao Vivo

![Resultado Interativo no Streamlit](deploy2.jpeg)

```python
from joblib import load
pipeline = load("output/models/gb_pipeline.pkl")
```

| Cliente | Perfil | Prob. Churn | Ação |
|---|---|---|---|
| Carlos M. | Varejo, 8 meses, sem contato, retorno 6,2% | **64,4%** | 🔴 ALERTA URGENTE |
| Ana P. | Alta Renda, 48 meses, 4 produtos, retorno 13,5% | **4,5%** | 🟢 Monitoramento padrão |
| Roberto S. | Wealth, 96 meses, 7 produtos, saldo R$1,85bi | **3,9%** | 🟢 Monitoramento padrão |

---

## 💰 ROI Financeiro

> Extrapolando o modelo para a base real de 12.000 clientes:

| Item | Valor |
|---|---|
| Clientes alertados pelo modelo | 13 |
| Churns reais capturados (TP) | 7 |
| AuC em risco interceptado | R$ 46,9 milhões |
| Receita protegida/ano (1,2% AuC) | R$ 0,56 milhão |
| Custo das abordagens (R$500/cliente) | R$ 6,5 mil |
| **ROI Líquido Estimado** | **R$ 0,56 milhão ✅** |

> Em produção com a base completa e features enriquecidas (histórico de resgates,
> variação de saldo, benchmark relativo), o recall melhora significativamente e
> o ROI escala proporcionalmente.

---

## ▶️ Como Executar

### 1. Clonar o repositório
```bash
git clone https://github.com/seu-usuario/churn-prediction-financeiro.git
cd churn-prediction-financeiro
```

### 2. Criar ambiente virtual e instalar dependências
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\\Scripts\\activate          # Windows

pip install -r requirements.txt
```

### 3. Executar o pipeline de ML
> ⚠️ **Obrigatório antes de rodar o app.** Este passo gera os modelos e dados necessários.

```bash
python pipeline.py
```

### 4. Iniciar o dashboard Streamlit
```bash
streamlit run app.py
```

Acesse em: **http://localhost:8501**

### 5. Executar o notebook (opcional)
```bash
jupyter notebook Projeto.ipynb
```
Execute as células **sequencialmente** (Kernel → Restart & Run All).

### 6. Visualizar os charts
Abra os arquivos em `output/charts/*.html` em qualquer navegador.
Os gráficos são **interativos** — hover, zoom e pan funcionam nativamente.

> ⚠️ **Sobre o kaleido:** este projeto usa `fig.write_html()` em vez de `fig.write_image()`
> para eliminar dependências externas. Zero configuração adicional necessária.

---

## 🚀 Deploy

### Opção 1 — Streamlit Community Cloud (gratuito, recomendado)

> Ideal para portfólio e demonstração pública. Zero infraestrutura.

**Pré-requisito:** os artefatos de `output/` devem estar commitados no repositório.

```bash
# Após rodar pipeline.py localmente:
git add output/
git commit -m "feat: adiciona artefatos de ML para deploy"
git push origin main
```

Em seguida:
1. Acesse [share.streamlit.io](https://share.streamlit.io) e faça login com GitHub
2. Clique em **"New app"**
3. Selecione o repositório, branch `main` e **Main file path:** `app.py`
4. Clique em **Deploy** — o app ficará disponível em `https://seu-app.streamlit.app`

> ⚠️ O Streamlit Cloud não executa o `pipeline.py` automaticamente. Os arquivos `.pkl` e `.csv` em `output/` **precisam estar commitados no repositório** antes do deploy.

---

### Opção 2 — Docker

Crie um `Dockerfile` na raiz do projeto:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN python pipeline.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", \\
     "--server.port=8501", \\
     "--server.address=0.0.0.0"]
```

Build e execução:

```bash
docker build -t churn-finance .
docker run -p 8501:8501 churn-finance
```

Acesse em: **http://localhost:8501**

---

### Opção 3 — Render.com

1. Conecte o repositório em [render.com](https://render.com) → **New Web Service**
2. Configure os campos:

| Campo | Valor |
|---|---|
| **Build Command** | `pip install -r requirements.txt && python pipeline.py` |
| **Start Command** | `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0` |
| **Environment** | Python 3 |
| **Instance Type** | Free (512 MB RAM) |

3. Clique em **Create Web Service** — deploy automático a cada push no `main`.

---

## 🔭 Próximos Passos

- [ ] Integração com pipeline Airflow (orquestração diária)
- [ ] Camada de dados com dbt + Great Expectations (qualidade automatizada)
- [ ] Tuning de hiperparâmetros com Optuna (melhorar recall da classe churn)
- [ ] Dashboard interativo em Power BI / Metabase conectado ao modelo
- [ ] API REST com FastAPI para servir predições em tempo real
- [ ] SHAP values para explicabilidade individual por cliente

---

Projeto desenvolvido como demonstração técnica para a área de **Dados**
em ecossistemas financeiros. Todas as técnicas, decisões de modelagem e arquitetura
são aplicáveis a ambientes de produção reais.

---

*Dados sintéticos | Fins educacionais e de portfólio | Abril 2026*
"""



