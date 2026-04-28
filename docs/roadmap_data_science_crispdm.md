# 🗺️ Roadmap de Data Science com CRISP-DM
### Template de Estudo e Referência para Projetos Futuros

> **Como usar este documento:**
> - 📖 **1ª leitura:** Leia do início ao fim para entender o fluxo completo
> - ✍️ **Durante o projeto:** Use as seções de anotação para registrar suas descobertas
> - ♻️ **Próximos projetos:** Copie e adapte como template

---

## 🧭 O Framework CRISP-DM — A Espinha Dorsal

```
┌─────────────────────────────────────────────────────────────┐
│                    CRISP-DM (Visão Geral)                    │
│                                                               │
│   1. Business          2. Data              3. Data           │
│   Understanding    →   Understanding    →  Preparation       │
│        ↑                                        ↓            │
│   6. Deployment        5. Evaluation    ←  4. Modeling       │
└─────────────────────────────────────────────────────────────┘
```

> **Princípio central:** As fases são **iterativas**, não lineares. Você pode (e deve) voltar à fase anterior quando aprender algo novo. Um bom projeto dá no mínimo 2-3 voltas completas no ciclo.

**O que diferencia um projeto CRISP-DM de um projeto "só de ML"?**
- Começa com **dor de negócio**, não com dataset
- Tem critérios de sucesso **do negócio** antes de critérios de ML
- O modelo é apenas **um meio** — o fim é a **decisão melhor**

---

## FASE 1 — 🏢 Business Understanding (Entendimento do Negócio)

> *"Nunca comece pelo código. Comece pela pergunta."*

### O que fazer nessa fase

Antes de abrir qualquer dataset ou escrever qualquer código, você precisa entender **por que** esse projeto existe.

### Checklist de Business Understanding

```
[ ] 1. Qual é a DOR do negócio? (Problema real, não técnico)
[ ] 2. Quem sofre com essa dor? (Stakeholder principal)
[ ] 3. O que acontece se nada for feito? (Custo da inação)
[ ] 4. O que é sucesso para o negócio? (Meta de negócio)
[ ] 5. O que é sucesso para o modelo? (Meta analítica/técnica)
[ ] 6. Quais dados existem? (Avaliação inicial de recursos)
[ ] 7. Qual é o prazo e o orçamento?
[ ] 8. Como o resultado será consumido? (Relatório? API? Dashboard?)
```

### Perguntas Guia para Qualquer Projeto

| Pergunta | Por Que Importa |
|---|---|
| Quem toma a decisão com base no modelo? | Define o formato de entrega |
| Com que frequência o modelo será usado? | Define latência aceitável |
| Qual é o custo de um Falso Positivo? | Define o threshold ideal |
| Qual é o custo de um Falso Negativo? | Define a métrica principal |
| Existe um processo manual hoje? | Baseline de comparação |

### Exemplo Deste Projeto (NPS Predictor)

| Elemento | Resposta |
|---|---|
| **Dor de Negócio** | NPS médio caiu para 4.38, 84.4% dos clientes são Detratores |
| **Stakeholder** | Customer Success + Diretoria de Operações |
| **Custo da Inação** | Perda de LTV, boca-a-boca negativo, aumento de CAC |
| **Meta de Negócio** | Reduzir Detratores em 15% em 6 meses |
| **Meta Analítica** | F1-Score macro ≥ 0.50 em dados sem Data Leakage |
| **Como será usado** | Streamlit App com predição em tempo real |

### Armadilhas a Evitar

- **Ir direto ao código** sem entender o negócio
- **Definir sucesso pela acurácia** antes de entender o problema
- **Não perguntar "e daí?"** — qual ação o modelo vai habilitar?

### 📝 Minhas Anotações — Business Understanding

```
PROBLEMA IDENTIFICADO:


STAKEHOLDERS:


META DO NEGÓCIO:


META DO MODELO:


PERGUNTA ANALÍTICA CENTRAL:
```

---

## FASE 2 — 🔍 Data Understanding (Entendimento dos Dados)

> *"Dados sujos = Modelos ruins. Garbage in, Garbage out."*

### O que fazer nessa fase

Conhecer profundamente o dataset **antes** de transformá-lo. Esta fase é exploratória — não há hipótese para provar ainda, há perguntas para responder.

### Checklist de Data Understanding

```
[ ] 1. Dimensões do dataset (linhas x colunas)
[ ] 2. Tipos de dados de cada coluna (numérico, categórico, datas)
[ ] 3. Valores nulos — quantidade e padrão
[ ] 4. Valores duplicados
[ ] 5. Estatísticas descritivas (média, mediana, desvio, min, max)
[ ] 6. Distribuição da variável alvo (balanceamento de classes)
[ ] 7. Correlações entre features e variável alvo
[ ] 8. Outliers identificados
[ ] 9. Identificação de possíveis Leakage Variables
```

### Análises Essenciais de EDA

```python
# 1. Visão geral do dataset
df.shape          # dimensões
df.dtypes         # tipos de dados
df.head()         # primeiras linhas
df.describe()     # estatísticas descritivas

# 2. Qualidade dos dados
df.isnull().sum()       # valores nulos
df.duplicated().sum()   # duplicatas

# 3. Distribuição da variável alvo
df['target'].value_counts(normalize=True)  # proporção de classes

# 4. Correlações
df.corr()['target'].sort_values(ascending=False)
```

### O Que Investigar na Variável Alvo

```
TIPOS DE PROBLEMA x VARIÁVEL ALVO:

  Contínua    → Regressão  (predizer um número)
  Binária     → Classificação binária (sim/não, 0/1)
  Multiclasse → Classificação multi (A/B/C)
  Sem alvo    → Clustering (agrupar por similaridade)
```

### Armadilha Principal: Data Leakage

> **Data Leakage** é usar no treino informações que não existiriam no momento da predição real.

| Tipo | Exemplo | Sinal de Alerta |
|---|---|---|
| **Leakage Temporal** | Usar vendas de dezembro para prever novembro | Correlação suspeitosamente alta (>0.9) |
| **Leakage de Target** | Usar informações criadas após o evento | A feature "existe" só depois da target |
| **Leakage de Grupo** | Mesmo cliente em treino e teste | Performance cai bruscamente em produção |

**No projeto NPS**: `csat_internal_score` e `repeat_purchase_30d` eram Leakage — só existem após a pesquisa de NPS, não antes.

```
F1-Score SEM leakage (correto):  0.560
F1-Score COM leakage (errado):   0.900
Ganho ARTIFICIAL:                +61% (não existe em produção!)
```

### O Problema do Desbalanceamento de Classes

```
Se 95% dos dados são da Classe A:
  → Um modelo que SEMPRE prevê "A" tem 95% de Acurácia
  → Mas tem 0% de utilidade!

Solução: Usar F1-Score, Precision, Recall, AUC-ROC
  em vez de Acurácia como métrica principal
```

### Testes Estatísticos de Hipótese

| Situação | Teste | Interpretação |
|---|---|---|
| Comparar médias de 2 grupos | **T-Test de Welch** | p < 0.05 → diferença real |
| Comparar médias de 3+ grupos | **ANOVA** | p < 0.05 → pelo menos um diferente |
| Correlação numérico-numérico | **Pearson / Spearman** | r > 0.5 → correlação forte |
| Associação categórico-categórico | **Chi-quadrado** | p < 0.05 → associação real |

### 📝 Minhas Anotações — Data Understanding

```
DIMENSÕES DO DATASET:     ___ linhas x ___ colunas

TIPOS DE VARIÁVEIS:
  - Numéricas:
  - Categóricas:
  - Datas:

QUALIDADE:
  - Nulos totais:
  - Duplicatas:

VARIÁVEL ALVO:
  - Nome:
  - Tipo (contínua/discreta):
  - Distribuição de classes (se aplicável):
  - Desbalanceamento? Sim/Não — Quão grave?

PRINCIPAIS ACHADOS DA EDA:
  1.
  2.
  3.

SUSPEITAS DE LEAKAGE:


HIPÓTESES A TESTAR:
  H1:
  H2:
```

---

## FASE 3 — 🧹 Data Preparation (Preparação dos Dados)

> *"80% do trabalho de um Data Scientist é preparar os dados."*

### O que fazer nessa fase

Transformar os dados brutos em um formato que os algoritmos de ML possam aprender.

### Checklist de Data Preparation

```
[ ] 1. Tratar valores nulos (imputação ou remoção)
[ ] 2. Tratar duplicatas
[ ] 3. Encoding de variáveis categóricas
[ ] 4. Normalização/Padronização de variáveis numéricas
[ ] 5. Feature Engineering (criar novas variáveis)
[ ] 6. Remover variáveis de Leakage
[ ] 7. Separar treino e teste (ANTES de qualquer transformação)
[ ] 8. Aplicar transformações APENAS com base nos dados de treino
```

### A Ordem Certa de Preparação

```python
# CORRETO: Separar ANTES de transformar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ERRADO: Normalizar ANTES de separar (causa Data Leakage!)
# scaler.fit(X)   <- usa informação do teste no treino!
# X_scaled = scaler.transform(X)
# X_train, X_test = train_test_split(X_scaled, ...)

# CORRETO: Fit no treino, Transform no teste
scaler = StandardScaler()
scaler.fit(X_train)                   # Aprende estatísticas SÓ do treino
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test) # Usa parâmetros do treino
```

### Estratégias para Valores Nulos

| Situação | Estratégia |
|---|---|
| Nulos < 5% dos dados | Remover as linhas |
| Variável numérica | Imputar com mediana |
| Variável categórica | Imputar com moda |
| Nulos com padrão | Criar flag binária "era_nulo" |
| Muitos nulos (> 40%) | Considerar remover a coluna |

### Feature Engineering — O Diferencial Competitivo

> Features bem construídas valem mais que algoritmos sofisticados.

**Princípio:** Novas features devem ter **intuição de negócio clara**

```python
# Exemplo do projeto NPS
df['ratio_atraso_entrega'] = df['delay'] / (df['prazo'] + 1)
# Por que? 3 dias de atraso em frete expresso != 3 dias em frete normal

df['score_logistica'] = -df['delay']*2 - df['tentativas'] + df['pontual']*5
# Por que? Score composto captura a experiência logística como um todo
```

### Pipeline Sklearn — Eliminando Training-Serving Skew

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# O Pipeline garante que o Scaler seja aplicado automaticamente
# em produção, evitando inconsistências entre treino e deployment
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

pipeline.fit(X_train, y_train)
# Salvar: joblib.dump(pipeline, 'models/pipeline.pkl')
```

### 📝 Minhas Anotações — Data Preparation

```
TRATAMENTO DE NULOS:
  - Estratégia usada:
  - Justificativa:

VARIÁVEIS REMOVIDAS (Leakage ou irrelevantes):
  -
  -

ENCODING APLICADO:
  - Tipo:
  - Nas colunas:

NORMALIZAÇÃO:
  - Tipo (StandardScaler / MinMaxScaler):
  - Aplicado em:

FEATURES CRIADAS (Feature Engineering):
  | Feature | Fórmula | Intuição de Negócio |
  |---------|---------|---------------------|
  |         |         |                     |

DIMENSÕES FINAIS PARA MODELAGEM:
  - X_train: _____ linhas x _____ features
  - X_test:  _____ linhas x _____ features
  - Classes: _____ (balanceadas? _____%)
```

---

## FASE 4 — 🤖 Modeling (Modelagem)

> *"O modelo é apenas uma ferramenta. A sabedoria está em escolher a ferramenta certa para o problema certo."*

### O que fazer nessa fase

Selecionar, treinar e comparar algoritmos. Não existe "melhor algoritmo" — existe o melhor para **este** problema com **estes** dados.

### Checklist de Modelagem

```
[ ] 1. Escolher algoritmos candidatos baseado no tipo de problema
[ ] 2. Definir a métrica principal de avaliação (alinhada ao negócio)
[ ] 3. Treinar baseline (modelo simples como referência)
[ ] 4. Treinar e comparar múltiplos modelos
[ ] 5. Aplicar Cross-Validation para validação robusta
[ ] 6. Ajustar hiperparâmetros do melhor modelo
[ ] 7. Analisar a Matriz de Confusão
[ ] 8. Otimizar threshold de decisão se necessário
```

### Mapa de Algoritmos por Tipo de Problema

```
CLASSIFICAÇÃO:
  Rápido/Interpretável: Logistic Regression, Decision Tree
  Ensemble (recomendado): Random Forest, Gradient Boosting, XGBoost
  Poderoso: SVM, Neural Networks

REGRESSÃO:
  Rápido/Interpretável: Linear Regression, Ridge, Lasso
  Ensemble: Random Forest Regressor, Gradient Boosting
  Poderoso: SVR, Neural Networks

AGRUPAMENTO (sem alvo):
  Baseado em distância: K-Means, DBSCAN
  Hierárquico: Agglomerative Clustering
```

### Métricas por Tipo de Problema

```python
# CLASSIFICAÇÃO
# Dados balanceados        → Accuracy
# Dados desbalanceados     → F1-Score Macro, ROC-AUC
# Custo de FP alto         → Precision
# Custo de FN alto         → Recall

# REGRESSÃO
# MAE (erro médio absoluto)      — robusto a outliers
# RMSE (raiz do erro quadrático) — penaliza erros grandes
# R² (coeficiente de determinação) — % da variância explicada
```

### Cross-Validation — Validação Robusta

```python
from sklearn.model_selection import cross_val_score

# Em vez de um único treino/teste, usa-se K folds
scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='f1_macro')

print(f"F1 médio:  {scores.mean():.3f}")
print(f"F1 stdev:  {scores.std():.3f}")   # Baixo = modelo estável
```

> **Por que usar?** Um único resultado treino/teste pode ser sorte (ou azar). 5-Fold CV usa 5 divisões diferentes e reporta a média — resultado muito mais confiável.

### A Armadilha da Acurácia em Dados Desbalanceados

```
EXEMPLO (Projeto NPS — 84.4% Detratores):

Modelo A — Prevê SEMPRE "Detrator":
  Accuracy = 84.4%  (parece ótimo!)
  F1-Macro = 0.30   (terrível!)
  Neutros e Promotores detectados = 0%

Modelo B — Random Forest com class_weight='balanced':
  Accuracy = 72%    (parece pior...)
  F1-Macro = 0.56   (muito melhor!)
  Detecta todas as 3 classes
```

### Otimização de Threshold

> Por padrão, Sklearn usa 50% como threshold. Em problemas reais, ajuste esse corte com base nos custos de FP vs. FN.

```python
from sklearn.metrics import precision_recall_curve

probs = modelo.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)

# Encontrar threshold que maximiza Recall (quando FN é mais caro)
melhor_threshold = thresholds[np.argmax(recall >= 0.78)]
```

### 📝 Minhas Anotações — Modelagem

```
BASELINE (modelo mais simples de referência):
  - Algoritmo:
  - Resultado (métrica principal): _____

MODELOS TESTADOS:
  | Modelo | F1-Macro | Accuracy | Observações |
  |--------|----------|----------|-------------|
  |        |          |          |             |
  |        |          |          |             |
  |        |          |          |             |

MELHOR MODELO:

CROSS-VALIDATION (5 folds):
  - F1 médio:
  - Desvio padrão:
  - Estável? Sim/Não

HIPERPARÂMETROS AJUSTADOS:
  - Parâmetros:
  - Método: GridSearchCV / RandomSearchCV / Manual

THRESHOLD OTIMIZADO:
  - Padrão (0.5) vs. Otimizado (___):
  - Impacto no Recall:
```

---

## FASE 5 — 📊 Evaluation (Avaliação)

> *"Um modelo que funciona bem no lab mas falha em produção não vale nada."*

### O que fazer nessa fase

Avaliar se o modelo atende às metas de negócio definidas na Fase 1 — não apenas às métricas técnicas.

### Checklist de Avaliação

```
[ ] 1. Performance nos dados de teste (não vistos no treino)
[ ] 2. Matriz de Confusão — Entender cada tipo de erro
[ ] 3. Comparar com o baseline
[ ] 4. Validar que a meta analítica foi atingida
[ ] 5. Verificar se a meta de negócio é alcançável com essa performance
[ ] 6. Feature Importance — "Por que" o modelo decide assim?
[ ] 7. Análise de erros — O que o modelo erra sistematicamente?
[ ] 8. Sanity checks — O modelo faz sentido intuitivamente?
```

### Lendo a Matriz de Confusão

```
                    PREDITO
                 Positivo  Negativo
REAL  Positivo     VP         FN     ← Recall    = VP / (VP + FN)
      Negativo     FP         VN
                   ↑
             Precision = VP / (VP + FP)
```

| Erro | O Que Significa | Quando é Mais Caro |
|---|---|---|
| **FP** (Falso Positivo) | Previu positivo, era negativo | Spam filter, alarme falso |
| **FN** (Falso Negativo) | Previu negativo, era positivo | Diagnóstico médico, fraude |

**No projeto NPS:** FN é mais caro — não detectar um Detrator significa perder o cliente sem poder agir.

### Conectando ML ao Negócio — O ROI do Modelo

```
Framework de Tradução:

  Recall do modelo
    → Quantos Detratores detectamos por mês?
    → Quantas ações preventivas (cupons, CS VIP) disparar?
    → Taxa de retenção esperada após a ação
    → Receita preservada (Retidos x LTV)
    → Receita - Custo da ação = ROI do Modelo
```

### Feature Importance — Explicabilidade

```python
# Para Random Forest (Gini Importance)
importances = modelo_rf.feature_importances_
indices     = np.argsort(importances)[::-1]
features    = X_train.columns

plt.figure(figsize=(10, 5))
plt.barh(range(len(indices)), importances[indices], color='#e74c3c')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Peso Decisório das Variáveis Preditivas")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

> **Por que isso importa?** Permite ao negócio entender **onde atuar** — não apenas o que o modelo decidiu. Um modelo sem explicabilidade não é implantável em decisões críticas.

### 📝 Minhas Anotações — Avaliação

```
PERFORMANCE NO CONJUNTO DE TESTE:
  - F1-Macro:
  - Accuracy:
  - Precision:
  - Recall:

COMPARAÇÃO COM BASELINE:
  - Baseline:
  - Modelo atual:
  - Ganho real:

META ANALÍTICA (definida na Fase 1):
  - Meta era:
  - Atingido? Sim/Não

META DE NEGÓCIO (definida na Fase 1):
  - Meta era:
  - Alcançável com essa performance? Sim/Não
  - Estimativa de ROI:

TOP 5 FEATURES MAIS IMPORTANTES:
  1.
  2.
  3.
  4.
  5.

ANÁLISE DE ERROS SISTEMÁTICOS:
  - O modelo erra mais em qual grupo?
  - Hipótese para explicar:
```

---

## FASE 6 — 🚀 Deployment (Implantação)

> *"Um modelo que ninguém usa não tem valor. Modelos são feitos para gerar decisões, não papers."*

### O que fazer nessa fase

Tornar o modelo acessível e utilizável pelo público-alvo.

### Checklist de Deployment

```
[ ] 1. Salvar o modelo (pickle / joblib)
[ ] 2. Criar interface de uso (API, Streamlit, Dashboard)
[ ] 3. Documentar como reproduzir o modelo
[ ] 4. Testar o modelo em dados novos (pós-treino)
[ ] 5. Planejar monitoramento de drift (modelo envelhece!)
[ ] 6. Definir critério de re-treinamento
```

### Salvando e Carregando o Modelo

```python
import joblib

# Salvar — inclui o Pipeline completo (Scaler + Modelo)
joblib.dump(pipeline, 'models/pipeline_completo.pkl')

# Carregar em produção
pipeline_prod = joblib.load('models/pipeline_completo.pkl')
predicao = pipeline_prod.predict(novos_dados)
```

### Streamlit — Deploy Rápido

```python
import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load('models/pipeline_completo.pkl')

st.title("Preditor de NPS")

# Inputs do usuário
atraso       = st.sidebar.slider("Dias de Atraso", 0, 10)
contatos_sac = st.sidebar.number_input("Contatos com SAC", 0, 10)

if st.button("Analisar"):
    dados    = pd.DataFrame({'atraso': [atraso], 'sac': [contatos_sac]})
    resultado = pipeline.predict(dados)
    st.success(f"Classificação: {resultado[0]}")
```

### Estrutura de Repositório Recomendada

```
meu-projeto-ds/
│
├── data/
│   ├── raw/              # Dados originais — NUNCA modificar
│   └── processed/        # Dados prontos para modelagem
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preparacao.ipynb
│   └── 03_modelagem.ipynb
│
├── models/
│   └── pipeline.pkl      # Modelo serializado
│
├── app/
│   └── main.py           # Streamlit ou API
│
├── reports/
│   └── dashboard.png     # Visualizações exportadas
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Model Drift — O Modelo Envelhece

```
O que é Drift?
  → Os dados em produção mudam com o tempo
  → O modelo foi treinado em dados históricos
  → Performance degradada = Drift ocorreu

Sinais de Alerta:
  - Distribuição das features mudou significativamente
  - Taxa de predição de cada classe mudou muito
  - Performance real caiu abaixo do threshold aceitável

Exemplo real: Modelo de crédito treinado pré-pandemia
  → Comportamento de pagamento mudou radicalmente em 2020
  → Modelo ficou obsoleto em semanas
```

### 📝 Minhas Anotações — Deployment

```
FORMA DE ENTREGA:
  [ ] Streamlit App
  [ ] API REST (FastAPI/Flask)
  [ ] Dashboard (Power BI / Tableau)
  [ ] Relatório estático

MODELO SALVO:
  - Arquivo:
  - Inclui Pipeline completo (Scaler + Modelo)? Sim/Não

LINK DO APP (se disponível):

PLANO DE MONITORAMENTO:
  - Frequência de avaliação:
  - Critério de re-treino:
```

---

## 🔄 O Ciclo Completo — Resumo

```
  NEGÓCIO      DADOS       FEATURES      MODELO      DECISÃO
  Qual é       O que       Como criar    Qual        Como
  a dor?   →  temos?    → valor a    →  algoritmo ? → entregar
                           partir          ↓          o insight?
  ↑            ↑           disso?        Avaliar
  └────────────┴───────────────────────── e iterar ──┘
```

---

## 🧪 Framework de Qualidade do Projeto

> Use este checklist no final para avaliar a maturidade do projeto:

### Nível 1 — Básico (entregável de curso)
```
[ ] O modelo foi treinado e avaliado
[ ] A métrica principal foi escolhida com justificativa
[ ] Há separação treino/teste
```

### Nível 2 — Intermediário (portfólio profissional)
```
[ ] EDA completa com hipóteses testadas estatisticamente
[ ] Feature Engineering com intuição de negócio
[ ] Cross-Validation aplicado (cv=5 mínimo)
[ ] Comparação de múltiplos algoritmos
[ ] Ausência de Data Leakage verificada
[ ] Resultados conectados ao negócio (além de métricas)
```

### Nível 3 — Avançado (projeto real de empresa)
```
[ ] ROI do modelo calculado
[ ] Explicabilidade (SHAP ou Feature Importance)
[ ] Deploy funcional (API ou App)
[ ] Análise de sensibilidade das premissas de negócio
[ ] Documentação completa e reprodutível
[ ] Monitoramento de drift planejado
[ ] Pipeline Sklearn (elimina Training-Serving Skew)
```

---

## 📚 Glossário Rápido

| Termo | Definição |
|---|---|
| **EDA** | Exploratory Data Analysis — análise exploratória dos dados |
| **Feature Engineering** | Criação de novas variáveis a partir das existentes |
| **Data Leakage** | Usar dados futuros inadvertidamente no treino |
| **Overfitting** | Modelo memoriza o treino e não generaliza |
| **Underfitting** | Modelo é simples demais para capturar padrões |
| **Cross-Validation** | Validação com múltiplas divisões treino/teste |
| **Baseline** | Modelo mais simples possível — referência de comparação |
| **Threshold** | Corte de probabilidade para classificação |
| **Model Drift** | Degradação do modelo ao longo do tempo em produção |
| **Pipeline** | Encapsula pré-processamento + modelo em uma unidade |
| **Precision** | Dos que o modelo disse "sim", quantos realmente eram? |
| **Recall** | Dos que realmente eram "sim", quantos o modelo acertou? |
| **F1-Score** | Média harmônica entre Precision e Recall |
| **ROC-AUC** | Capacidade do modelo de rankear positivos acima de negativos |
| **class_weight='balanced'** | Penaliza erros em classes minoritárias proporcionalmente |
| **LTV** | Lifetime Value — valor total que um cliente gera para a empresa |
| **Churn** | Abandono do cliente |
| **Training-Serving Skew** | Diferença entre como o modelo foi treinado e como opera em prod |

---

## 🎯 Resumo do Projeto NPS como Referência

| Fase CRISP-DM | O Que Foi Feito |
|---|---|
| **Business Understanding** | NPS 4.38 = crise sistêmica; meta de reduzir Detratores 15% |
| **Data Understanding** | EDA em 2.500 pedidos; Testes T e ANOVA; Identificação de Leakage |
| **Data Preparation** | 7 features criadas; remoção de Leakage; Pipeline Sklearn |
| **Modeling** | 4 algoritmos comparados; Random Forest selecionado; threshold otimizado |
| **Evaluation** | F1-Macro 0.56; Recall 78% Detratores; ROI 308% calculado |
| **Deployment** | Streamlit App: Predição + Simulador ROI + Feature Importance |

---

*Roadmap baseado no projeto Tech Challenge Fase 1 — FIAP AI Scientist*
*Versão 1.0 | Abril 2026*
