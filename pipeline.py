# =============================================================
# pipeline.py — Geração standalone dos artefatos de ML
# Executa o pipeline completo e salva os modelos em output/models/
# Uso: python pipeline.py
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, f1_score
)

# Importa o Custom Transformer para evitar Data Leakage
from transformers import FeatureEngineer

# ── Pastas de saída ──────────────────────────────────────────
for folder in ["output/data", "output/models", "output/charts"]:
    os.makedirs(folder, exist_ok=True)

print("=" * 60)
print("PIPELINE CHURN FINANCE — Gerando artefatos de ML")
print("=" * 60)

# ── FASE 1: Geração de dados sintéticos ─────────────────────
print("\n[1/6] Gerando dados sintéticos...")
np.random.seed(42)
N = 1200

segmentos = ["Varejo", "Alta Renda", "Wealth", "Corporate"]
seg_prob = [0.65, 0.24, 0.08, 0.03]
seg = np.random.choice(segmentos, N, p=seg_prob)

meses_cli = np.random.randint(1, 144, N)
qtd_prod = np.random.randint(1, 9, N)
retorno = np.random.normal(11.5, 4.2, N).round(2)
freq_cont = np.random.poisson(2.8, N)
saldo = np.random.lognormal(-1.8, 1.3, N).round(4)

taxa_base = {"Varejo": 0.18, "Alta Renda": 0.09, "Wealth": 0.05, "Corporate": 0.04}

churn = np.zeros(N, dtype=int)
for s in segmentos:
    idx = np.where(seg == s)[0]
    for i in idx:
        mod = 1.0
        if retorno[i] < 8.0:    mod *= 1.50
        if freq_cont[i] == 0:   mod *= 1.70
        if qtd_prod[i] == 1:    mod *= 1.25
        if meses_cli[i] < 12:   mod *= 1.35
        if saldo[i] < 0.1:      mod *= 1.40
        p = min(taxa_base[s] * mod, 0.75)
        churn[i] = int(np.random.rand() < p)

df = pd.DataFrame({
    "cliente_id"      : [f"CLI{str(i).zfill(5)}" for i in range(N)],
    "segmento"        : seg,
    "meses_cliente"   : meses_cli,
    "qtd_produtos"    : qtd_prod,
    "retorno_12m_pct" : retorno,
    "freq_contato_mes": freq_cont,
    "saldo_bi"        : saldo,
    "churn"           : churn
})
df.to_csv("output/data/base_clientes.csv", index=False)
vc = df["churn"].value_counts()
print(f"  Shape: {df.shape} | Churn: {vc[1]} ({vc[1]/N*100:.1f}%) | Não-Churn: {vc[0]} ({vc[0]/N*100:.1f}%)")

# ── FASE 2: Split estratificado ANTES da Engenharia de Features
print("\n[2/6] Split estratificado...")
TARGET = "churn"
FEATURES_BASE = [
    "segmento", "meses_cliente", "qtd_produtos", 
    "retorno_12m_pct", "freq_contato_mes", "saldo_bi"
]

X = df[FEATURES_BASE]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]} | Churn no teste: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

# ── FASE 3: Feature Engineering sem Data Leakage ─────────────
print("\n[3/6] Engenharia de Features (Aprendendo apenas do treino)...")
fe = FeatureEngineer()
# Ajusta apenas nos dados de TREINO
fe.fit(X_train)

# Aplicando para gerar o CSV base usado nos insights (mas o modelo usará Pipeline dinâmico)
X_fe_all = fe.transform(df)

# Codificando segmentos apenas para o output CSV que os dashboards dependem
encoder = OrdinalEncoder(categories=[["Varejo", "Alta Renda", "Wealth", "Corporate"]])
encoder.fit(X_train[["segmento"]])
X_fe_all["segmento_enc"] = encoder.transform(X_fe_all[["segmento"]])

df_fe = pd.concat([df["cliente_id"], X_fe_all, df[["churn"]]], axis=1)
df_fe.to_csv("output/data/base_feature_eng.csv", index=False)
print(f"  5 novas features criadas. CSV para análise salvo em output/data/base_feature_eng.csv.")

# Para enviar dados base para App (Training-Serving Skew mitigation)
model_params = {
    "media_retorno": float(fe.media_retorno_),
    "categories": [c.tolist() for c in encoder.categories_]
}
with open("output/models/fe_params.json", "w") as f:
    json.dump(model_params, f)

# ── DEFINIÇÃO DO PIPELINE PADRÃO
FEATURES_AFTER_FE = [
    "meses_cliente", "qtd_produtos", "retorno_12m_pct",
    "freq_contato_mes", "saldo_bi", "engajamento_score", 
    "retorno_relativo", "flag_risco", "intensidade_rel"
]

preprocessing = ColumnTransformer(
    transformers=[
        ("ordinals", encoder, ["segmento"]),
        ("pass", "passthrough", FEATURES_AFTER_FE)
    ]
)

def create_pipeline(classifier, scale=False):
    steps = [
        ("fe", FeatureEngineer()),
        ("prep", preprocessing)
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", classifier))
    return Pipeline(steps)

# ── FASE 4: Benchmark de modelos ─────────────────────────────
print("\n[4/6] Benchmark de algoritmos...")
modelos_bench = {
    "Dummy (baseline)"   : create_pipeline(DummyClassifier(strategy="stratified", random_state=42)),
    "Logistic Regression": create_pipeline(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42), scale=True),
    "Decision Tree"      : create_pipeline(DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)),
    "Random Forest"      : create_pipeline(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
    "Gradient Boosting"  : create_pipeline(GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, random_state=42))
}

benchmark_results = []
print(f"  {'Modelo':<26} {'F1-macro':>9} {'F1-churn':>9} {'ROC-AUC':>9}")
print("  " + "-" * 55)

for nome, pipeline in modelos_bench.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    f1_mac  = f1_score(y_test, y_pred, average="macro")
    f1_chur = f1_score(y_test, y_pred, pos_label=1, average="binary")
    roc     = roc_auc_score(y_test, y_prob)
    acc     = (y_pred == y_test).mean()

    benchmark_results.append({
        "modelo": nome, "acuracia": acc,
        "f1_macro": f1_mac, "f1_churn": f1_chur, "roc_auc": roc
    })
    print(f"  {nome:<26} {f1_mac:>9.4f} {f1_chur:>9.4f} {roc:>9.4f}")

pd.DataFrame(benchmark_results).to_csv("output/data/benchmark_results.csv", index=False)

# ── FASE 5: Cross-Validation + Modelo Final ───────────────────
print("\n[5/6] Cross-Validation + Modelo final...")
gb_final = create_pipeline(GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, random_state=42))

gb_final.fit(X_train, y_train)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(gb_final, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)

y_pred_final = gb_final.predict(X_test)
y_prob_final = gb_final.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm.ravel()

f1_final  = f1_score(y_test, y_pred_final, average="macro")
roc_final = roc_auc_score(y_test, y_prob_final)

print(f"  CV F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Teste — F1-macro: {f1_final:.4f} | ROC-AUC: {roc_final:.4f}")
print(f"  Confusão — TN={tn} FP={fp} FN={fn} TP={tp}")

# Feature Importance extraído do Classifier após transformação
gb_clf = gb_final.named_steps["clf"]
importances = pd.DataFrame({
    "feature"   : ["segmento_enc"] + FEATURES_AFTER_FE,
    "importance": gb_clf.feature_importances_
}).sort_values("importance", ascending=False)
importances.to_csv("output/data/feature_importance.csv", index=False)

# CV scores
pd.DataFrame({
    "fold"    : [f"Fold {i+1}" for i in range(5)],
    "f1_macro": cv_scores.round(4)
}).to_csv("output/data/cv_scores.csv", index=False)

# Confusion matrix
pd.DataFrame({
    "tn": [tn], "fp": [fp], "fn": [fn], "tp": [tp]
}).to_csv("output/data/confusion_matrix.csv", index=False)

# ── FASE 6: Persistência dos artefatos ───────────────────────
print("\n[6/6] Salvando pipeline consolidado...")
# Salvamos SOMENTE o pipeline, que já engloba feature engineer e encoder!
joblib.dump(gb_final, "output/models/gb_pipeline.pkl")
print("  [OK] output/models/gb_pipeline.pkl")
print("  [OK] output/models/fe_params.json")

print("\n" + "=" * 60)
print("[OK] Pipeline concluído com sucesso!")
print(f"  Modelo: Gradient Boosting (dentro do Pipeline) | F1-macro: {f1_final:.4f} | ROC-AUC: {roc_final:.4f}")
print(f"  Meta analítica: F1-macro >= 0.55 | ROC-AUC >= 0.70")
meta_f1  = "[ATINGIDA]"     if f1_final  >= 0.55 else "[NÃO atingida]"
meta_roc = "[ATINGIDA]"     if roc_final >= 0.70 else "[NÃO atingida]"
print(f"  F1-macro {meta_f1} | ROC-AUC {meta_roc}")
print("=" * 60)
print("\nPróximo passo: streamlit run app.py")
