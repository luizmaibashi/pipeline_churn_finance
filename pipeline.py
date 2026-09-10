# =============================================================
# pipeline.py — Geração modular dos artefatos de ML (Kedro Pattern)
# Executa o pipeline completo através de nodes puros e do Data Catalog
# Uso: python pipeline.py
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import sys

# Garante que o diretório raiz e o diretório 'src' estão no path de importação
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.kedro_runner import DataCatalog, load_parameters
from src.data_processing.nodes import (
    generate_synthetic_data,
    generate_advisors_data,
    attach_advisor_and_behavioral_features,
    inject_data_quality_issues,
    split_data,
    run_feature_engineering
)
from src.model_training.nodes import (
    benchmark_algorithms,
    train_final_model,
    evaluate_final_model,
    cross_validate,
    get_feature_importance
)

print("=" * 60)
print("PIPELINE CHURN FINANCE — Gerando artefatos de ML (Kedro Pattern)")
print("=" * 60)

# Inicializa o Data Catalog e carrega parâmetros
catalog = DataCatalog("conf/base/catalog.yml")
parameters = load_parameters("conf/base/parameters.yml")

# ── FASE 1: Geração de dados sintéticos ─────────────────────
print("\n[1/6] Gerando dados sintéticos...")
df = generate_synthetic_data(
    n_samples=parameters.get("n_samples", 1200),
    seed=parameters.get("random_state", 42)
)
catalog.save("base_clientes", df)

vc = df["churn"].value_counts()
print(f"  Shape: {df.shape} | Churn: {vc[1]} ({vc[1]/len(df)*100:.1f}%) | Não-Churn: {vc[0]} ({vc[0]/len(df)*100:.1f}%)")

# ── FASE 1.5: Assessores + Features Comportamentais (ADR-0001) ─
print("\n[1.5/6] Gerando assessores e features de early-warning (Direção A+B)...")
df_advisors = generate_advisors_data(
    n_advisors=parameters.get("n_advisors", 300),
    seed=parameters.get("random_state", 42)
)

df_v2, df_advisors = attach_advisor_and_behavioral_features(df, df_advisors, seed=parameters.get("random_state", 42))
catalog.save("base_assessores", df_advisors)
catalog.save("base_clientes_v2", df_v2)

vc_adv = df_advisors["risco_saida"].value_counts()
print(f"  Assessores: {df_advisors.shape[0]} | Risco de saída: {vc_adv.get(1, 0)} ({vc_adv.get(1, 0)/len(df_advisors)*100:.1f}%)")
print(f"  AuC exposto (v2): R$ {df_v2['auc_exposto'].sum():.2f}bi de R$ {df_v2['saldo_bi'].sum():.2f}bi total ({df_v2['auc_exposto'].sum()/df_v2['saldo_bi'].sum()*100:.1f}%)")

# ── FASE 1.6: Injeção de sujeira de dado realista (Etapa 1, aprovada) ─
print("\n[1.6/6] Injetando problemas de qualidade de dado (dataset bruto)...")
df_v2_bruto, df_advisors_bruto = inject_data_quality_issues(
    df_v2, df_advisors, seed=parameters.get("random_state", 42)
)
catalog.save("base_clientes_v2_bruto", df_v2_bruto)
catalog.save("base_assessores_bruto", df_advisors_bruto)
print(f"  Shape bruto: {df_v2_bruto.shape} (v2 limpo: {df_v2.shape}) — {df_v2_bruto.shape[0] - df_v2.shape[0]} linha(s) duplicada(s)")
print(f"  Nulos introduzidos: {df_v2_bruto.isna().sum().sum()} células | dataset limpo permanece em base_clientes_v2 (não sobrescrito)")

# ── FASE 2: Split estratificado ANTES da Engenharia de Features
print("\n[2/6] Split estratificado...")
train_df, test_df = split_data(
    df,
    test_size=parameters.get("test_size", 0.20),
    random_state=parameters.get("random_state", 42)
)
print(f"  Treino: {train_df.shape[0]} | Teste: {test_df.shape[0]} | Churn no teste: {test_df['churn'].sum()} ({test_df['churn'].mean()*100:.1f}%)")

# ── FASE 3: Feature Engineering sem Data Leakage ─────────────
print("\n[3/6] Engenharia de Features (Aprendendo apenas do treino)...")
df_fe, fe_params = run_feature_engineering(df, train_df)

catalog.save("base_feature_eng", df_fe)
catalog.save("fe_params", fe_params)
print(f"  5 novas features criadas. CSV para análise salvo em output/data/base_feature_eng.csv.")

# ── FASE 4: Benchmark de modelos ─────────────────────────────
print("\n[4/6] Benchmark de algoritmos...")
benchmark_results = benchmark_algorithms(train_df, test_df, parameters)

catalog.save("benchmark_results", benchmark_results)

print(f"  {'Modelo':<26} {'F1-macro':>9} {'F1-churn':>9} {'ROC-AUC':>9}")
print("  " + "-" * 55)
for _, row in benchmark_results.iterrows():
    print(f"  {row['modelo']:<26} {row['f1_macro']:>9.4f} {row['f1_churn']:>9.4f} {row['roc_auc']:>9.4f}")

# ── FASE 5: Cross-Validation + Modelo Final ───────────────────
print("\n[5/6] Cross-Validation + Modelo final...")
gb_final = train_final_model(train_df, parameters)

# Executa cross-validation
cv_scores, cv_mean, cv_std = cross_validate(gb_final, df, parameters)
catalog.save("cv_scores", cv_scores)

# Executa avaliação do modelo final no conjunto de teste
metrics, df_cm = evaluate_final_model(gb_final, test_df)
catalog.save("confusion_matrix", df_cm)

# Extrai e salva importâncias de feature
importances = get_feature_importance(gb_final)
catalog.save("feature_importance", importances)

print(f"  CV F1-macro: {cv_mean:.4f} ± {cv_std:.4f}")
print(f"  Teste — F1-macro: {metrics['f1_macro']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"  Confusão — TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}")

# ── FASE 6: Persistência dos artefatos ───────────────────────
print("\n[6/6] Salvando pipeline consolidado...")
catalog.save("gb_pipeline", gb_final)

print("\n" + "=" * 60)
print("[OK] Pipeline concluído com sucesso!")
print(f"  Modelo: Gradient Boosting (dentro do Pipeline) | F1-macro: {metrics['f1_macro']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"  Meta analítica: F1-macro >= 0.55 | ROC-AUC >= 0.70")
meta_f1  = "[ATINGIDA]"     if metrics['f1_macro']  >= 0.55 else "[NÃO atingida]"
meta_roc = "[ATINGIDA]"     if metrics['roc_auc'] >= 0.70 else "[NÃO atingida]"
print(f"  F1-macro {meta_f1} | ROC-AUC {meta_roc}")
print("=" * 60)
print("\nPróximo passo: streamlit run app.py")
