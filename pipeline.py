# =============================================================
# pipeline.py — Geração modular dos artefatos de ML (Kedro Pattern)
# Executa o pipeline completo através de nodes puros e do Data Catalog
# Uso: python pipeline.py
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Garante que o diretório raiz e o diretório 'src' estão no path de importação
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.kedro_runner import DataCatalog, load_parameters
from src.data_processing.nodes import (
    generate_synthetic_data,
    generate_advisors_data,
    attach_advisor_and_behavioral_features,
    inject_data_quality_issues,
    clean_clientes_v2_bruto,
    aggregate_carteira_exposta_por_assessor,
    split_data,
    run_feature_engineering
)
from src.model_training.nodes import (
    benchmark_algorithms,
    train_final_model,
    evaluate_final_model,
    cross_validate,
    get_feature_importance,
    train_and_compare_v1_v2,
    bootstrap_ic_diferenca_recall,
    cross_validate_v2,
    train_final_model_v2,
    get_feature_importance_v2,
    calibrate_thresholds_v2,
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
    parameters=parameters,
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
print(f"  AuC total: R$ {df_v2['auc_milhoes'].sum() / 1000:.1f} bi | mediana: R$ {df_v2['auc_milhoes'].median():.1f} M")
print(f"  AuC exposto (v2): R$ {df_v2['auc_exposto'].sum() / 1000:.1f} bi ({df_v2['auc_exposto'].sum()/df_v2['auc_milhoes'].sum()*100:.1f}% do total)")
print(f"  Assessores sem book: {(df_advisors['qtd_clientes_carteira'] == 0).sum()}")

# ── FASE 1.6: Injeção de sujeira de dado realista (Etapa 1, aprovada) ─
print("\n[1.6/6] Injetando problemas de qualidade de dado (dataset bruto)...")
df_v2_bruto, df_advisors_bruto = inject_data_quality_issues(
    df_v2, df_advisors, seed=parameters.get("random_state", 42)
)
catalog.save("base_clientes_v2_bruto", df_v2_bruto)
catalog.save("base_assessores_bruto", df_advisors_bruto)
print(f"  Shape bruto: {df_v2_bruto.shape} (v2 limpo: {df_v2.shape}) — {df_v2_bruto.shape[0] - df_v2.shape[0]} linha(s) duplicada(s)")
print(f"  Nulos introduzidos: {df_v2_bruto.isna().sum().sum()} células | dataset limpo permanece em base_clientes_v2 (não sobrescrito)")

# Direção B como produto de dado separado (correção pós-auditoria: auc_exposto
# não é feature de churn, é agregação de risco de carteira por assessor)
carteira_exposta = aggregate_carteira_exposta_por_assessor(df_v2, df_advisors)
catalog.save("carteira_exposta_por_assessor", carteira_exposta)
top3 = carteira_exposta.head(3)
print(f"  Carteira exposta (Direção B, dashboard próprio) — top 3 assessores de maior risco:")
for _, row in top3.iterrows():
    print(f"    {row['assessor_id']} ({row['canal']}) — AuC exposto: R$ {row['auc_exposto_total']:.2f} M ({row['pct_carteira_exposta']*100:.0f}% da carteira)")

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

# ── FASE 5.5: Comparação v1 (reativa) vs v2 (early-warning + advisor) ─
print("\n[5.5/6] Comparando modelo v2 (ADR-0001) contra baseline v1...")
df_v2_limpo = clean_clientes_v2_bruto(df_v2_bruto, df_advisors_bruto)
catalog.save("base_clientes_v2_limpo", df_v2_limpo)

comparacao, y_test_comum, y_pred_v1, y_pred_v2 = train_and_compare_v1_v2(df, df_v2_limpo, parameters)
catalog.save("recall_early_warning_vs_baseline_reativo", comparacao)

print(f"  {'Modelo':<28} {'Recall(churn)':>13} {'F1-churn':>9} {'ROC-AUC':>9}")
print("  " + "-" * 62)
for _, row in comparacao.iterrows():
    print(f"  {row['modelo']:<28} {row['recall_churn']:>13.4f} {row['f1_churn']:>9.4f} {row['roc_auc']:>9.4f}")
recall_v1 = comparacao.loc[comparacao['modelo']=='v1_baseline_reativa', 'recall_churn'].iloc[0]
recall_v2 = comparacao.loc[comparacao['modelo']=='v2_early_warning_advisor', 'recall_churn'].iloc[0]
n_churn_teste = int(y_test_comum.sum())
print(f"  n_teste=240 | n_churn_teste={n_churn_teste} (gate: proporcao sempre com n)")

# GATE ML: proporcao sempre com n e IC -- bootstrap sobre as predicoes
ic = bootstrap_ic_diferenca_recall(y_test_comum, y_pred_v1, y_pred_v2, n_bootstrap=1000, seed=parameters.get("random_state", 42))
print(f"  Diferenca de recall (v2-v1): {ic['diferenca_media']:+.4f} | IC95% [{ic['ic95_lo']:+.4f}, {ic['ic95_hi']:+.4f}]")
veredito_ic = "[IC exclui zero -- diferenca real]" if ic["ic_exclui_zero"] else "[IC INCLUI ZERO -- pode ser ruido amostral]"
print(f"  {veredito_ic}")

# Cross-validation na v2 (single split de 240 e instavel demais sozinho)
cv_v2_scores, cv_v2_mean, cv_v2_std = cross_validate_v2(df_v2_limpo, parameters)
catalog.save("cv_scores_v2", cv_v2_scores)
print(f"  CV 5-fold v2 recall: {cv_v2_mean:.4f} +/- {cv_v2_std:.4f}")

veredito = "[SUPEROU]" if recall_v2 > recall_v1 else "[NAO superou]"
print(f"  Criterio ADR-0001 (ponto): v2 recall > v1 recall? {veredito} ({recall_v2:.4f} vs {recall_v1:.4f})")

# Persiste o modelo v2 final (treinado no dataset limpo inteiro) para uso
# em shap_analysis_v2.py / API futura
gb_final_v2 = train_final_model_v2(df_v2_limpo, parameters)
catalog.save("gb_pipeline_v2", gb_final_v2)
importances_v2 = get_feature_importance_v2(gb_final_v2)
catalog.save("feature_importance_v2", importances_v2)

# Thresholds são selecionados em validação interna, nunca no conjunto de teste.
cal_train, cal_valid = train_test_split(
    df_v2_limpo, test_size=0.25, random_state=parameters.get("random_state", 42),
    stratify=df_v2_limpo["churn"],
)
modelo_calibracao = train_final_model_v2(cal_train, parameters)
thresholds_v2 = calibrate_thresholds_v2(modelo_calibracao, cal_valid, parameters)
catalog.save("thresholds_v2", thresholds_v2)
Path("reports").mkdir(exist_ok=True)
with open("reports/thresholds_v2.md", "w", encoding="utf-8") as report:
    report.write("# Thresholds v2\n\n")
    report.write("Seleção por custo `10 × FN + FP`, com recall mínimo de 0,75. ")
    report.write("Segmentos com menos de 5 positivos na validação usam o threshold global.\n\n")
    report.write(thresholds_v2.to_markdown(index=False))
    report.write("\n")
print("  Thresholds v2 calibrados em validação interna e salvos em output/data/thresholds_v2.csv")
print(f"  Importância comportamental v2: {importances_v2[importances_v2['feature'].isin(['dias_desde_ultimo_contato', 'variacao_freq_contato_3m', 'tempo_resposta_medio_horas'])]['importance'].sum():.1%}")

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
