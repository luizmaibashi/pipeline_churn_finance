# =============================================================
# shap_analysis_v2.py — Explicabilidade do Modelo v2 (Early-Warning)
# ADR-0001: por que o sinal comportamental pesa mais que o saldo
#
# Pré-requisito: pipeline.py já foi executado (gera gb_pipeline_v2.pkl)
# Uso: python shap_analysis_v2.py
# Output: output/shap/v2/  (plots + CSVs + relatório por cliente)
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("output/shap/v2", exist_ok=True)

print("=" * 60)
print("SHAP ANALYSIS v2 — Early-Warning Comportamental (ADR-0001)")
print("=" * 60)

# ── 1. Carrega artefatos gerados pelo pipeline.py ────────────
print("\n[1/5] Carregando pipeline v2 e dados...")

# joblib.load: artefato local gerado por pipeline.py neste mesmo repo,
# nunca de fonte externa/untrusted — mesmo padrão do shap_analysis.py original
pipeline = joblib.load("output/models/gb_pipeline_v2.pkl")
df = pd.read_csv("output/data/base_clientes_v2_limpo.csv")

FEATURES_V2_BASE = [
    "segmento", "meses_cliente", "qtd_produtos",
    "retorno_12m_pct", "freq_contato_mes", "saldo_bi",
    "dias_desde_ultimo_contato", "variacao_freq_contato_3m",
    "tempo_resposta_medio_horas",
    "sem_historico_12m", "cliente_novo_sem_contato_hist",
]
TARGET = "churn"

X = df[FEATURES_V2_BASE]
y = df[TARGET]

print(f"  Clientes carregados: {len(df)} | Churn: {y.sum()} ({y.mean()*100:.1f}%)")

# ── 2. Extrai dados transformados (o SHAP precisa do X numérico)
print("\n[2/5] Aplicando transformações do pipeline para extração SHAP...")

preprocessing_pipeline = pipeline[:-1]   # tudo menos o clf
X_transformed = preprocessing_pipeline.transform(X)

FEATURE_NAMES = ["segmento_enc"] + [
    "meses_cliente", "qtd_produtos", "retorno_12m_pct",
    "freq_contato_mes", "saldo_bi", "engajamento_score",
    "retorno_relativo", "flag_risco", "intensidade_rel",
    "dias_desde_ultimo_contato", "variacao_freq_contato_3m",
    "tempo_resposta_medio_horas",
    "sem_historico_12m", "cliente_novo_sem_contato_hist",
]

# ── 3. Calcula SHAP Values ───────────────────────────────────
print("\n[3/5] Calculando SHAP Values (TreeExplainer)...")

clf = pipeline.named_steps["clf"]
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_transformed)

print(f"  SHAP Values calculados: {shap_values.shape}")

df_shap = pd.DataFrame(shap_values, columns=FEATURE_NAMES)
df_shap.insert(0, "cliente_id", df["cliente_id"].values)
df_shap.insert(1, "churn_real", y.values)
df_shap.insert(2, "churn_prob", pipeline.predict_proba(X)[:, 1].round(4))
df_shap.to_csv("output/shap/v2/shap_values.csv", index=False)
print("  Salvo: output/shap/v2/shap_values.csv")

# ── 4. Gráficos de Explicabilidade ──────────────────────────
print("\n[4/5] Gerando visualizações...")

PLOT_STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#0f1117",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#333",
}
plt.rcParams.update(PLOT_STYLE)

print("  [4a] Summary plot...")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(
    shap_values, X_transformed,
    feature_names=FEATURE_NAMES,
    show=False, plot_type="dot", color_bar=True
)
plt.title("SHAP Summary v2 — Sinal Comportamental vs. Saldo no Churn",
          fontsize=13, color="#e0e0e0", pad=12)
plt.tight_layout()
plt.savefig("output/shap/v2/summary_plot.png", dpi=150,
            facecolor="#0f1117", bbox_inches="tight")
plt.close()
print("  Salvo: output/shap/v2/summary_plot.png")

print("  [4b] Bar importance plot...")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(
    importance_df["feature"],
    importance_df["mean_abs_shap"],
    color="#7c3aed", edgecolor="#4c1d95", height=0.6
)
for bar, val in zip(bars, importance_df["mean_abs_shap"]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left",
            fontsize=9, color="#e0e0e0")

ax.set_xlabel("Impacto Médio Absoluto (SHAP)", fontsize=11)
ax.set_title("Importância das Features v2 — |SHAP| Médio Global",
             fontsize=13, color="#e0e0e0", pad=12)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("output/shap/v2/feature_importance_shap.png", dpi=150,
            facecolor="#0f1117", bbox_inches="tight")
plt.close()
importance_df.sort_values("mean_abs_shap", ascending=False).to_csv(
    "output/shap/v2/feature_importance_shap.csv", index=False
)
print("  Salvo: output/shap/v2/feature_importance_shap.png")
print("  Salvo: output/shap/v2/feature_importance_shap.csv")

print("  [4c] Dependence plot (variacao_freq_contato_3m — feature mais forte)...")
feat_idx = FEATURE_NAMES.index("variacao_freq_contato_3m")
fig, ax = plt.subplots(figsize=(9, 5))
scatter = ax.scatter(
    X_transformed[:, feat_idx],
    shap_values[:, feat_idx],
    c=X_transformed[:, feat_idx],
    cmap="RdYlGn_r", alpha=0.7, s=15
)
ax.axhline(0, color="#666", linewidth=0.8, linestyle="--")
ax.set_xlabel("Variação da frequência de contato (3m) — negativo = esfriando", fontsize=11)
ax.set_ylabel("SHAP Value (impacto no risco de churn)", fontsize=11)
ax.set_title("Dependência SHAP: Esfriamento de Contato vs Risco de Churn",
             fontsize=13, color="#e0e0e0", pad=12)
plt.colorbar(scatter, ax=ax, label="Variação de cadência")
plt.tight_layout()
plt.savefig("output/shap/v2/dependence_variacao_contato.png", dpi=150,
            facecolor="#0f1117", bbox_inches="tight")
plt.close()
print("  Salvo: output/shap/v2/dependence_variacao_contato.png")

# ── 5. Relatório por Cliente (linguagem natural — LGPD ready) ─
print("\n[5/5] Gerando relatório de explicações por cliente...")

def top3_razoes(row_shap: np.ndarray, feature_names: list, prob: float) -> str:
    TRADUCAO = {
        "retorno_12m_pct":            "Retorno da carteira nos últimos 12 meses",
        "freq_contato_mes":           "Frequência de contato com assessor (meses)",
        "retorno_relativo":           "Retorno relativo ao benchmark de mercado",
        "engajamento_score":          "Score de engajamento do cliente",
        "saldo_bi":                   "Saldo total na custódia",
        "qtd_produtos":               "Quantidade de produtos contratados",
        "meses_cliente":              "Tempo como cliente (meses)",
        "flag_risco":                 "Flag de risco comportamental",
        "intensidade_rel":            "Intensidade relativa de movimentação",
        "segmento_enc":               "Segmento do cliente",
        "dias_desde_ultimo_contato":  "Dias desde o último contato com o assessor",
        "variacao_freq_contato_3m":   "Variação da cadência de contato (últimos 3 meses)",
        "tempo_resposta_medio_horas": "Tempo médio de resposta do cliente ao assessor",
        "sem_historico_12m":          "Cliente sem 12 meses de histórico de retorno",
        "cliente_novo_sem_contato_hist": "Cliente novo, sem histórico de contato",
    }
    sorted_idx = np.argsort(np.abs(row_shap))[::-1]
    razoes = []
    for i in sorted_idx[:3]:
        feat  = feature_names[i]
        val   = row_shap[i]
        sinal = "[+] AUMENTA risco" if val > 0 else "[-] REDUZ risco"
        nome  = TRADUCAO.get(feat, feat)
        razoes.append(f"  • {nome}: {sinal} (SHAP={val:+.4f})")

    nivel = "[ALTO]" if prob >= 0.6 else ("[MEDIO]" if prob >= 0.35 else "[BAIXO]")
    return f"Risco de Churn: {nivel} ({prob*100:.1f}%)\n" + "\n".join(razoes)


relatorio = []
for i in range(len(df)):
    prob = df_shap.iloc[i]["churn_prob"]
    if prob < 0.35:
        continue
    cli_id   = df_shap.iloc[i]["cliente_id"]
    churn_r  = df_shap.iloc[i]["churn_real"]
    segmento = df.iloc[i]["segmento"]
    explicacao = top3_razoes(shap_values[i], FEATURE_NAMES, prob)

    relatorio.append({
        "cliente_id": cli_id,
        "segmento":   segmento,
        "churn_prob": prob,
        "churn_real": churn_r,
        "explicacao": explicacao
    })

df_relatorio = pd.DataFrame(relatorio).sort_values("churn_prob", ascending=False)
df_relatorio.to_csv("output/shap/v2/client_explanations.csv", index=False)

print(f"\n  Clientes de risco médio/alto identificados: {len(df_relatorio)}")
print("\n  === TOP 5 CLIENTES EM RISCO (v2) ===")
for _, row in df_relatorio.head(5).iterrows():
    print(f"\n  Cliente: {row['cliente_id']} | Segmento: {row['segmento']}")
    print(f"  {row['explicacao']}")
    print("  " + "-" * 50)

print("\nSalvo: output/shap/v2/client_explanations.csv")

print("\n" + "=" * 60)
print("[OK] SHAP Analysis v2 concluído!")
print("  Artefatos gerados em output/shap/v2/")
print("=" * 60)
