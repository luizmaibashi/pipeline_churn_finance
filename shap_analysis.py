# =============================================================
# shap_analysis.py — Explicabilidade do Modelo de Churn
# Fase 1 do Roadmap: Transparência (LGPD / Confiança Executiva)
#
# Pré-requisito: pipeline.py já foi executado
# Uso: python shap_analysis.py
# Output: output/shap/  (plots + CSVs + relatório por cliente)
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")           # renderiza sem abrir janela
import matplotlib.pyplot as plt

# ── Diretórios ───────────────────────────────────────────────
os.makedirs("output/shap", exist_ok=True)

print("=" * 60)
print("SHAP ANALYSIS — Transparência do Modelo de Churn")
print("=" * 60)

# ── 1. Carrega artefatos gerados pelo pipeline.py ────────────
print("\n[1/5] Carregando pipeline e dados...")

pipeline = joblib.load("output/models/gb_pipeline.pkl")
df = pd.read_csv("output/data/base_clientes.csv")

FEATURES_BASE = [
    "segmento", "meses_cliente", "qtd_produtos",
    "retorno_12m_pct", "freq_contato_mes", "saldo_bi"
]
TARGET = "churn"

X = df[FEATURES_BASE]
y = df[TARGET]

print(f"  Clientes carregados: {len(df)} | Churn: {y.sum()} ({y.mean()*100:.1f}%)")

# ── 2. Extrai dados transformados (o SHAP precisa do X numérico)
print("\n[2/5] Aplicando transformações do pipeline para extração SHAP...")

# Extrai os steps intermediários (Feature Engineer + Preprocessor)
# sem o classificador, para obter a matrix numérica
preprocessing_pipeline = pipeline[:-1]   # tudo menos o clf
X_transformed = preprocessing_pipeline.transform(X)

# Nomes das features após transformação
FEATURE_NAMES = ["segmento_enc"] + [
    "meses_cliente", "qtd_produtos", "retorno_12m_pct",
    "freq_contato_mes", "saldo_bi", "engajamento_score",
    "retorno_relativo", "flag_risco", "intensidade_rel"
]

# ── 3. Calcula SHAP Values ───────────────────────────────────
print("\n[3/5] Calculando SHAP Values (TreeExplainer)...")

clf = pipeline.named_steps["clf"]

# TreeExplainer: nativo para GradientBoosting → rápido e exato
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_transformed)   # shape (N, features)

print(f"  SHAP Values calculados: {shap_values.shape}")

# Salva os valores brutos para uso futuro (ex: Streamlit, FastAPI)
df_shap = pd.DataFrame(shap_values, columns=FEATURE_NAMES)
df_shap.insert(0, "cliente_id", df["cliente_id"].values)
df_shap.insert(1, "churn_real", y.values)
df_shap.insert(2, "churn_prob", pipeline.predict_proba(X)[:, 1].round(4))
df_shap.to_csv("output/shap/shap_values.csv", index=False)
print("  Salvo: output/shap/shap_values.csv")

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

# --- 4a. Summary Plot (impacto global de cada feature) ------
print("  [4a] Summary plot...")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(
    shap_values, X_transformed,
    feature_names=FEATURE_NAMES,
    show=False, plot_type="dot", color_bar=True
)
plt.title("SHAP Summary — Impacto Global das Features no Churn",
          fontsize=13, color="#e0e0e0", pad=12)
plt.tight_layout()
plt.savefig("output/shap/summary_plot.png", dpi=150,
            facecolor="#0f1117", bbox_inches="tight")
plt.close()
print("  Salvo: output/shap/summary_plot.png")

# --- 4b. Bar Plot (importância média |SHAP|) ----------------
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
# Adiciona valores nas barras
for bar, val in zip(bars, importance_df["mean_abs_shap"]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left",
            fontsize=9, color="#e0e0e0")

ax.set_xlabel("Impacto Médio Absoluto (SHAP)", fontsize=11)
ax.set_title("Importância das Features — |SHAP| Médio Global",
             fontsize=13, color="#e0e0e0", pad=12)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("output/shap/feature_importance_shap.png", dpi=150,
            facecolor="#0f1117", bbox_inches="tight")
plt.close()
importance_df.sort_values("mean_abs_shap", ascending=False).to_csv(
    "output/shap/feature_importance_shap.csv", index=False
)
print("  Salvo: output/shap/feature_importance_shap.png")
print("  Salvo: output/shap/feature_importance_shap.csv")

# --- 4c. Dependence Plot: retorno_12m_pct (feature crítica) -
print("  [4c] Dependence plot (retorno_12m_pct)...")
feat_idx = FEATURE_NAMES.index("retorno_12m_pct")
fig, ax = plt.subplots(figsize=(9, 5))
scatter = ax.scatter(
    X_transformed[:, feat_idx],
    shap_values[:, feat_idx],
    c=X_transformed[:, feat_idx],
    cmap="RdYlGn", alpha=0.7, s=15
)
ax.axhline(0, color="#666", linewidth=0.8, linestyle="--")
ax.set_xlabel("Retorno 12m (%)", fontsize=11)
ax.set_ylabel("SHAP Value (impacto no risco de churn)", fontsize=11)
ax.set_title("Dependência SHAP: Retorno 12m vs Risco de Churn",
             fontsize=13, color="#e0e0e0", pad=12)
plt.colorbar(scatter, ax=ax, label="Retorno 12m (%)")
plt.tight_layout()
plt.savefig("output/shap/dependence_retorno.png", dpi=150,
            facecolor="#0f1117", bbox_inches="tight")
plt.close()
print("  Salvo: output/shap/dependence_retorno.png")

# ── 5. Relatório por Cliente (linguagem natural — LGPD ready) ─
print("\n[5/5] Gerando relatório de explicações por cliente...")

def top3_razoes(row_shap: np.ndarray, feature_names: list,
                prob: float) -> str:
    """Retorna string com top-3 fatores de risco em linguagem natural."""

    TRADUCAO = {
        "retorno_12m_pct":   "Retorno da carteira nos últimos 12 meses",
        "freq_contato_mes":  "Frequência de contato com assessor (meses)",
        "retorno_relativo":  "Retorno relativo ao benchmark de mercado",
        "engajamento_score": "Score de engajamento do cliente",
        "saldo_bi":          "Saldo total na custódia",
        "qtd_produtos":      "Quantidade de produtos contratados",
        "meses_cliente":     "Tempo como cliente (meses)",
        "flag_risco":        "Flag de risco comportamental",
        "intensidade_rel":   "Intensidade relativa de movimentação",
        "segmento_enc":      "Segmento do cliente",
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
    if prob < 0.35:          # imprime apenas clientes de risco médio/alto
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
df_relatorio.to_csv("output/shap/client_explanations.csv", index=False)

# Exibe top 5 para conferência no terminal
print(f"\n  Clientes de risco médio/alto identificados: {len(df_relatorio)}")
print("\n  === TOP 5 CLIENTES EM RISCO ===")
for _, row in df_relatorio.head(5).iterrows():
    print(f"\n  Cliente: {row['cliente_id']} | Segmento: {row['segmento']}")
    print(f"  {row['explicacao']}")
    print("  " + "-" * 50)

print("\nSalvo: output/shap/client_explanations.csv")

# ── Resumo Final ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[OK] SHAP Analysis concluído!")
print("  Artefatos gerados em output/shap/:")
print("    • shap_values.csv            — valores brutos por cliente")
print("    • feature_importance_shap.csv — ranking global de features")
print("    • summary_plot.png            — distribuição de impacto")
print("    • feature_importance_shap.png — barplot de importância")
print("    • dependence_retorno.png      — dependência retorno vs churn")
print("    • client_explanations.csv     — explicações por cliente (LGPD)")
print("=" * 60)
print("\nPróximo passo: python monitor.py  (Fase 2 — MLOps Lite)")
