# =============================================================
# monitor.py — Data Drift Monitor (MLOps Lite)
# Fase 2 do Roadmap: Detectar quando o modelo "envelhece"
#
# Conceito: compara a distribuição dos dados atuais com a do
# treino usando KS-Test (numérico) e Chi-Quadrado (categórico).
# Se o drift ultrapassar o threshold, emite alerta de re-treino.
#
# Uso:
#   python monitor.py                     → usa dados do treino como referência
#   python monitor.py --ref v1            → compara com perfil salvo na v1
#   python monitor.py --alert-only        → só exibe alertas (para CI/CD)
#
# Output: output/monitor/drift_report_YYYY-MM-DD.json + .txt
# =============================================================

import os
import sys
import json
import argparse
import datetime
import warnings
warnings.filterwarnings("ignore")


class NumpyEncoder(json.JSONEncoder):
    """Converte tipos numpy/scipy para tipos Python nativos antes de serializar."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

import numpy as np
import pandas as pd
from scipy import stats

# ── Configuração ─────────────────────────────────────────────
MONITOR_DIR  = os.path.join("output", "monitor")
DATA_DIR     = os.path.join("output", "data")
MODELS_DIR   = os.path.join("output", "models")
POINTER_FILE = os.path.join(MODELS_DIR, "current_version.txt")

# Features numéricas e categóricas do contrato (PROBLEM.md)
NUMERIC_FEATURES = [
    "meses_cliente", "qtd_produtos", "retorno_12m_pct",
    "freq_contato_mes", "saldo_bi"
]
CATEGORICAL_FEATURES = ["segmento"]

# Thresholds definidos no PROBLEM.md — Seção 6.3
KS_DRIFT_THRESHOLD   = 0.20   # > 20% de drift em features numéricas → alerta
CHI_ALPHA            = 0.05   # p-value < 0.05 → distribuição significativamente diferente
SCORE_DRIFT_THRESHOLD = 0.20  # mudança de > 20% na dist. de scores → alerta


# ── Carrega perfil de referência (treino) ────────────────────

def build_reference_profile(df_train: pd.DataFrame) -> dict:
    """Constrói o perfil estatístico do dataset de treinamento."""
    profile = {"numeric": {}, "categorical": {}, "target": {}}

    for col in NUMERIC_FEATURES:
        if col not in df_train.columns:
            continue
        profile["numeric"][col] = {
            "mean"   : float(df_train[col].mean()),
            "std"    : float(df_train[col].std()),
            "min"    : float(df_train[col].min()),
            "max"    : float(df_train[col].max()),
            "p25"    : float(df_train[col].quantile(0.25)),
            "p50"    : float(df_train[col].quantile(0.50)),
            "p75"    : float(df_train[col].quantile(0.75)),
            "values" : df_train[col].tolist(),   # para KS-Test
        }

    for col in CATEGORICAL_FEATURES:
        if col not in df_train.columns:
            continue
        vc = df_train[col].value_counts(normalize=True)
        profile["categorical"][col] = vc.to_dict()

    profile["target"]["churn_rate"] = float(df_train["churn"].mean())
    profile["n_samples"]            = len(df_train)
    return profile


def load_reference_from_version(version: str) -> dict | None:
    """Carrega perfil salvo no metadata de uma versão específica."""
    meta_path = os.path.join(MODELS_DIR, version, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("data_profile")


# ── KS-Test para features numéricas ─────────────────────────

def ks_test_feature(ref_values: list, curr_values: list) -> dict:
    """Aplica KS-Test entre a distribuição de referência e a atual."""
    stat, pvalue = stats.ks_2samp(ref_values, curr_values)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value"      : round(float(pvalue), 4),
        "drift_detected": stat > KS_DRIFT_THRESHOLD,
    }


# ── Chi-Quadrado para features categóricas ──────────────────

def chi2_test_feature(ref_dist: dict, curr_series: pd.Series) -> dict:
    """Aplica Chi-Quadrado entre a distribuição de referência e a atual."""
    curr_dist = curr_series.value_counts(normalize=True)
    categories = list(ref_dist.keys())

    ref_counts  = np.array([ref_dist.get(c, 0)  for c in categories])
    curr_counts = np.array([curr_dist.get(c, 0) for c in categories])

    # Evita divisão por zero
    if ref_counts.sum() == 0 or curr_counts.sum() == 0:
        return {"drift_detected": False, "note": "distribuicao vazia"}

    # Normaliza para mesma escala
    ref_counts  = ref_counts  / ref_counts.sum()
    curr_counts = curr_counts / curr_counts.sum()

    chi2, pvalue = stats.chisquare(f_obs=curr_counts + 1e-10,
                                    f_exp=ref_counts  + 1e-10)
    return {
        "chi2_statistic": round(float(chi2), 4),
        "p_value"        : round(float(pvalue), 4),
        "drift_detected" : pvalue < CHI_ALPHA,
    }


# ── Score Drift (distribuição das previsões) ─────────────────

def score_drift(df_ref: pd.DataFrame, df_curr: pd.DataFrame,
                pipeline) -> dict:
    """Compara a distribuição dos scores do modelo entre ref e atual."""
    FEATURES = ["segmento", "meses_cliente", "qtd_produtos",
                "retorno_12m_pct", "freq_contato_mes", "saldo_bi"]

    scores_ref  = pipeline.predict_proba(df_ref[FEATURES])[:, 1]
    scores_curr = pipeline.predict_proba(df_curr[FEATURES])[:, 1]

    result = ks_test_feature(scores_ref.tolist(), scores_curr.tolist())
    result["mean_score_ref"]  = round(float(scores_ref.mean()),  4)
    result["mean_score_curr"] = round(float(scores_curr.mean()), 4)
    result["delta_mean"]      = round(abs(result["mean_score_curr"] - result["mean_score_ref"]), 4)
    return result


# ── Geração do relatório ─────────────────────────────────────

def run_monitor(ref_version: str | None = None, alert_only: bool = False):
    os.makedirs(MONITOR_DIR, exist_ok=True)

    print("=" * 60)
    print("DATA DRIFT MONITOR — Churn Finance")
    print("=" * 60)

    # Carrega dados base (simula "dados de hoje" com sample)
    print("\n[1/4] Carregando dados...")
    df_full  = pd.read_csv(os.path.join(DATA_DIR, "base_clientes.csv"))

    # Simula split treino/referência vs. produção atual
    # Em produção real: df_ref = dados de treino, df_curr = dados da semana
    np.random.seed(99)  # seed diferente → simula "dados novos"
    df_ref   = df_full.sample(frac=0.70, random_state=42)   # proxy do treino
    df_curr  = df_full.drop(df_ref.index)                    # proxy de "hoje"

    print(f"  Referencia (treino proxy): {len(df_ref)} amostras")
    print(f"  Atual (producao proxy):    {len(df_curr)} amostras")

    # Carrega pipeline de produção
    print("\n[2/4] Carregando modelo de producao...")
    import joblib

    if os.path.exists(POINTER_FILE):
        with open(POINTER_FILE) as f:
            current_ver = f.read().strip()
        pkl = os.path.join(MODELS_DIR, current_ver, "gb_pipeline.pkl")
        print(f"  Versao de producao: {current_ver}")
    else:
        pkl = os.path.join(MODELS_DIR, "gb_pipeline.pkl")
        current_ver = "N/A"
        print("  Usando modelo padrao (sem versao)")

    pipeline = joblib.load(pkl)

    # Constrói perfil de referência
    print("\n[3/4] Executando testes de drift...")
    ref_profile = build_reference_profile(df_ref)

    report = {
        "run_at"          : datetime.datetime.now().isoformat(),
        "model_version"   : current_ver,
        "n_ref"           : len(df_ref),
        "n_curr"          : len(df_curr),
        "numeric_drift"   : {},
        "categorical_drift": {},
        "score_drift"     : {},
        "summary"         : {},
    }

    drift_alerts = []

    # ── Testa features numéricas (KS-Test)
    for col in NUMERIC_FEATURES:
        if col not in df_curr.columns:
            continue
        ref_vals  = ref_profile["numeric"][col]["values"]
        curr_vals = df_curr[col].tolist()
        result    = ks_test_feature(ref_vals, curr_vals)
        result["ref_mean"]  = ref_profile["numeric"][col]["mean"]
        result["curr_mean"] = round(float(df_curr[col].mean()), 4)
        report["numeric_drift"][col] = result

        if result["drift_detected"]:
            drift_alerts.append(f"NUMERICO [{col}] — KS={result['ks_statistic']:.4f} (threshold={KS_DRIFT_THRESHOLD})")

    # ── Testa features categóricas (Chi-Quadrado)
    for col in CATEGORICAL_FEATURES:
        if col not in df_curr.columns:
            continue
        ref_dist = ref_profile["categorical"].get(col, {})
        result   = chi2_test_feature(ref_dist, df_curr[col])
        report["categorical_drift"][col] = result

        if result["drift_detected"]:
            drift_alerts.append(f"CATEGORICO [{col}] — Chi2={result.get('chi2_statistic','?'):.4f} p={result.get('p_value','?'):.4f}")

    # ── Testa drift nos scores
    sd = score_drift(df_ref, df_curr, pipeline)
    report["score_drift"] = sd
    if sd["drift_detected"]:
        drift_alerts.append(f"SCORE DRIFT — KS={sd['ks_statistic']:.4f} | mean_ref={sd['mean_score_ref']} | mean_curr={sd['mean_score_curr']}")

    # ── Drift na taxa de churn
    churn_ref  = float(df_ref["churn"].mean())
    churn_curr = float(df_curr["churn"].mean())
    churn_delta = abs(churn_curr - churn_ref)
    report["target_drift"] = {
        "churn_rate_ref"  : round(churn_ref,   4),
        "churn_rate_curr" : round(churn_curr,  4),
        "delta"           : round(churn_delta, 4),
        "alert"           : churn_delta > 0.05,
    }
    if churn_delta > 0.05:
        drift_alerts.append(f"TARGET DRIFT — Taxa churn ref={churn_ref:.2%} | curr={churn_curr:.2%} | delta={churn_delta:.2%}")

    # ── Summary
    n_alerts = len(drift_alerts)
    status   = "CRITICO" if n_alerts >= 3 else ("ATENCAO" if n_alerts >= 1 else "OK")
    report["summary"] = {
        "status"            : status,
        "n_alerts"          : n_alerts,
        "alerts"            : drift_alerts,
        "retraining_needed" : n_alerts >= 2,
        "recommendation"    : (
            "Executar pipeline.py + version_manager.py imediatamente."
            if n_alerts >= 2 else
            "Monitoramento normal. Verificar novamente na proxima semana."
        ),
    }

    # ── Salva relatório
    today     = datetime.datetime.now().strftime("%Y-%m-%d")
    json_path = os.path.join(MONITOR_DIR, f"drift_report_{today}.json")
    txt_path  = os.path.join(MONITOR_DIR, f"drift_report_{today}.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    # ── Relatório em texto (para leitura humana / e-mail)
    lines = [
        "=" * 60,
        f"DRIFT REPORT — Churn Finance | {today}",
        f"Modelo em producao: {current_ver}",
        "=" * 60,
        f"\nSTATUS GERAL: [{status}]",
        f"Alertas detectados: {n_alerts}",
        "",
    ]

    if drift_alerts:
        lines.append("ALERTAS:")
        for a in drift_alerts:
            lines.append(f"  [!] {a}")
        lines.append("")

    lines.append("FEATURES NUMERICAS (KS-Test):")
    for col, r in report["numeric_drift"].items():
        flag = "[DRIFT]" if r["drift_detected"] else "[OK]   "
        lines.append(f"  {flag} {col:<24} KS={r['ks_statistic']:.4f}  mean_ref={r['ref_mean']:.4f}  mean_curr={r['curr_mean']:.4f}")

    lines.append("\nFEATURES CATEGORICAS (Chi-Quadrado):")
    for col, r in report["categorical_drift"].items():
        flag = "[DRIFT]" if r["drift_detected"] else "[OK]   "
        lines.append(f"  {flag} {col:<24} p={r.get('p_value','?')}")

    lines.append(f"\nSCORE DRIFT (KS): {report['score_drift']['ks_statistic']} — {'[DRIFT]' if report['score_drift']['drift_detected'] else '[OK]'}")
    lines.append(f"TARGET DRIFT:      churn_ref={churn_ref:.2%}  churn_curr={churn_curr:.2%}  delta={churn_delta:.2%}")

    lines.append(f"\nRECOMENDACAO: {report['summary']['recommendation']}")
    lines.append("=" * 60)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ── Exibe no terminal
    print("\n[4/4] Resultados:")
    print("\n".join(lines))
    print(f"\nRelatorio salvo:")
    print(f"  {json_path}")
    print(f"  {txt_path}")

    # Exit code para CI/CD: 1 se re-treino necessário
    if alert_only and report["summary"]["retraining_needed"]:
        sys.exit(1)


# ── Entry point ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Data Drift Monitor — Churn Finance MLOps Lite"
    )
    parser.add_argument("--ref",        type=str,            help="Versao de referencia (ex: v1)")
    parser.add_argument("--alert-only", action="store_true", help="Retorna exit code 1 se drift critico (para CI/CD)")
    args = parser.parse_args()

    run_monitor(ref_version=args.ref, alert_only=args.alert_only)


if __name__ == "__main__":
    main()
