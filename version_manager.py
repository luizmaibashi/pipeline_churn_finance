# =============================================================
# version_manager.py — Versionamento de Modelos (MLOps Lite)
# Fase 1 do Roadmap: Model Registry simples sem MLflow
#
# Uso:
#   python version_manager.py            → promove output/models/ para nova versão
#   python version_manager.py --list     → lista versões disponíveis
#   python version_manager.py --load v2  → imprime caminho do modelo v2
#
# Estrutura gerada:
#   output/
#     models/
#       current -> v2          (arquivo de ponteiro)
#       v1/
#         gb_pipeline.pkl
#         fe_params.json
#         metadata.json
#       v2/
#         gb_pipeline.pkl
#         ...
# =============================================================

import os
import sys
import json
import shutil
import joblib
import argparse
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

MODELS_DIR  = os.path.join("output", "models")
DATA_DIR    = os.path.join("output", "data")
POINTER_FILE = os.path.join(MODELS_DIR, "current_version.txt")

# ── Helpers ──────────────────────────────────────────────────

def get_all_versions() -> list[str]:
    """Retorna lista ordenada de versões existentes (ex: ['v1', 'v2'])."""
    versions = [
        d for d in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, d))
        and d.startswith("v") and d[1:].isdigit()
    ]
    return sorted(versions, key=lambda v: int(v[1:]))


def get_next_version(versions: list[str]) -> str:
    """Retorna o próximo número de versão."""
    if not versions:
        return "v1"
    last = int(versions[-1][1:])
    return f"v{last + 1}"


def get_current_version() -> str | None:
    """Lê o ponteiro para a versão atual de produção."""
    if not os.path.exists(POINTER_FILE):
        return None
    with open(POINTER_FILE, "r") as f:
        return f.read().strip()


def set_current_version(version: str):
    """Atualiza o ponteiro para a versão de produção."""
    with open(POINTER_FILE, "w") as f:
        f.write(version)


# ── Ação principal: promover novo modelo ────────────────────

def promote():
    """Empacota os artefatos atuais em uma nova versão versionada."""

    print("=" * 60)
    print("VERSION MANAGER — Promovendo novo modelo")
    print("=" * 60)

    # Verifica pré-requisitos
    required = [
        os.path.join(MODELS_DIR, "gb_pipeline.pkl"),
        os.path.join(MODELS_DIR, "fe_params.json"),
    ]
    for f in required:
        if not os.path.exists(f):
            print(f"\n[ERRO] Arquivo nao encontrado: {f}")
            print("       Execute 'python pipeline.py' primeiro.")
            sys.exit(1)

    versions  = get_all_versions()
    new_ver   = get_next_version(versions)
    ver_dir   = os.path.join(MODELS_DIR, new_ver)
    os.makedirs(ver_dir, exist_ok=True)

    print(f"\nCriando versao: {new_ver}")
    print(f"Diretorio:      {ver_dir}")

    # ── Copia artefatos do modelo
    for fname in ["gb_pipeline.pkl", "fe_params.json"]:
        src = os.path.join(MODELS_DIR, fname)
        dst = os.path.join(ver_dir, fname)
        shutil.copy2(src, dst)
        print(f"  Copiado: {fname}")

    # ── Calcula métricas para o metadata
    print("\nCalculando metricas para o registro...")
    pipeline = joblib.load(os.path.join(ver_dir, "gb_pipeline.pkl"))

    df = pd.read_csv(os.path.join(DATA_DIR, "base_clientes.csv"))
    FEATURES = ["segmento", "meses_cliente", "qtd_produtos",
                "retorno_12m_pct", "freq_contato_mes", "saldo_bi"]
    X = df[FEATURES]
    y = df["churn"]

    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    f1  = float(f1_score(y, y_pred, average="macro"))
    roc = float(roc_auc_score(y, y_prob))
    n_churn_risk = int((y_prob >= 0.35).sum())

    # ── Feature importance do SHAP (se existir)
    shap_csv = os.path.join("output", "shap", "feature_importance_shap.csv")
    top_feature = "N/A"
    if os.path.exists(shap_csv):
        imp = pd.read_csv(shap_csv)
        top_feature = str(imp.iloc[0]["feature"])

    # ── Metadata.json
    meta = {
        "version"      : new_ver,
        "promoted_at"  : datetime.datetime.now().isoformat(),
        "algorithm"    : "GradientBoostingClassifier",
        "random_state" : 42,
        "metrics": {
            "f1_macro"  : round(f1,  4),
            "roc_auc"   : round(roc, 4),
            "threshold" : 0.40,
        },
        "data_profile": {
            "n_samples"       : int(len(df)),
            "churn_rate_pct"  : round(float(y.mean() * 100), 2),
            "n_features"      : len(FEATURES),
            "top_shap_feature": top_feature,
        },
        "artifacts": [
            "gb_pipeline.pkl",
            "fe_params.json",
            "metadata.json",
        ],
        "notes": f"Promovido via version_manager.py. Versao anterior: {versions[-1] if versions else 'N/A'}.",
    }

    with open(os.path.join(ver_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Gerado: metadata.json")

    # ── Atualiza ponteiro de produção
    set_current_version(new_ver)
    print(f"\n[OK] Producao apontando para: {new_ver}")

    # ── Resumo
    print("\n" + "-" * 40)
    print(f"  Versao   : {new_ver}")
    print(f"  F1-macro : {f1:.4f}")
    print(f"  ROC-AUC  : {roc:.4f}")
    print(f"  Clientes em risco (prob >= 0.35): {n_churn_risk}")
    print(f"  Top feature (SHAP): {top_feature}")
    print("-" * 40)

    # ── Histórico de versões
    versions = get_all_versions()
    if len(versions) > 1:
        print("\nHistorico de versoes disponivel:")
        list_versions()


# ── Ação: listar versões ─────────────────────────────────────

def list_versions():
    """Exibe todas as versões com suas métricas."""
    versions = get_all_versions()
    current  = get_current_version()

    if not versions:
        print("\nNenhuma versao encontrada. Execute sem argumentos para criar a primeira.")
        return

    print(f"\n{'Versao':<8} {'F1-macro':<10} {'ROC-AUC':<10} {'Data':<22} {'Status'}")
    print("-" * 65)
    for v in versions:
        meta_path = os.path.join(MODELS_DIR, v, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        status = "<-- PRODUCAO" if v == current else ""
        dt     = m["promoted_at"][:16].replace("T", " ")
        f1_v   = m["metrics"]["f1_macro"]
        roc_v  = m["metrics"]["roc_auc"]
        print(f"  {v:<6} {f1_v:<10.4f} {roc_v:<10.4f} {dt:<22} {status}")


# ── Ação: carregar versão específica ─────────────────────────

def load_version(version: str):
    """Retorna o caminho do pipeline de uma versão específica."""
    ver_dir = os.path.join(MODELS_DIR, version)
    pkl     = os.path.join(ver_dir, "gb_pipeline.pkl")
    if not os.path.exists(pkl):
        print(f"[ERRO] Versao '{version}' nao encontrada em {ver_dir}")
        sys.exit(1)
    print(f"\nCaminho do modelo {version}: {os.path.abspath(pkl)}")
    meta_path = os.path.join(ver_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        print(f"F1-macro: {m['metrics']['f1_macro']} | ROC-AUC: {m['metrics']['roc_auc']}")
        print(f"Promovido em: {m['promoted_at'][:16]}")
    return pkl


# ── Entry point ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Version Manager — Churn Finance MLOps Lite"
    )
    parser.add_argument("--list",    action="store_true", help="Lista todas as versoes")
    parser.add_argument("--load",    type=str,            help="Carrega versao especifica (ex: v2)")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.list:
        list_versions()
    elif args.load:
        load_version(args.load)
    else:
        promote()


if __name__ == "__main__":
    main()
