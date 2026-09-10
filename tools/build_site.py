"""Monta o artefato estático publicado pelo GitHub Pages."""
from __future__ import annotations
import gzip
import shutil, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.transformers  # noqa: F401
from src.serving_contract import FEATURES_V2_BASE, load_threshold_map, risk_level
from tools.export_model import export_model
import joblib

DATA = ["comparacao_v1_v2.csv", "feature_importance_v2.csv", "cv_scores_v2.csv", "thresholds_v2.csv", "confusion_matrix.csv", "carteira_exposta_por_assessor.csv"]
MAX_MODEL_GZIP_BYTES = 150 * 1024


def gzip_size_bytes(path: Path) -> int:
    """Mede o payload transferido, não o JSON textual antes da compressão HTTP."""
    return len(gzip.compress(path.read_bytes()))
def main():
    docs, data = ROOT / "docs", ROOT / "docs" / "data"
    data.mkdir(parents=True, exist_ok=True)
    model = joblib.load(ROOT / "output/models/gb_pipeline_v2.pkl")
    clients = pd.read_csv(ROOT / "output/data/base_clientes_v2_limpo.csv")
    clients["prob_churn"] = model.predict_proba(clients[FEATURES_V2_BASE])[:, 1]
    thresholds = load_threshold_map(ROOT / "output/data/thresholds_v2.csv")
    clients["risco"] = [risk_level(prob, segment, thresholds) for prob, segment in zip(clients.prob_churn, clients.segmento)]
    clients.to_csv(data / "base_clientes_v2_limpo.csv", index=False)
    for name in DATA: shutil.copy2(ROOT / "output/data" / name, data / name)
    explanations = pd.read_csv(ROOT / "output/shap/v2/client_explanations.csv")
    explanations.to_json(data / "client_explanations.json", orient="records", force_ascii=False)
    export_model(docs / "model.json")
    if gzip_size_bytes(docs / "model.json") > MAX_MODEL_GZIP_BYTES:
        raise ValueError("model.json excede o orçamento de 150 KB gzip")
    shutil.copy2(ROOT / "web" / "infer.mjs", docs / "infer.mjs")
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    html = html.replace("</body>", '<script type="module" src="./shap-layer.mjs"></script></body>')
    (docs / "index.html").write_text(html, encoding="utf-8")
    shutil.copy2(ROOT / "web" / "shap-layer.mjs", docs / "shap-layer.mjs")
    (docs / ".nojekyll").touch()
if __name__ == "__main__": main()
