"""Exporta o Pipeline v2 para o formato consumido por ``web/infer.mjs``."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.transformers  # noqa: F401  # necessário para desserializar o pipeline


def export_model(output: Path | None = None) -> Path:
    output = output or ROOT / "web" / "model.json"
    pipeline = joblib.load(ROOT / "output/models/gb_pipeline_v2.pkl")
    imputer, fe, prep, clf = (pipeline.named_steps[k] for k in ("null_imputer", "fe", "prep", "clf"))
    encoder = next(t for _, t, _ in prep.transformers_ if hasattr(t, "categories_"))
    trees = []
    for estimator in clf.estimators_.ravel():
        tree = estimator.tree_
        trees.append({"cl": tree.children_left.tolist(), "cr": tree.children_right.tolist(),
                      "f": tree.feature.tolist(), "th": tree.threshold.tolist(),
                      "val": tree.value.ravel().tolist()})
    payload = {
        "imputer_medians": {key: float(value) for key, value in imputer.medianas_.items()},
        "imputer_cols": list(imputer.colunas),
        "fe": {"freq_max": float(fe.freq_max_), "qtd_max": float(fe.qtd_max_),
               "media_retorno": float(fe.media_retorno_)},
        "encoder": {"categories": [list(categories) for categories in encoder.categories_]},
        "feature_order": list(prep.get_feature_names_out()),
        "gb": {"init_raw": float(clf._raw_predict_init(np.zeros((1, clf.n_features_in_))).ravel()[0]),
               "lr": float(clf.learning_rate)},
        "trees": trees,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return output


if __name__ == "__main__":
    print(export_model())
