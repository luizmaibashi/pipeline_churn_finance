"""Prova de viabilidade: extrair o Pipeline v2 e reimplementar o forward pass
em Python 'estilo JS' (sem sklearn/numpy no caminho de inferência), comparar
contra model.predict_proba. Se o max abs diff for ~1e-12, Opção A é trivial."""
import json, math, subprocess, sys
import numpy as np, pandas as pd, joblib

ROOT = "c:/Users/Luiz Maibashi/Base_de_Conhecimento/PROJETOS/02_PORTFOLIO/pipeline_churn_finance"
sys.path.insert(0, ROOT)
import src.transformers  # noqa: F401  (unpickle do Pipeline v2)
P = joblib.load(f"{ROOT}/output/models/gb_pipeline_v2.pkl")
imp, fe, prep, clf = P.named_steps["null_imputer"], P.named_steps["fe"], P.named_steps["prep"], P.named_steps["clf"]

FEATURES_V2_BASE = ["segmento","meses_cliente","qtd_produtos","retorno_12m_pct","freq_contato_mes",
    "auc_milhoes","dias_desde_ultimo_contato","variacao_freq_contato_3m","tempo_resposta_medio_horas",
    "sem_historico_12m","cliente_novo_sem_contato_hist"]

# ---- 1. extrair params ----
params = {
    "imputer_medians": {k: float(v) for k, v in imp.medianas_.items()},
    "imputer_cols": list(imp.colunas),
    "fe": {"freq_max": float(fe.freq_max_), "qtd_max": float(fe.qtd_max_), "media_retorno": float(fe.media_retorno_)},
    "gb": {"lr": float(clf.learning_rate), "n_est": int(clf.n_estimators)},
}
# ColumnTransformer: qual coluna é ordinal-encoded e com que ordem?
print("prep transformers:", [(n, t.__class__.__name__, cols) for n, t, cols in prep.transformers_])
for n, t, cols in prep.transformers_:
    if hasattr(t, "categories_"):
        params["encoder"] = {"col": list(cols), "categories": [list(c) for c in t.categories_]}
print("prep feature_names_out:", list(prep.get_feature_names_out()))
# init prior (DummyClassifier -> log-odds)
raw_init = clf._raw_predict_init(np.zeros((1, clf.n_features_in_)))
params["gb"]["init_raw"] = float(raw_init.ravel()[0])

# árvores
trees = []
for est in clf.estimators_.ravel():
    t = est.tree_
    trees.append({
        "cl": t.children_left.tolist(), "cr": t.children_right.tolist(),
        "f": t.feature.tolist(), "th": t.threshold.tolist(),
        "val": t.value.ravel().tolist(),
    })
params["trees"] = trees
total_nodes = sum(len(t["f"]) for t in trees)

# ordem das features que o clf vê (saída do ColumnTransformer)
prep_out = list(prep.get_feature_names_out())

# ---- 2. forward pass 'estilo JS' ----
def predict_one(row: dict) -> float:
    r = dict(row)
    # imputer: mediana em nulo estrutural
    for c in params["imputer_cols"]:
        v = r.get(c)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            r[c] = params["imputer_medians"][c]
    # feature engineering
    fx = params["fe"]
    eng = round((r["freq_contato_mes"] / fx["freq_max"]) * (r["qtd_produtos"] / fx["qtd_max"]), 4)
    ret_rel = round(r["retorno_12m_pct"] - fx["media_retorno"], 2)
    flag = 1 if (ret_rel < 0 and r["freq_contato_mes"] == 0 and r["qtd_produtos"] == 1) else 0
    intens = round(math.log1p(r["meses_cliente"]) * math.log1p(r["freq_contato_mes"]), 4)
    r["engajamento_score"], r["retorno_relativo"], r["flag_risco"], r["intensidade_rel"] = eng, ret_rel, flag, intens
    # encoder do segmento
    cats = params["encoder"]["categories"][0]
    r["segmento_enc"] = float(cats.index(r["segmento"])) if r["segmento"] in cats else -1.0
    # vetor na ordem que o clf espera
    x = []
    for name in prep_out:
        key = name.split("__", 1)[-1]
        x.append(r["segmento_enc"] if name.startswith("ordinals__") else float(r[key]))
    # GB: soma das árvores
    raw = params["gb"]["init_raw"]
    lr = params["gb"]["lr"]
    for t in params["trees"]:
        node = 0
        while t["cl"][node] != -1:
            node = t["cl"][node] if x[t["f"][node]] <= t["th"][node] else t["cr"][node]
        raw += lr * t["val"][node]
    return 1.0 / (1.0 + math.exp(-raw))

# ---- 3. comparar ----
df = pd.read_csv(f"{ROOT}/output/data/base_clientes_v2_limpo.csv")
X = df[FEATURES_V2_BASE]
sk = P.predict_proba(X)[:, 1]
mine = np.array([predict_one(row) for row in X.to_dict("records")])
diff = np.abs(sk - mine)
print(f"\n{total_nodes} nós em {len(trees)} árvores")
print(f"n={len(df)}  max|Δ|={diff.max():.3e}  mean|Δ|={diff.mean():.3e}")

# grid com nulos opcionais
rng = np.random.default_rng(42)
rows = []
for _ in range(2000):
    has_ret, has_cont = rng.random() > 0.2, rng.random() > 0.2
    rows.append({
        "segmento": rng.choice(["Alta Renda","Private","Wealth","Family Office"]),
        "meses_cliente": int(rng.integers(6, 360)), "qtd_produtos": int(rng.integers(1, 12)),
        "retorno_12m_pct": round(float(rng.uniform(-20, 40)), 1) if has_ret else np.nan,
        "freq_contato_mes": int(rng.integers(0, 15)), "auc_milhoes": round(float(rng.uniform(3, 2000)), 1),
        "dias_desde_ultimo_contato": float(rng.integers(0, 200)) if has_cont else np.nan,
        "variacao_freq_contato_3m": round(float(rng.uniform(-0.9, 0.9)), 2),
        "tempo_resposta_medio_horas": round(float(rng.uniform(0, 120)), 1),
        "sem_historico_12m": int(not has_ret), "cliente_novo_sem_contato_hist": int(not has_cont),
    })
g = pd.DataFrame(rows)
sk2 = P.predict_proba(g[FEATURES_V2_BASE])[:, 1]
mine2 = np.array([predict_one(r) for r in g.to_dict("records")])
d2 = np.abs(sk2 - mine2)
print(f"grid n=2000 (c/ nulos)  max|Δ|={d2.max():.3e}  mean|Δ|={d2.mean():.3e}")

import gzip
blob = json.dumps(params, separators=(",", ":")).encode()
print(f"\nparams JSON: {len(blob)/1024:.1f} KB  |  gzip: {len(gzip.compress(blob))/1024:.1f} KB")
