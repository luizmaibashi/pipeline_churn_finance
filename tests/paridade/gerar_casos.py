"""Gera casos determinísticos para o gate JS/Python."""
from __future__ import annotations
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.transformers  # noqa: F401
from src.serving_contract import FEATURES_V2_BASE

OUT = Path(__file__).with_name("casos.json")
def main():
    df = pd.read_csv(ROOT / "output/data/base_clientes_v2_limpo.csv")
    rng = np.random.default_rng(42)
    rows = df[FEATURES_V2_BASE].to_dict("records")
    for _ in range(1800):
        retorno, contato = rng.random() > .2, rng.random() > .2
        rows.append({"segmento": str(rng.choice(["Alta Renda", "Private", "Wealth", "Family Office"])), "meses_cliente": int(rng.integers(6, 361)), "qtd_produtos": int(rng.integers(1, 13)), "retorno_12m_pct": round(float(rng.uniform(-20, 40)), 1) if retorno else None, "freq_contato_mes": int(rng.integers(0, 16)), "auc_milhoes": round(float(rng.uniform(3, 2000)), 1), "dias_desde_ultimo_contato": float(rng.integers(0, 201)) if contato else None, "variacao_freq_contato_3m": round(float(rng.uniform(-.9, .9)), 2), "tempo_resposta_medio_horas": round(float(rng.uniform(0, 120)), 1), "sem_historico_12m": int(not retorno), "cliente_novo_sem_contato_hist": int(not contato)})
    model = joblib.load(ROOT / "output/models/gb_pipeline_v2.pkl")
    expected = model.predict_proba(pd.DataFrame(rows)[FEATURES_V2_BASE])[:, 1]
    # CSV preserva NaN; JSON não. O contrato do browser representa nulo estrutural
    # por ``null`` e o inferidor o imputa antes do feature engineering.
    rows = [{key: (None if pd.isna(value) else value) for key, value in row.items()} for row in rows]
    OUT.write_text(json.dumps([{"input": row, "prob_esperada": float(prob)} for row, prob in zip(rows, expected)], separators=(",", ":")), encoding="utf-8")
if __name__ == "__main__": main()
