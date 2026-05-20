import os
import pytest
import joblib
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

MODEL_PKL = "output/models/gb_pipeline.pkl"
DATA_CSV = "output/data/base_clientes.csv"

def test_model_loading_and_prediction():
    """Testa se o modelo serializado existe, pode ser carregado e faz previsões válidas."""
    assert os.path.exists(MODEL_PKL), "Execute pipeline.py para gerar o modelo final."
    
    model = joblib.load(MODEL_PKL)
    
    # Perfil de cliente hipotético
    cli = pd.DataFrame([{
        "segmento":        "Wealth",
        "meses_cliente":   36,
        "qtd_produtos":    3,
        "retorno_12m_pct": 11.5,
        "freq_contato_mes":2,
        "saldo_bi":        0.5
    }])
    
    pred = model.predict(cli)
    prob = model.predict_proba(cli)[:, 1]
    
    assert len(pred) == 1
    assert pred[0] in [0, 1]
    assert 0.0 <= prob[0] <= 1.0


def test_model_performance_thresholds():
    """Testa se o modelo atende aos limites analíticos estabelecidos no contrato (F1 e ROC-AUC)."""
    assert os.path.exists(MODEL_PKL)
    assert os.path.exists(DATA_CSV)
    
    model = joblib.load(MODEL_PKL)
    df = pd.read_csv(DATA_CSV)
    
    FEATURES_BASE = [
        "segmento", "meses_cliente", "qtd_produtos",
        "retorno_12m_pct", "freq_contato_mes", "saldo_bi"
    ]
    
    X = df[FEATURES_BASE]
    y = df["churn"]
    
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    f1_macro = f1_score(y, y_pred, average="macro")
    roc_auc = roc_auc_score(y, y_prob)
    
    # Limites definidos no PROBLEM.md
    assert f1_macro >= 0.55, f"F1-macro {f1_macro:.4f} abaixo do mínimo contratual de 0.55"
    assert roc_auc >= 0.70, f"ROC-AUC {roc_auc:.4f} abaixo do mínimo contratual de 0.70"
