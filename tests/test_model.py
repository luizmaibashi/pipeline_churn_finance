import os
import pytest
import joblib
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

MODEL_PKL = "output/models/gb_pipeline_v2.pkl"
DATA_CSV = "output/data/base_clientes_v2_limpo.csv"

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
        "auc_milhoes":     120.0,
        "dias_desde_ultimo_contato": 18.0,
        "variacao_freq_contato_3m": -0.05,
        "tempo_resposta_medio_horas": 20.0,
        "sem_historico_12m": 0,
        "cliente_novo_sem_contato_hist": 0,
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
        "retorno_12m_pct", "freq_contato_mes", "auc_milhoes",
        "dias_desde_ultimo_contato", "variacao_freq_contato_3m",
        "tempo_resposta_medio_horas", "sem_historico_12m",
        "cliente_novo_sem_contato_hist",
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


def test_thresholds_v2_registrados_e_por_segmento():
    """A API só pode servir thresholds persistidos pela calibração interna."""
    report_path = "reports/thresholds_v2.md"
    csv_path = "output/data/thresholds_v2.csv"
    assert os.path.exists(report_path)
    assert os.path.exists(csv_path)
    txt = open(report_path, encoding="utf-8").read()
    for segmento in ["Alta Renda", "Private", "Wealth", "Family Office"]:
        assert segmento in txt


def test_thresholds_v2_nao_reportam_contagens_fora_do_segmento():
    thresholds = pd.read_csv("output/data/thresholds_v2.csv")
    assert (thresholds["fn"] <= thresholds["n_valid"]).all()
    assert (thresholds["fp"] <= thresholds["n_valid"]).all()
