import pytest
import numpy as np
import pandas as pd
from transformers import FeatureEngineer

def test_feature_engineer_creation():
    """Testa se as colunas esperadas são adicionadas pelo FeatureEngineer."""
    df_dummy = pd.DataFrame({
        "segmento":        ["Varejo", "Wealth"],
        "meses_cliente":   [12, 24],
        "qtd_produtos":    [2, 4],
        "retorno_12m_pct": [10.0, 15.0],
        "freq_contato_mes":[1, 3],
        "saldo_bi":        [0.05, 0.8]
    })
    
    fe = FeatureEngineer()
    fe.fit(df_dummy)
    df_transformed = fe.transform(df_dummy)
    
    # Verifica novas colunas
    new_cols = ["engajamento_score", "retorno_relativo", "flag_risco", "intensidade_rel"]
    for col in new_cols:
        assert col in df_transformed.columns
        
    # Verifica integridade: dimensões originais não alteradas
    assert len(df_transformed) == len(df_dummy)


def test_feature_engineer_logic():
    """Testa o cálculo matemático das colunas de engenharia de features."""
    df_dummy = pd.DataFrame({
        "segmento":        ["Varejo", "Wealth", "Alta Renda"],
        "meses_cliente":   [10, 20, 30],
        "qtd_produtos":    [1, 5, 2],
        "retorno_12m_pct": [5.0, 12.0, 10.0],
        "freq_contato_mes":[0, 10, 2],
        "saldo_bi":        [0.01, 1.2, 0.3]
    })
    
    fe = FeatureEngineer()
    fe.fit(df_dummy)
    df_transformed = fe.transform(df_dummy)
    
    # freq_max = 10, qtd_max = 5
    # engajamento_score = (freq_contato_mes / freq_max) * (qtd_produtos / qtd_max)
    # Cliente 0: (0/10) * (1/5) = 0.0
    # Cliente 1: (10/10) * (5/5) = 1.0
    assert df_transformed.loc[0, "engajamento_score"] == 0.0
    assert df_transformed.loc[1, "engajamento_score"] == 1.0
    
    # media_retorno = (5 + 12 + 10) / 3 = 9.0
    # retorno_relativo = retorno_12m_pct - media_retorno
    # Cliente 0: 5.0 - 9.0 = -4.0
    assert df_transformed.loc[0, "retorno_relativo"] == -4.0
    
    # flag_risco = (retorno_relativo < 0) & (freq_contato_mes == 0) & (qtd_produtos == 1)
    # Cliente 0: retorno_relativo = -4 (<0), freq = 0, qtd = 1 -> deve ser 1
    # Cliente 2: retorno_relativo = 1.0 (>=0), freq = 2, qtd = 2 -> deve ser 0
    assert df_transformed.loc[0, "flag_risco"] == 1
    assert df_transformed.loc[2, "flag_risco"] == 0
