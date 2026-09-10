import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class StructuralNullImputer(BaseEstimator, TransformerMixin):
    """
    Imputa nulo estrutural (Direção A, ADR-0001) por mediana aprendida
    SÓ no treino (evita leakage — mesma disciplina do FeatureEngineer).
    Preserva DataFrame para não quebrar transformers subsequentes que
    dependem de nome de coluna (FeatureEngineer, ColumnTransformer).
    """
    def __init__(self, colunas):
        self.colunas = colunas

    def fit(self, X, y=None):
        self.medianas_ = {c: X[c].median() for c in self.colunas}
        return self

    def transform(self, X):
        X_out = X.copy()
        for c in self.colunas:
            X_out[c] = X_out[c].fillna(self.medianas_[c])
        return X_out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.freq_max_ = X["freq_contato_mes"].max()
        self.qtd_max_ = X["qtd_produtos"].max()
        self.media_retorno_ = X["retorno_12m_pct"].mean()
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        # 1. Engajamento
        X_out["engajamento_score"] = (
            (X_out["freq_contato_mes"] / self.freq_max_) *
            (X_out["qtd_produtos"] / self.qtd_max_)
        ).round(4)
        
        # 2. Retorno Relativo
        X_out["retorno_relativo"] = (X_out["retorno_12m_pct"] - self.media_retorno_).round(2)
        
        # 3. Flag de Risco
        X_out["flag_risco"] = (
            (X_out["retorno_relativo"] < 0) &
            (X_out["freq_contato_mes"] == 0) &
            (X_out["qtd_produtos"] == 1)
        ).astype(int)
        
        # 4. Intensidade de Relacionamento
        X_out["intensidade_rel"] = (
            np.log1p(X_out["meses_cliente"]) *
            np.log1p(X_out["freq_contato_mes"])
        ).round(4)
        
        return X_out
