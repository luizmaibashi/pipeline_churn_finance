import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from transformers import FeatureEngineer

def generate_synthetic_data(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Gera o dataset de clientes com dados sintéticos simulados."""
    np.random.seed(seed)
    N = n_samples

    segmentos = ["Varejo", "Alta Renda", "Wealth", "Corporate"]
    seg_prob = [0.65, 0.24, 0.08, 0.03]
    seg = np.random.choice(segmentos, N, p=seg_prob)

    meses_cli = np.random.randint(1, 144, N)
    qtd_prod = np.random.randint(1, 9, N)
    retorno = np.random.normal(11.5, 4.2, N).round(2)
    freq_cont = np.random.poisson(2.8, N)
    saldo = np.random.lognormal(-1.8, 1.3, N).round(4)

    taxa_base = {"Varejo": 0.18, "Alta Renda": 0.09, "Wealth": 0.05, "Corporate": 0.04}

    churn = np.zeros(N, dtype=int)
    for s in segmentos:
        idx = np.where(seg == s)[0]
        for i in idx:
            mod = 1.0
            if retorno[i] < 8.0:    mod *= 1.50
            if freq_cont[i] == 0:   mod *= 1.70
            if qtd_prod[i] == 1:    mod *= 1.25
            if meses_cli[i] < 12:   mod *= 1.35
            if saldo[i] < 0.1:      mod *= 1.40
            p = min(taxa_base[s] * mod, 0.75)
            churn[i] = int(np.random.rand() < p)

    df = pd.DataFrame({
        "cliente_id"      : [f"CLI{str(i).zfill(5)}" for i in range(N)],
        "segmento"        : seg,
        "meses_cliente"   : meses_cli,
        "qtd_produtos"    : qtd_prod,
        "retorno_12m_pct" : retorno,
        "freq_contato_mes": freq_cont,
        "saldo_bi"        : saldo,
        "churn"           : churn
    })
    return df


def split_data(df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o dataset de entrada de forma estratificada pelo target churn."""
    from sklearn.model_selection import train_test_split
    
    # Mantém o dataframe completo no split para preservar os IDs e dados originais
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["churn"]
    )
    return train_df.copy(), test_df.copy()


def run_feature_engineering(df: pd.DataFrame, train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Executa a engenharia de features de forma pura (sem data leakage).
    Ajusta (fits) os transformadores no conjunto de treino e aplica em todo o dataset.
    Retorna o dataset transformado para visualização/dashboard e os parâmetros aprendidos.
    """
    FEATURES_BASE = [
        "segmento", "meses_cliente", "qtd_produtos", 
        "retorno_12m_pct", "freq_contato_mes", "saldo_bi"
    ]
    
    X_train = train_df[FEATURES_BASE]
    
    # Ajusta o feature engineer apenas nos dados de treino
    fe = FeatureEngineer()
    fe.fit(X_train)
    
    # Transforma todo o dataset
    X_fe_all = fe.transform(df)
    
    # Ajusta o OrdinalEncoder de segmento apenas no treino
    encoder = OrdinalEncoder(categories=[["Varejo", "Alta Renda", "Wealth", "Corporate"]])
    encoder.fit(X_train[["segmento"]])
    X_fe_all["segmento_enc"] = encoder.transform(X_fe_all[["segmento"]])
    
    # Junta de volta as IDs e targets
    df_fe = pd.concat([df["cliente_id"], X_fe_all, df[["churn"]]], axis=1)
    
    # Metadados/parâmetros para persistência
    fe_params = {
        "media_retorno": float(fe.media_retorno_),
        "categories": [c.tolist() for c in encoder.categories_]
    }
    
    return df_fe, fe_params
