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


def generate_advisors_data(n_advisors: int, seed: int = 42) -> pd.DataFrame:
    """
    Gera o dataset de assessores (Direção B, ADR-0001).
    Taxa de risco de saída calibrada com parâmetros reais de mercado (2026):
    Broker-Dealer retém 88,9%/ano (risco base ~11,1%); RIA é o canal de
    destino do maior fluxo de migração 2025 (32,1%), logo risco de SAIR
    do RIA é o menor; Wirehouse tem o maior risco estrutural reportado
    ("2026 maior ano de churn de liderança em wealth").
    """
    np.random.seed(seed)
    N = n_advisors

    canais = ["Wirehouse", "Broker-Dealer", "RIA"]
    canal_prob = [0.35, 0.45, 0.20]
    canal = np.random.choice(canais, N, p=canal_prob)

    anos_de_casa = np.random.gamma(shape=2.2, scale=3.5, size=N).round(1)
    anos_de_casa = np.clip(anos_de_casa, 0.2, 35.0)

    risco_base = {"Wirehouse": 0.15, "Broker-Dealer": 0.111, "RIA": 0.06}

    risco_saida = np.zeros(N, dtype=int)
    prob_saida = np.zeros(N)
    for c in canais:
        idx = np.where(canal == c)[0]
        for i in idx:
            mod = 1.0
            # Assessor novo na firma migra mais (turnover universal de carreira)
            if anos_de_casa[i] < 2:   mod *= 1.5
            elif anos_de_casa[i] < 5: mod *= 1.15
            # Assessor muito sênior tende a ficar (patrimônio/relacionamento consolidado)
            if anos_de_casa[i] > 15:  mod *= 0.75
            p = min(risco_base[c] * mod, 0.60)
            prob_saida[i] = p
            risco_saida[i] = int(np.random.rand() < p)

    qtd_clientes_carteira = np.random.poisson(18, N)
    qtd_clientes_carteira = np.clip(qtd_clientes_carteira, 3, 80)

    df = pd.DataFrame({
        "assessor_id"            : [f"ADV{str(i).zfill(4)}" for i in range(N)],
        "canal"                  : canal,
        "anos_de_casa"           : anos_de_casa,
        "qtd_clientes_carteira"  : qtd_clientes_carteira,
        "prob_saida_calibrada"   : prob_saida.round(4),
        "risco_saida"            : risco_saida,
    })
    return df


def attach_advisor_and_behavioral_features(
    df_clientes: pd.DataFrame, df_advisors: pd.DataFrame, seed: int = 42
) -> pd.DataFrame:
    """
    Liga cada cliente a um assessor e gera as features comportamentais
    de early-warning (Direção A, ADR-0001).

    Sinal antecedente vs. reativo: o target `churn` (v1, reativo) já existe
    no df_clientes. As features comportamentais aqui são geradas com
    correlação PARCIAL e deliberadamente ruidosa contra esse churn — não
    determinística — para não produzir um dataset "bom demais" (checkpoint
    anti-artefato-de-simulação do ADR-0001 §5). Inclui 1 feature red-herring
    (`qtd_emails_marketing_recebidos`) sem relação causal com o target,
    igual dado real tem ruído que não é sinal.
    """
    np.random.seed(seed)
    N = len(df_clientes)

    # Atribuição cliente -> assessor ponderada pela carteira de cada assessor
    pesos = df_advisors["qtd_clientes_carteira"].values
    pesos = pesos / pesos.sum()
    assessor_idx = np.random.choice(len(df_advisors), size=N, p=pesos)
    assessor_ids = df_advisors["assessor_id"].values[assessor_idx]
    risco_saida_assessor = df_advisors["risco_saida"].values[assessor_idx]

    churn = df_clientes["churn"].values

    # Recência de contato: clientes em churn tendem a ter maior hiato,
    # mas com sobreposição real (nem todo churn "avisa" via silêncio) —
    # shift moderado para manter correlação em faixa realista (~0.15-0.35)
    base_dias = np.random.gamma(shape=2.2, scale=7.0, size=N)
    dias_desde_ultimo_contato = base_dias + churn * np.random.gamma(1.0, 6.0, N)
    dias_desde_ultimo_contato = dias_desde_ultimo_contato.round(1)

    # Variação de cadência nos últimos 3 meses (negativo = esfriando)
    variacao_base = np.random.normal(0, 0.22, N)
    variacao_churn_shift = churn * np.random.normal(-0.12, 0.20, N)
    variacao_freq_contato_3m = (variacao_base + variacao_churn_shift).round(3)

    # Latência de resposta do cliente ao assessor
    latencia_base = np.random.gamma(shape=2.4, scale=6.5, size=N)
    latencia_churn_shift = churn * np.random.gamma(0.8, 7.0, N)
    tempo_resposta_medio_horas = (latencia_base + latencia_churn_shift).round(1)

    # Red herring: volume de e-mail de marketing recebido — sem relação causal
    qtd_emails_marketing_recebidos = np.random.poisson(4.5, N)

    out = df_clientes.copy()
    out["assessor_id"] = assessor_ids
    out["risco_saida_assessor"] = risco_saida_assessor
    out["auc_exposto"] = (out["saldo_bi"] * risco_saida_assessor).round(4)
    out["dias_desde_ultimo_contato"] = dias_desde_ultimo_contato
    out["variacao_freq_contato_3m"] = variacao_freq_contato_3m
    out["tempo_resposta_medio_horas"] = tempo_resposta_medio_horas
    out["qtd_emails_marketing_recebidos"] = qtd_emails_marketing_recebidos

    return out


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
