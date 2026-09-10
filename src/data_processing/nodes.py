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

    # Peso de atração de carteira — NÃO é contagem de clientes (isso só se
    # sabe depois da atribuição real). Assessor sênior atrai carteira maior
    # e mais consolidada; é usado como peso relativo em attach_advisor_*.
    # `qtd_clientes_carteira` é preenchido depois, por contagem real, para
    # nunca divergir do dataset de clientes (achado de EDA: campo declarado
    # antes da atribuição real batia 17,9 de média contra 4,1 real).
    peso_atracao_carteira = (1.0 + anos_de_casa * 0.15).round(3)

    df = pd.DataFrame({
        "assessor_id"            : [f"ADV{str(i).zfill(4)}" for i in range(N)],
        "canal"                  : canal,
        "anos_de_casa"           : anos_de_casa,
        "peso_atracao_carteira"  : peso_atracao_carteira,
        "prob_saida_calibrada"   : prob_saida.round(4),
        "risco_saida"            : risco_saida,
    })
    return df


def attach_advisor_and_behavioral_features(
    df_clientes: pd.DataFrame, df_advisors: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    Retorna (df_clientes_v2, df_advisors_com_carteira_real) — a contagem de
    clientes por assessor é calculada aqui, pós-atribuição, para nunca
    divergir do dataset de clientes (ver nota em generate_advisors_data).
    """
    np.random.seed(seed)
    N = len(df_clientes)

    # Atribuição cliente -> assessor ponderada pelo peso de atração
    pesos = df_advisors["peso_atracao_carteira"].values
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

    # Carteira real por assessor — contagem pós-atribuição, nunca declarada
    # antes (a divergência anterior era: campo declarado dizia média 17,9,
    # atribuição real dava média 4,1 — mesma família de "coluna promete e
    # não é honrada pelo dado" do checklist de EDA).
    contagem_real = out["assessor_id"].value_counts().rename("qtd_clientes_carteira")
    df_advisors_out = df_advisors.merge(
        contagem_real, left_on="assessor_id", right_index=True, how="left"
    )
    df_advisors_out["qtd_clientes_carteira"] = df_advisors_out["qtd_clientes_carteira"].fillna(0).astype(int)

    return out, df_advisors_out


def inject_data_quality_issues(
    df_clientes: pd.DataFrame, df_advisors: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Etapa 1 (proposta aprovada pelo Luiz): injeta 6 tipos de sujeira de dado,
    cada um ancorado numa causa de negócio plausível de gestora de alta
    renda com sistemas legados — não é ruído aleatório sem motivo.

    Deliberadamente NÃO trata nada aqui (sem imputação, sem dedup, sem
    correção de escala). Tratamento é Etapa 2, separada, sobre um EDA
    escrito em cima deste dataset sujo — a separação existe para que o
    processo de descoberta fique documentado, não escondido atrás de uma
    função que suja e limpa na mesma passada.

    1. Duplicata de cliente — onboarding por dois assessores diferentes
       ou migração de sistema legado pós-fusão (mesmo cliente_id, um
       campo levemente divergente — o caso difícil de pegar).
    2. Nulo em freq_contato_mes — log de CRM incompleto; assessor mais
       sênior/menos digital registra menos contato (viés deliberado:
       correlaciona com anos_de_casa do assessor, não é MCAR).
    3. Nulo estrutural em dias_desde_ultimo_contato/tempo_resposta —
       cliente novo (<60 dias) ainda não acumulou histórico suficiente
       para a métrica existir.
    4. Outlier por erro de digitação em saldo_bi — assessor erra escala
       (casa decimal) ao digitar, valor sai 1000x maior.
    5. Sentinela mascarada em retorno_12m_pct — sistema legado usa -999
       em vez de NULL quando cliente não completou 12 meses de histórico.
    6. canal do assessor sujo — cadastro manual inconsistente
       ("RIA" / "R.I.A." / "ria").
    """
    rng = np.random.default_rng(seed)
    df = df_clientes.copy()
    adv = df_advisors.copy()
    N = len(df)

    # 1. Duplicata de cliente (~1,5% da base) — mesmo ID, saldo_bi levemente
    #    divergente (recadastro capturou saldo num dia diferente)
    n_dup = max(1, int(N * 0.015))
    idx_dup = rng.choice(N, size=n_dup, replace=False)
    linhas_dup = df.iloc[idx_dup].copy()
    linhas_dup["saldo_bi"] = (linhas_dup["saldo_bi"] * rng.uniform(0.97, 1.03, n_dup)).round(4)
    df = pd.concat([df, linhas_dup], ignore_index=True)

    # 2. Nulo em freq_contato_mes — probabilidade de nulo cresce com
    #    anos_de_casa do assessor (assessor sênior digitaliza menos)
    anos_por_cliente = df["assessor_id"].map(adv.set_index("assessor_id")["anos_de_casa"])
    prob_nulo_contato = np.clip(0.03 + anos_por_cliente.fillna(0) * 0.012, 0, 0.45)
    mask_nulo_contato = rng.random(len(df)) < prob_nulo_contato.values
    df.loc[mask_nulo_contato, "freq_contato_mes"] = np.nan

    # 3. Nulo estrutural — cliente novo sem histórico suficiente
    mask_cliente_novo = df["meses_cliente"] < 2
    df.loc[mask_cliente_novo, "dias_desde_ultimo_contato"] = np.nan
    df.loc[mask_cliente_novo, "tempo_resposta_medio_horas"] = np.nan

    # 4. Outlier por erro de digitação em saldo_bi (~0,4% da base)
    n_erro_escala = max(1, int(len(df) * 0.004))
    idx_erro_escala = rng.choice(len(df), size=n_erro_escala, replace=False)
    df.loc[df.index[idx_erro_escala], "saldo_bi"] = df.loc[df.index[idx_erro_escala], "saldo_bi"] * 1000

    # 5. Sentinela mascarada em retorno_12m_pct para cliente sem 12 meses
    mask_sem_12m = df["meses_cliente"] < 12
    idx_sem_12m = df.index[mask_sem_12m]
    frac_sentinela = rng.random(len(idx_sem_12m)) < 0.6
    df.loc[idx_sem_12m[frac_sentinela], "retorno_12m_pct"] = -999.0

    # 6. canal do assessor sujo — variantes de string para ~10% dos "RIA"
    idx_ria = adv.index[adv["canal"] == "RIA"]
    n_sujo = max(1, int(len(idx_ria) * 0.10))
    idx_sujo = rng.choice(idx_ria, size=min(n_sujo, len(idx_ria)), replace=False)
    variantes = rng.choice(["R.I.A.", "ria"], size=len(idx_sujo))
    adv.loc[idx_sujo, "canal"] = variantes

    return df, adv


def aggregate_carteira_exposta_por_assessor(
    df_clientes_v2: pd.DataFrame, df_advisors: pd.DataFrame
) -> pd.DataFrame:
    """
    Direção B (ADR-0001, correção pós-auditoria 2026-09-09): AuC exposto
    NÃO é feature do modelo de churn do cliente (removido de
    model_training/nodes.py — importância 1,3%, risco de saída de
    assessor é quase independente do churn individual, corr=-0,04).

    É um produto de dado separado: agregação por assessor respondendo
    "se ESTE assessor sair, quanto AuC da carteira dele está exposto?" —
    insumo para dashboard de risco de carteira, não para o classificador.
    """
    agg = df_clientes_v2.groupby("assessor_id").agg(
        qtd_clientes=("cliente_id", "count"),
        auc_total_carteira=("saldo_bi", "sum"),
        auc_exposto_total=("auc_exposto", "sum"),
    ).reset_index()

    agg["pct_carteira_exposta"] = (
        agg["auc_exposto_total"] / agg["auc_total_carteira"]
    ).round(4)

    out = agg.merge(
        df_advisors[["assessor_id", "canal", "anos_de_casa", "risco_saida"]],
        on="assessor_id", how="left"
    )
    return out.sort_values("auc_exposto_total", ascending=False).reset_index(drop=True)


def clean_clientes_v2_bruto(df_bruto: pd.DataFrame, df_advisors_bruto: pd.DataFrame) -> pd.DataFrame:
    """
    Etapa 3 (ADR-0001): aplica as 6 decisões de tratamento documentadas em
    reports/eda_clientes_v2_bruto.md — cada uma pela causa, não por reflexo.
    Não imputa nada que a EDA marcou como nulo estrutural.
    """
    df = df_bruto.copy()
    adv = df_advisors_bruto.copy()

    # 1. Duplicata de cliente_id — manter 1ª ocorrência (sem timestamp real
    #    no dataset sintético para decidir "mais recente" de outra forma;
    #    limitação documentada no EDA).
    df = df.drop_duplicates(subset="cliente_id", keep="first").reset_index(drop=True)

    # 6. canal sujo — padronizar variantes de string (confirmado no EDA:
    #    mesma categoria, não 3 grupos reais)
    adv["canal"] = adv["canal"].replace({"R.I.A.": "RIA", "ria": "RIA"})

    # 4. Outlier de escala em saldo_bi — critério relacional (>p99 do
    #    segmento * 20), corrige dividindo por 1000 em vez de descartar
    for seg in df["segmento"].unique():
        mask_seg = df["segmento"] == seg
        p99_seg = df.loc[mask_seg, "saldo_bi"].quantile(0.99)
        mask_outlier = mask_seg & (df["saldo_bi"] > p99_seg * 20)
        df.loc[mask_outlier, "saldo_bi"] = df.loc[mask_outlier, "saldo_bi"] / 1000

    # 5. Sentinela -999 em retorno_12m_pct — vira nulo estrutural (cliente
    #    sem 12 meses de histórico), nunca valor numérico
    df.loc[df["retorno_12m_pct"] == -999.0, "retorno_12m_pct"] = np.nan

    # 3. Nulo estrutural (dias_desde_ultimo_contato/tempo_resposta/retorno)
    #    — flag para o modelo saber que a ausência é por cliente novo/sem
    #    histórico, não imputação que inventaria comportamento inexistente
    df["sem_historico_12m"] = df["retorno_12m_pct"].isna().astype(int)
    df["cliente_novo_sem_contato_hist"] = df["dias_desde_ultimo_contato"].isna().astype(int)

    # 2. Nulo em freq_contato_mes — EDA confirmou que a nulidade não prediz
    #    o alvo (qui-quadrado p=0.72), então imputação é segura. Imputa
    #    pela média do grupo segmento+canal do assessor (preserva o viés
    #    operacional real em vez de usar a média global, que o esconderia)
    canal_por_cliente = df["assessor_id"].map(adv.set_index("assessor_id")["canal"])
    df["_canal_assessor_tmp"] = canal_por_cliente
    media_grupo = df.groupby(["segmento", "_canal_assessor_tmp"])["freq_contato_mes"].transform("mean")
    df["freq_contato_mes"] = df["freq_contato_mes"].fillna(media_grupo)
    # fallback: grupo sem nenhum valor não-nulo (raro) usa média global
    df["freq_contato_mes"] = df["freq_contato_mes"].fillna(df["freq_contato_mes"].mean())
    df = df.drop(columns=["_canal_assessor_tmp"])

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
