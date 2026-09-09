"""
Testes de contrato de dados — anti-leakage e checkpoint de calibração sintética.

Cobre duas lacunas identificadas no Blind Spot Pass (ADR-0001):
1. PROBLEM.md define regras R-01 a R-05 (anti-leakage temporal) que nunca
   tinham teste algum, apesar de serem o "guardrail crítico" declarado.
2. Dataset sintético novo (Direção A+B) precisa de checkpoint que prove
   que a correlação sinal-comportamental vs. churn não é "boa demais"
   (artefato de simulação, não achado transferível) nem "zero" (feature
   inútil) — faixa alvo documentada no ADR-0001 §5.
"""
import numpy as np
import pytest

from src.data_processing.nodes import (
    generate_synthetic_data,
    generate_advisors_data,
    attach_advisor_and_behavioral_features,
)

SEED = 42


@pytest.fixture(scope="module")
def dataset_v2():
    df_cli = generate_synthetic_data(1200, seed=SEED)
    df_adv_raw = generate_advisors_data(300, seed=SEED)
    df_v2, df_adv = attach_advisor_and_behavioral_features(df_cli, df_adv_raw, seed=SEED)
    return df_cli, df_adv, df_v2


# ── R-03: features proibidas nunca podem existir no dataset de scoring ──

def test_nao_existem_colunas_proibidas_por_leakage(dataset_v2):
    """PROBLEM.md R-03: proibido usar como feature dado calculado após D-0."""
    _, _, df_v2 = dataset_v2
    colunas_proibidas = {
        "data_encerramento",
        "motivo_saida",
        "flag_solicitacao_resgate_pendente",
    }
    assert colunas_proibidas.isdisjoint(df_v2.columns), (
        f"Dataset contém coluna proibida por R-03: "
        f"{colunas_proibidas & set(df_v2.columns)}"
    )


# ── Checkpoint anti-artefato-de-simulação (ADR-0001 §5) ──

@pytest.mark.parametrize("coluna,corr_min,corr_max", [
    ("dias_desde_ultimo_contato", 0.10, 0.40),
    ("variacao_freq_contato_3m", -0.40, -0.10),
    ("tempo_resposta_medio_horas", 0.10, 0.40),
])
def test_sinal_comportamental_correlaciona_em_faixa_realista(dataset_v2, coluna, corr_min, corr_max):
    """
    Correlação forte demais (>0.4 ou <-0.4) indica dado sintético
    determinístico demais — vira artefato de simulação, não achado
    transferível. Correlação fraca demais (~0) indica feature inútil.
    """
    _, _, df_v2 = dataset_v2
    corr = np.corrcoef(df_v2[coluna], df_v2["churn"])[0, 1]
    assert corr_min <= corr <= corr_max, (
        f"{coluna}: corr={corr:.3f} fora da faixa realista "
        f"[{corr_min}, {corr_max}] — dataset sintético bom/ruim demais"
    )


def test_red_herring_nao_correlaciona_com_churn(dataset_v2):
    """qtd_emails_marketing_recebidos é ruído deliberado — sem relação causal."""
    _, _, df_v2 = dataset_v2
    corr = np.corrcoef(df_v2["qtd_emails_marketing_recebidos"], df_v2["churn"])[0, 1]
    assert abs(corr) < 0.10, (
        f"Red herring correlaciona demais (corr={corr:.3f}) — "
        f"gerador vazou sinal não intencional na feature de ruído"
    )


# ── Direção B: calibração de risco de saída de assessor ──

def test_risco_saida_assessor_calibrado_por_canal(dataset_v2):
    """
    RIA deve ter o menor risco de saída (é o canal de destino do maior
    fluxo de migração 2025 — ADR-0001). Não testamos os valores exatos
    (variância amostral), só a ordenação relativa que a pesquisa de
    mercado sustenta.
    """
    _, df_adv, _ = dataset_v2
    risco_por_canal = df_adv.groupby("canal")["risco_saida"].mean()
    assert risco_por_canal["RIA"] < risco_por_canal["Broker-Dealer"], (
        "RIA deveria ter risco de saída menor que Broker-Dealer"
    )
    assert risco_por_canal["RIA"] < risco_por_canal["Wirehouse"], (
        "RIA deveria ter risco de saída menor que Wirehouse"
    )


def test_auc_exposto_e_produto_saldo_por_risco(dataset_v2):
    """AuC exposto é métrica calculada (saldo x risco), não treinada — verifica fórmula."""
    _, _, df_v2 = dataset_v2
    esperado = (df_v2["saldo_bi"] * df_v2["risco_saida_assessor"]).round(4)
    assert (df_v2["auc_exposto"] == esperado).all()


def test_auc_exposto_agregado_dentro_da_faixa_de_mercado(dataset_v2):
    """
    Pesquisa de mercado (ADR-0001): 10-15% dos clientes/receita em risco
    a qualquer momento. AuC exposto agregado deve refletir essa ordem
    de grandeza, não ser trivialmente 0% ou >50%.
    """
    _, _, df_v2 = dataset_v2
    pct_exposto = df_v2["auc_exposto"].sum() / df_v2["saldo_bi"].sum()
    assert 0.05 <= pct_exposto <= 0.25, (
        f"AuC exposto agregado = {pct_exposto*100:.1f}% — fora da faixa "
        f"plausível de mercado (5%-25%)"
    )


# ── Contrato de schema: assessor_id sempre presente e válido ──

def test_todo_cliente_tem_assessor_atribuido(dataset_v2):
    _, df_adv, df_v2 = dataset_v2
    assert df_v2["assessor_id"].notna().all()
    assert set(df_v2["assessor_id"]).issubset(set(df_adv["assessor_id"]))


def test_qtd_clientes_carteira_bate_com_atribuicao_real(dataset_v2):
    """
    Regressão do achado de EDA: qtd_clientes_carteira já foi um campo
    declarado ANTES da atribuição real (média 17,9 vs. 4,1 real) — uma
    promessa que os dados não cumpriam. Agora é calculado pós-atribuição;
    este teste garante que a divergência não volte.
    """
    _, df_adv, df_v2 = dataset_v2
    contagem_real = df_v2["assessor_id"].value_counts()
    for assessor_id, declarado in df_adv.set_index("assessor_id")["qtd_clientes_carteira"].items():
        real = contagem_real.get(assessor_id, 0)
        assert declarado == real, (
            f"{assessor_id}: qtd_clientes_carteira declarado={declarado} "
            f"mas atribuição real={real}"
        )
