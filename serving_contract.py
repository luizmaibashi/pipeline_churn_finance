# =============================================================
# serving_contract.py — Fonte única do contrato de scoring v2
#
# Toda regra que decide "como pontuar um cliente" vive aqui: a lista de
# features na ordem canônica, os segmentos válidos, o carregamento dos
# thresholds calibrados, a regra de nível de risco, o fluxo operacional e o
# fator de perda de AuC no churn.
#
# Consumido por api.py, app.py, agent.py, monitor.py, shap_analysis_v2.py e
# src/model_training/nodes.py. Decisão registrada em docs/adr/0003.
#
# Dependência única: pandas (agent.py roda em modo demo sem nada de ML;
# importar de src/ arrastaria sklearn para cá). Fica na raiz de propósito.
# =============================================================

from __future__ import annotations

import os

import pandas as pd

# ── Constantes do contrato ───────────────────────────────────
SEGMENTOS_VALIDOS = ["Alta Renda", "Private", "Wealth", "Family Office"]

# Ordem canônica das colunas cruas que o Pipeline v2 consome. Deve bater com
# src.model_training.nodes.FEATURES_V2_BASE (assert no import daquele módulo)
# e com o que api.py/shap_analysis_v2.py passam a predict_proba
# (tests/test_serving_contract.py trava a igualdade).
FEATURES_V2_BASE = [
    "segmento", "meses_cliente", "qtd_produtos",
    "retorno_12m_pct", "freq_contato_mes", "auc_milhoes",
    "dias_desde_ultimo_contato", "variacao_freq_contato_3m",
    "tempo_resposta_medio_horas",
    "sem_historico_12m", "cliente_novo_sem_contato_hist",
]

PCT_AUC_LOSS_ON_CHURN = 0.30   # queda de AuC que caracteriza churn (PROBLEM.md v2.0 §2)
RISK_MID_FACTOR       = 0.6    # fronteira MÉDIO = fração do threshold ALTO do segmento
HUMAN_REVIEW_AUC_MM   = 250    # AuC (R$ milhões) acima do qual o fluxo vai a especialista

# ── Limiares dos fatores de risco (antes copiados em api.py e app.py) ─────────
RETORNO_12M_BAIXO_PCT     = 9.0   # retorno de 12m abaixo disso conta como fator
DIAS_SEM_CONTATO_ALERTA   = 45    # dias desde o último contato acima disso: alerta
QUEDA_CADENCIA_ALERTA     = -0.2  # variação de cadência abaixo disso: cadência caindo
TEMPO_RESPOSTA_ALERTA_H   = 40    # latência de resposta acima disso: engajamento em queda
AUC_FIDELIZACAO_MIN_MM    = 15    # AuC abaixo disso: candidato a campanha de fidelização
QTD_PRODUTOS_MONOPRODUTO  = 1     # cliente monoproduto

THRESHOLDS_CSV = os.path.join("output", "data", "thresholds_v2.csv")


# ── Thresholds calibrados ────────────────────────────────────
def load_threshold_map(path: str = THRESHOLDS_CSV) -> dict[str, float]:
    """{segmento: threshold} a partir de thresholds_v2.csv (gerado pelo pipeline).

    Levanta se o arquivo não existir ou não cobrir exatamente os 4 segmentos —
    nunca devolve um mapa parcial em silêncio (a v2 removeu de propósito
    qualquer corte de risco embutido no código).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Thresholds v2 não encontrados em {path}. Execute 'python pipeline.py'."
        )
    df = pd.read_csv(path)
    faltando = {"segmento", "threshold"} - set(df.columns)
    if faltando:
        raise ValueError(f"{path} sem colunas obrigatórias: {faltando}")
    mapa = {str(s): float(t) for s, t in zip(df["segmento"], df["threshold"])}
    if set(mapa) != set(SEGMENTOS_VALIDOS):
        raise ValueError(f"{path} não cobre exatamente os segmentos da v2: {SEGMENTOS_VALIDOS}")
    return mapa


# ── Regras de decisão ────────────────────────────────────────
def risk_level(prob: float, segmento: str, thr_map: dict[str, float]) -> str:
    """ALTO / MEDIO / BAIXO contra o threshold calibrado do segmento.

    `segmento` já vem validado contra SEGMENTOS_VALIDOS pelos chamadores; um
    KeyError aqui é bug, não caminho normal.
    """
    thr = thr_map[segmento]
    if prob >= thr:
        return "ALTO"
    if prob >= thr * RISK_MID_FACTOR:
        return "MEDIO"
    return "BAIXO"


def needs_human_review(segmento: str, auc_milhoes: float) -> bool:
    """Decisão booleana de encaminhar para especialista (PROBLEM.md v2.0 §5)."""
    return segmento in {"Wealth", "Family Office"} or auc_milhoes >= HUMAN_REVIEW_AUC_MM


def operational_flow(segmento: str, auc_milhoes: float) -> str:
    """Forma de contrato do fluxo — string que a API devolve no JSON.
    O dashboard formata seu próprio rótulo a partir de needs_human_review()."""
    return "REVISAO_HUMANA (especialista)" if needs_human_review(segmento, auc_milhoes) else "AUTO → CRM"


def auc_at_risk_mm(auc_milhoes: float, prob: float) -> float:
    """Receita/AuC anual estimada em risco: AuC × queda-de-churn × probabilidade."""
    return round(auc_milhoes * PCT_AUC_LOSS_ON_CHURN * prob, 2)


def risk_factors(features: dict) -> list[str]:
    """Códigos canônicos dos fatores de risco disparados pelo perfil do cliente.

    Fonte única dos limiares que `api._recommended_action` e o painel de insights
    do `app.py` mantinham em cópia. Cada consumidor formata o código à sua
    maneira (ação recomendada na API, alerta no dashboard).

    Códigos: retorno_baixo, sem_contato_recente, cadencia_caindo, resposta_lenta,
    monoproduto, auc_baixo.
    """
    f = features
    retorno  = f.get("retorno_12m_pct")
    dias     = f.get("dias_desde_ultimo_contato")
    variacao = f.get("variacao_freq_contato_3m")
    resposta = f.get("tempo_resposta_medio_horas")

    fatores: list[str] = []
    if retorno is not None and retorno < RETORNO_12M_BAIXO_PCT:
        fatores.append("retorno_baixo")
    if f.get("freq_contato_mes", 99) == 0 or (dias is not None and dias > DIAS_SEM_CONTATO_ALERTA):
        fatores.append("sem_contato_recente")
    if variacao is not None and variacao < QUEDA_CADENCIA_ALERTA:
        fatores.append("cadencia_caindo")
    if resposta is not None and resposta > TEMPO_RESPOSTA_ALERTA_H:
        fatores.append("resposta_lenta")
    if f.get("qtd_produtos", 99) == QTD_PRODUTOS_MONOPRODUTO:
        fatores.append("monoproduto")
    if f.get("auc_milhoes", 99) < AUC_FIDELIZACAO_MIN_MM:
        fatores.append("auc_baixo")
    return fatores
