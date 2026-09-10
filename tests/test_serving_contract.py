"""Trava o contrato de scoring v2 (serving_contract.py, ADR-0003) como fonte única.

Se qualquer cópia de FEATURES_V2_BASE, da lista de segmentos ou das regras de
risco/fluxo ressurgir e divergir, um destes testes quebra.
"""
import os

import pandas as pd
import pytest

from src import serving_contract as sc


def test_features_v2_base_bate_com_o_treino():
    from src.model_training.nodes import FEATURES_V2_BASE as treino
    assert sc.FEATURES_V2_BASE == treino
    assert sc.FEATURES_V2_BASE[0] == "segmento"          # ordem canônica
    assert len(sc.FEATURES_V2_BASE) == len(set(sc.FEATURES_V2_BASE))


def test_features_v2_base_bate_com_api():
    from src import api
    assert api.FEATURES_V2_BASE is sc.FEATURES_V2_BASE

    # shap_analysis_v2 roda no import (script), então só confere o fonte: não redefine a lista
    raiz = os.path.dirname(os.path.dirname(__file__))
    txt = open(os.path.join(raiz, "src", "shap_analysis_v2.py"), encoding="utf-8").read()
    assert "from src.serving_contract import FEATURES_V2_BASE" in txt
    assert "FEATURES_V2_BASE = [" not in txt


def test_features_v2_base_e_o_que_o_modelo_consome():
    import joblib
    modelo = joblib.load("output/models/gb_pipeline_v2.pkl")
    df = pd.read_csv("output/data/base_clientes_v2_limpo.csv").head(5)
    # não levanta = as 11 colunas na ordem canônica são exatamente o que o Pipeline espera
    modelo.predict_proba(df[sc.FEATURES_V2_BASE])


def test_load_threshold_map_cobre_os_quatro_segmentos():
    mapa = sc.load_threshold_map()
    assert set(mapa) == set(sc.SEGMENTOS_VALIDOS)


def test_load_threshold_map_rejeita_csv_incompleto(tmp_path):
    parcial = tmp_path / "thr.csv"
    parcial.write_text("segmento,threshold\nWealth,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        sc.load_threshold_map(str(parcial))

    faltando_col = tmp_path / "thr2.csv"
    faltando_col.write_text("segmento\nWealth\n", encoding="utf-8")
    with pytest.raises(ValueError):
        sc.load_threshold_map(str(faltando_col))

    with pytest.raises(FileNotFoundError):
        sc.load_threshold_map(str(tmp_path / "nao_existe.csv"))


def test_risk_level_usa_o_threshold_do_segmento():
    thr = {"Alta Renda": 0.10, "Private": 0.07, "Wealth": 0.10, "Family Office": 0.10}
    assert sc.risk_level(0.20, "Private", thr) == "ALTO"        # >= 0.07
    assert sc.risk_level(0.05, "Private", thr) == "MEDIO"       # >= 0.07 * 0.6
    assert sc.risk_level(0.01, "Private", thr) == "BAIXO"
    with pytest.raises(KeyError):                               # segmento inválido é bug, não 0.5
        sc.risk_level(0.5, "Varejo", thr)


def test_operational_flow_e_needs_human_review():
    assert sc.needs_human_review("Wealth", 10) is True
    assert sc.needs_human_review("Family Office", 10) is True
    assert sc.needs_human_review("Private", 300) is True        # AuC >= 250
    assert sc.needs_human_review("Private", 50) is False
    assert sc.operational_flow("Private", 50) == "AUTO → CRM"
    assert sc.operational_flow("Wealth", 50) == "REVISAO_HUMANA (especialista)"


def test_auc_at_risk_mm():
    assert sc.auc_at_risk_mm(100.0, 0.5) == 15.0               # 100 * 0.30 * 0.5
    assert sc.auc_at_risk_mm(0.0, 0.9) == 0.0


def test_risk_factors_dispara_os_codigos_certos():
    base = {
        "retorno_12m_pct": 12.0, "dias_desde_ultimo_contato": 5.0,
        "variacao_freq_contato_3m": 0.0, "tempo_resposta_medio_horas": 8.0,
        "freq_contato_mes": 3, "qtd_produtos": 4, "auc_milhoes": 120.0,
    }
    assert sc.risk_factors(base) == []

    critico = {**base, "retorno_12m_pct": 4.0, "dias_desde_ultimo_contato": 70.0,
               "variacao_freq_contato_3m": -0.45, "tempo_resposta_medio_horas": 60.0,
               "qtd_produtos": 1, "auc_milhoes": 8.0}
    assert sc.risk_factors(critico) == [
        "retorno_baixo", "sem_contato_recente", "cadencia_caindo",
        "resposta_lenta", "monoproduto", "auc_baixo",
    ]

    # nulo estrutural não dispara retorno_baixo / resposta_lenta
    nulos = {**base, "retorno_12m_pct": None, "tempo_resposta_medio_horas": None}
    assert "retorno_baixo" not in sc.risk_factors(nulos)
    assert "resposta_lenta" not in sc.risk_factors(nulos)

    # freq_contato_mes == 0 dispara sem_contato_recente mesmo com dias baixo
    assert "sem_contato_recente" in sc.risk_factors({**base, "freq_contato_mes": 0})


def test_nenhuma_copia_do_contrato_nos_consumidores():
    """grep de regressão: as regras não podem ser redefinidas fora de serving_contract."""
    raiz = os.path.dirname(os.path.dirname(__file__))
    proibido = ("* 0.30 *", "* 0.6 else", 'in {"Wealth", "Family Office"} or')
    for arq in ("api.py", "app.py", "agent.py", "monitor.py"):
        txt = open(os.path.join(raiz, "src", arq), encoding="utf-8").read()
        for padrao in proibido:
            assert padrao not in txt, f"{arq} reimplementa regra do contrato: {padrao!r}"

    # os limiares de fator de risco vêm do contrato, não reescritos na API
    api_txt = open(os.path.join(raiz, "src", "api.py"), encoding="utf-8").read()
    for lit in ("< 9.0", "< -0.2", "> 45"):
        assert lit not in api_txt, f"api.py reescreve limiar de risk_factors: {lit!r}"
