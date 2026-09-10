# =============================================================
# app.py — Dashboard Streamlit: Churn Finance Pipeline
# Uso: streamlit run app.py
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

from transformers import FeatureEngineer, StructuralNullImputer  # noqa: F401 — unpickle do Pipeline v2
from serving_contract import (
    SEGMENTOS_VALIDOS as SEGMENTOS_V2, FEATURES_V2_BASE, RISK_MID_FACTOR,
    load_threshold_map, risk_level, needs_human_review, auc_at_risk_mm,
    DIAS_SEM_CONTATO_ALERTA, QUEDA_CADENCIA_ALERTA, TEMPO_RESPOSTA_ALERTA_H,
    QTD_PRODUTOS_MONOPRODUTO,
)

# ── Configuração da página ────────────────────────────────────
st.set_page_config(
    page_title="Churn Finance | Dashboard ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS customizado ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background: #0f1117;
}

/* Cards de métrica */
.metric-card {
    background: linear-gradient(135deg, #1e2130 0%, #252a3d 100%);
    border: 1px solid #2d3250;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

.metric-label {
    color: #8b95b0;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.metric-value {
    color: #e8ecf4;
    font-size: 28px;
    font-weight: 700;
    line-height: 1.2;
}

.metric-delta-pos {
    color: #51cf66;
    font-size: 13px;
    font-weight: 500;
}

.metric-delta-neg {
    color: #ff6b6b;
    font-size: 13px;
    font-weight: 500;
}

/* Badge de risco */
.badge-alto {
    background: linear-gradient(135deg, #c92a2a, #e03131);
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    display: inline-block;
}

.badge-medio {
    background: linear-gradient(135deg, #e67700, #f59f00);
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    display: inline-block;
}

.badge-baixo {
    background: linear-gradient(135deg, #2b8a3e, #37b24d);
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    display: inline-block;
}

/* Seção header */
.section-header {
    color: #e8ecf4;
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 4px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2d3250;
}

.section-sub {
    color: #8b95b0;
    font-size: 13px;
    margin-bottom: 20px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141824 0%, #1a1f2e 100%);
    border-right: 1px solid #2d3250;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #1a1f2e;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    color: #8b95b0;
    border-radius: 8px;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: #2d3250 !important;
    color: #e8ecf4 !important;
}

/* Alert box */
.alert-box {
    background: rgba(255, 107, 107, 0.08);
    border: 1px solid rgba(255, 107, 107, 0.3);
    border-left: 4px solid #ff6b6b;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 12px 0;
}

.alert-box-green {
    background: rgba(81, 207, 102, 0.08);
    border: 1px solid rgba(81, 207, 102, 0.3);
    border-left: 4px solid #51cf66;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 12px 0;
}

/* Números de destaque */
.kpi-row {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
}

.insight-box {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    color: #c1c8de;
    line-height: 1.6;
}

/* Plotly dark override */
.js-plotly-plot .plotly .modebar {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────
FEAT_LABELS = {
    "variacao_freq_contato_3m":   "Variação de cadência de contato (3m)",
    "dias_desde_ultimo_contato":  "Dias desde o último contato",
    "tempo_resposta_medio_horas": "Latência de resposta do cliente (h)",
    "auc_milhoes":                "AuC sob custódia (R$ mi)",
    "retorno_12m_pct":            "Retorno 12m (%)",
    "meses_cliente":              "Meses como Cliente",
    "retorno_relativo":           "Retorno Relativo à Média",
    "engajamento_score":          "Score de Engajamento",
    "intensidade_rel":            "Intensidade Relacionamento",
    "qtd_produtos":               "Qtd. Produtos",
    "freq_contato_mes":           "Freq. Contato/Mês",
    "segmento_enc":               "Segmento",
    "flag_risco":                 "Flag de Risco",
    "sem_historico_12m":          "Sem histórico de 12m",
    "cliente_novo_sem_contato_hist": "Cliente novo (sem histórico de contato)",
}

SEG_COLORS = {
    "Alta Renda":    "#ffd43b",
    "Private":       "#74c0fc",
    "Wealth":        "#63e6be",
    "Family Office": "#f783ac",
}


@st.cache_data
def load_thresholds_table():
    """Tabela completa de thresholds v2 (reports/thresholds_v2.md) — lida uma vez."""
    return pd.read_csv("output/data/thresholds_v2.csv")


@st.cache_data
def load_thresholds_v2():
    """{segmento: threshold} calibrado por segmento — contrato de serving (ADR-0003)."""
    return load_threshold_map()


_RISCO_PT = {"ALTO": "Alto", "MEDIO": "Médio", "BAIXO": "Baixo"}


def risco_por_threshold(prob, segmento, thr_map):
    """Rótulo PT-BR do nível de risco do contrato de serving (ALTO/MEDIO/BAIXO)."""
    return _RISCO_PT[risk_level(prob, segmento, thr_map)]


def band_color(prob, thr, high="#f03e3e", mid="#ffd43b", low="#51cf66"):
    """Cor da faixa de risco para o mesmo corte ALTO/MÉDIO/BAIXO."""
    return high if prob >= thr else mid if prob >= thr * RISK_MID_FACTOR else low

PLOTLY_DARK = dict(
    paper_bgcolor="#1a1f2e",
    plot_bgcolor="#1a1f2e",
    font=dict(color="#c1c8de", family="Inter"),
    margin=dict(t=60, b=40, l=50, r=20),
)


@st.cache_resource
def load_artifacts():
    """Carrega o Pipeline v2 (early-warning comportamental, ADR-0001)."""
    model  = joblib.load("output/models/gb_pipeline_v2.pkl")
    return model


@st.cache_data
def load_data():
    """Carrega os CSVs v2 gerados pelo pipeline."""
    df_cli = pd.read_csv("output/data/base_clientes_v2_limpo.csv")
    imp    = pd.read_csv("output/data/feature_importance_v2.csv")
    cmp    = pd.read_csv("output/data/comparacao_v1_v2.csv")
    cv     = pd.read_csv("output/data/cv_scores_v2.csv")
    return df_cli, imp, cmp, cv



def risk_badge(prob, thr=0.5):
    if prob >= thr:
        return '<span class="badge-alto">🔴 Alto Risco</span>'
    elif prob >= thr * RISK_MID_FACTOR:
        return '<span class="badge-medio">🟡 Médio Risco</span>'
    else:
        return '<span class="badge-baixo">🟢 Baixo Risco</span>'


def gauge_chart(prob, thr=0.5):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 38, "color": "#e8ecf4"}},
        delta={"reference": 20, "increasing": {"color": "#ff6b6b"},
               "decreasing": {"color": "#51cf66"}},
        gauge={
            "axis":    {"range": [0, 100], "tickcolor": "#8b95b0", "tickfont": {"color": "#8b95b0"}},
            "bar":     {"color": band_color(prob, thr), "thickness": 0.28},
            "bgcolor": "#252a3d",
            "bordercolor": "#2d3250",
            "steps": [
                {"range": [0,  thr * RISK_MID_FACTOR * 100],  "color": "rgba(81,207,102,0.08)"},
                {"range": [thr * RISK_MID_FACTOR * 100, thr * 100],  "color": "rgba(255,212,59,0.08)"},
                {"range": [thr * 100, 100], "color": "rgba(240,62,62,0.08)"},
            ],
            "threshold": {"line": {"color": "#a9b4d0", "width": 2}, "value": thr * 100}
        },
        title={"text": f"Prob. de churn · corte do segmento {thr*100:.0f}%", "font": {"size": 13, "color": "#8b95b0"}}
    ))
    fig.update_layout(height=280, paper_bgcolor="#1a1f2e", margin=dict(t=20, b=20, l=40, r=40))
    return fig


# ── Verificação de artefatos ──────────────────────────────────
artifacts_ok = all(os.path.exists(p) for p in [
    "output/models/gb_pipeline_v2.pkl",
    "output/data/base_clientes_v2_limpo.csv",
    "output/data/feature_importance_v2.csv",
    "output/data/comparacao_v1_v2.csv",
    "output/data/cv_scores_v2.csv",
    "output/data/thresholds_v2.csv",
])

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:40px;'>📊</div>
        <div style='font-size:18px; font-weight:700; color:#e8ecf4; margin-top:8px;'>Churn Finance</div>
        <div style='font-size:12px; color:#8b95b0; margin-top:4px;'>ML Pipeline — Tech Challenge</div>
    </div>
    <hr style='border-color:#2d3250; margin: 16px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("<div style='color:#8b95b0; font-size:11px; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;'>Status dos Artefatos</div>", unsafe_allow_html=True)

    checks = {
        "Pipeline v2":       "output/models/gb_pipeline_v2.pkl",
        "Base v2 (limpa)":   "output/data/base_clientes_v2_limpo.csv",
        "Feature Importance v2": "output/data/feature_importance_v2.csv",
        "Thresholds v2":     "output/data/thresholds_v2.csv",
    }
    for name, path in checks.items():
        ok = os.path.exists(path)
        st.markdown(
            f"<div style='font-size:13px; color:{'#51cf66' if ok else '#ff6b6b'}; padding:2px 0;'>"
            f"{'✅' if ok else '❌'} {name}</div>",
            unsafe_allow_html=True
        )

    if not artifacts_ok:
        st.markdown("""
        <div style='background:rgba(255,107,107,0.12); border:1px solid rgba(255,107,107,0.3);
                    border-radius:8px; padding:12px; margin-top:12px; font-size:12px; color:#ffa8a8;'>
            ⚠️ Artefatos ausentes.<br>Execute primeiro:<br>
            <code style='background:#0f1117; padding:2px 6px; border-radius:4px;'>python pipeline.py</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2d3250; margin:20px 0 12px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#8b95b0; font-size:11px; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;'>Contexto do Negócio</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:12px; color:#c1c8de; line-height:1.8;'>
        🏢 Gestora de wealth (private banking / multi-family office)<br>
        💰 ~R$76 bi de AuC gerado<br>
        👥 ~1.200 grupos econômicos · ~50 assessores<br>
        📈 Alta Renda · Private · Wealth · Family Office<br>
        ⚠️ ~11,7% de churn no dado sintético
    </div>
    <div style='font-size:11px; color:#6b7590; margin-top:8px; line-height:1.6;'>
        Dado 100% sintético e calibrado ao mercado (ANBIMA). Métricas medem a
        coerência do pipeline, não desempenho em produção.
    </div>
    """, unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div style='padding: 8px 0 24px 0;'>
    <h1 style='color:#e8ecf4; font-size:30px; font-weight:700; margin:0; line-height:1.2;'>
        Pipeline de Predição de Churn
    </h1>
    <p style='color:#8b95b0; margin:6px 0 0 0; font-size:14px;'>
        Gestora de wealth — early-warning comportamental (v2, ADR-0001) sobre Gradient Boosting
    </p>
</div>
""", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Predição Individual",
    "📈  Análise da Carteira",
    "🔬  Performance do Modelo",
    "🧭  Carteira Exposta por Assessor"
])


# ==============================================================
# TAB 1 — PREDIÇÃO INDIVIDUAL
# ==============================================================
with tab1:
    st.markdown('<p class="section-header">Preditor de Churn — Cliente Individual</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Perfil do cliente + sinais de <b>early-warning comportamental</b> (v2). O corte de risco é o threshold calibrado do segmento (<code>reports/thresholds_v2.md</code>).</p>', unsafe_allow_html=True)

    if not artifacts_ok:
        st.error("⚠️ Artefatos de modelo não encontrados. Execute `python pipeline.py` primeiro.")
    else:
        model = load_artifacts()
        df_cli, _, _, _ = load_data()
        thr_map = load_thresholds_v2()
        media_retorno = df_cli["retorno_12m_pct"].mean()

        col_form, col_result = st.columns([1, 1], gap="large")

        with col_form:
            st.markdown("<div style='color:#8b95b0; font-size:12px; font-weight:600; letter-spacing:0.8px; text-transform:uppercase; margin-bottom:14px;'>Perfil do Cliente</div>", unsafe_allow_html=True)

            with st.container():
                segmento = st.selectbox("Segmento", SEGMENTOS_V2, key="pred_segmento")
                col_a, col_b = st.columns(2)
                with col_a:
                    meses = st.slider("Tempo como cliente (meses)", 6, 360, 48, key="pred_meses")
                    qtd_prod = st.slider("Nº de produtos", 1, 12, 4, key="pred_qtd")
                    freq = st.slider("Contatos no último mês", 0, 15, 3, key="pred_freq")
                with col_b:
                    auc_milhoes = st.slider("AuC sob custódia (R$ mi)", 3.0, 2000.0, 120.0, 1.0,
                                            key="pred_auc", format="R$ %.0f mi")
                    tem_retorno = st.checkbox("Tem histórico de retorno 12m", value=True, key="pred_tem_ret")
                    retorno = st.slider("Retorno 12m (%)", -20.0, 40.0, 11.5, 0.1, key="pred_retorno",
                                        disabled=not tem_retorno)

                st.markdown("<div style='color:#8b95b0; font-size:11px; font-weight:600; letter-spacing:0.8px; text-transform:uppercase; margin:10px 0 4px 0;'>Early-warning comportamental</div>", unsafe_allow_html=True)
                col_c, col_d = st.columns(2)
                with col_c:
                    tem_contato = st.checkbox("Tem histórico de contato", value=True, key="pred_tem_cont")
                    dias_contato = st.slider("Dias desde o último contato", 0, 200, 15, key="pred_dias",
                                             disabled=not tem_contato)
                with col_d:
                    variacao = st.slider("Variação de cadência 3m (−0,3 = caiu 30%)", -0.9, 0.9, -0.02, 0.01,
                                         key="pred_var")
                    resposta = st.slider("Latência de resposta do cliente (h)", 0.0, 120.0, 16.0, 0.5,
                                         key="pred_resp")

                retorno_val = retorno if tem_retorno else None
                dias_val = dias_contato if tem_contato else None

                X_new = pd.DataFrame([{
                    "segmento": segmento,
                    "meses_cliente": meses,
                    "qtd_produtos": qtd_prod,
                    "retorno_12m_pct": retorno_val,
                    "freq_contato_mes": freq,
                    "auc_milhoes": auc_milhoes,
                    "dias_desde_ultimo_contato": dias_val,
                    "variacao_freq_contato_3m": variacao,
                    "tempo_resposta_medio_horas": resposta,
                    "sem_historico_12m": int(retorno_val is None),
                    "cliente_novo_sem_contato_hist": int(dias_val is None),
                }])[FEATURES_V2_BASE]
                prob = model.predict_proba(X_new)[0][1]
                thr = thr_map[segmento]   # contrato garante os 4 segmentos; sem fallback silencioso

        with col_result:
            st.markdown("<div style='color:#8b95b0; font-size:12px; font-weight:600; letter-spacing:0.8px; text-transform:uppercase; margin-bottom:14px;'>Resultado da Predição</div>", unsafe_allow_html=True)

            st.plotly_chart(gauge_chart(prob, thr), use_container_width=True, key="gauge")

            st.markdown(
                f"<div style='text-align:center; margin: -10px 0 16px 0;'>{risk_badge(prob, thr)}</div>",
                unsafe_allow_html=True
            )

            insights = []
            if dias_val is not None and dias_val > DIAS_SEM_CONTATO_ALERTA:
                insights.append(f"⚠️ {dias_val} dias sem contato — sinal antecedente de deterioração")
            if variacao < QUEDA_CADENCIA_ALERTA:
                insights.append(f"⚠️ Cadência de contato caindo {abs(variacao)*100:.0f}% — early-warning")
            if resposta > TEMPO_RESPOSTA_ALERTA_H:
                insights.append("⚠️ Latência de resposta alta — engajamento em queda")
            if freq == 0:
                insights.append("⚠️ Nenhum contato no último mês")
            if tem_retorno and retorno < media_retorno:
                insights.append(f"⚠️ Retorno abaixo da média da carteira ({media_retorno:.1f}%)")
            if qtd_prod == QTD_PRODUTOS_MONOPRODUTO:
                insights.append("⚠️ Monoproduto — menor fidelização")

            if insights:
                st.markdown("<div class='alert-box'><b style='color:#ffa8a8; font-size:13px;'>Fatores de Risco Identificados</b><br><div style='font-size:12px; color:#ffcdd2; line-height:2; margin-top:6px;'>" + "<br>".join(insights) + "</div></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='alert-box-green'><b style='color:#a9e34b; font-size:13px;'>✅ Perfil de baixo risco</b><br><span style='font-size:12px; color:#cbf078;'>Nenhum fator de risco crítico identificado para este cliente.</span></div>", unsafe_allow_html=True)

            auc_risco_mm = auc_at_risk_mm(auc_milhoes, prob)
            fluxo = "Revisão humana (especialista)" if needs_human_review(segmento, auc_milhoes) else "Auto → CRM"
            st.markdown(f"""
            <div class='insight-box' style='margin-top:12px;'>
                <div style='font-weight:600; color:#e8ecf4; margin-bottom:6px;'>💼 AuC em risco</div>
                <div>AuC sob custódia: <b style='color:#74c0fc;'>R$ {auc_milhoes:.0f} mi</b></div>
                <div>AuC em risco: <b style='color:{band_color(prob, thr, high="#ff6b6b")};'>R$ {auc_risco_mm:.1f} mi</b>
                    <span style='font-size:11px; color:#6b7590;'>(AuC × 30% × prob)</span></div>
                <div style='margin-top:6px; font-size:11px; color:#6b7590;'>Fluxo operacional: {fluxo}</div>
            </div>
            """, unsafe_allow_html=True)

        # Perfis rápidos
        st.markdown("<hr style='border-color:#2d3250; margin:28px 0 16px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='color:#8b95b0; font-size:12px; font-weight:600; letter-spacing:0.8px; text-transform:uppercase; margin-bottom:14px;'>Simule Perfis Pré-definidos</div>", unsafe_allow_html=True)

        p1, p2, p3 = st.columns(3)
        profiles = [
            {
                "label": "🔴 Alto Risco (early-warning)",
                "desc": "Private | 30m | 3 prod | 70 dias sem contato | cadência −45% | resposta 60h | R$25 mi",
                "color": "#ff6b6b",
                "vals": {"segmento":"Private", "meses_cliente":30, "qtd_produtos":3, "retorno_12m_pct":6.0,
                         "freq_contato_mes":0, "auc_milhoes":25.0, "dias_desde_ultimo_contato":70.0,
                         "variacao_freq_contato_3m":-0.45, "tempo_resposta_medio_horas":60.0,
                         "sem_historico_12m":0, "cliente_novo_sem_contato_hist":0},
            },
            {
                "label": "🟡 Médio Risco",
                "desc": "Alta Renda | 24m | 3 prod | 30 dias sem contato | cadência −15% | R$8 mi",
                "color": "#ffd43b",
                "vals": {"segmento":"Alta Renda", "meses_cliente":24, "qtd_produtos":3, "retorno_12m_pct":9.0,
                         "freq_contato_mes":1, "auc_milhoes":8.0, "dias_desde_ultimo_contato":30.0,
                         "variacao_freq_contato_3m":-0.15, "tempo_resposta_medio_horas":24.0,
                         "sem_historico_12m":0, "cliente_novo_sem_contato_hist":0},
            },
            {
                "label": "🟢 Baixo Risco",
                "desc": "Wealth | 96m | 6 prod | contato recente | cadência estável | R$150 mi",
                "color": "#51cf66",
                "vals": {"segmento":"Wealth", "meses_cliente":96, "qtd_produtos":6, "retorno_12m_pct":14.0,
                         "freq_contato_mes":5, "auc_milhoes":150.0, "dias_desde_ultimo_contato":6.0,
                         "variacao_freq_contato_3m":0.05, "tempo_resposta_medio_horas":8.0,
                         "sem_historico_12m":0, "cliente_novo_sem_contato_hist":0},
            },
        ]
        probs_perfis = model.predict_proba(
            pd.DataFrame([p["vals"] for p in profiles])[FEATURES_V2_BASE]
        )[:, 1]
        for col, prof, p_val in zip([p1, p2, p3], profiles, probs_perfis):
            with col:
                st.markdown(f"""
                <div style='background:#1e2130; border:1px solid #2d3250; border-left:3px solid {prof["color"]};
                            border-radius:10px; padding:16px; height:130px;'>
                    <div style='font-weight:700; color:{prof["color"]}; font-size:13px;'>{prof["label"]}</div>
                    <div style='font-size:11px; color:#8b95b0; margin:6px 0; line-height:1.6;'>{prof["desc"]}</div>
                    <div style='font-size:20px; font-weight:700; color:#e8ecf4;'>{p_val*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================
# TAB 2 — ANÁLISE DA CARTEIRA
# ==============================================================
with tab2:
    st.markdown('<p class="section-header">Análise da Carteira de Clientes</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Visão geral da base de 1.200 clientes com distribuição de risco e perfil de churn por segmento.</p>', unsafe_allow_html=True)

    if not artifacts_ok:
        st.error("⚠️ Dados não encontrados. Execute `python pipeline.py` primeiro.")
    else:
        model = load_artifacts()
        df_cli, imp, cmp, cv_df = load_data()
        thr_map = load_thresholds_v2()

        # Scoring ao vivo da carteira v2 com as features de early-warning
        df_plot = df_cli.copy()
        df_plot["prob_churn"] = model.predict_proba(df_plot[FEATURES_V2_BASE])[:, 1]
        df_plot["risco"] = [
            risco_por_threshold(p, s, thr_map)
            for p, s in zip(df_plot["prob_churn"], df_plot["segmento"])
        ]

        # KPIs
        total = len(df_plot)
        alto_risco = (df_plot["risco"] == "Alto").sum()
        medio_risco = (df_plot["risco"] == "Médio").sum()
        churn_real = df_plot["churn"].sum()
        auc_risco_bi = df_plot.loc[df_plot["risco"] == "Alto", "auc_milhoes"].sum() / 1000

        k1, k2, k3, k4 = st.columns(4)
        kpis = [
            (k1, "Total de relações", f"{total:,}", ""),
            (k2, "Alto Risco (modelo)", f"{alto_risco}", f"{alto_risco/total*100:.1f}% da carteira"),
            (k3, "Churn Real (base)", f"{churn_real}", f"{churn_real/total*100:.1f}% da base"),
            (k4, "AuC em relações de alto risco", f"R${auc_risco_bi:.1f}bi", "AuC sob custódia"),
        ]
        colors = ["#74c0fc", "#ff6b6b", "#ffd43b", "#f03e3e"]
        for col, (label, value, delta) in zip([k1, k2, k3, k4], [(k[1], k[2], k[3]) for k in kpis]):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value'>{value}</div>
                    <div class='metric-delta-neg'>{delta}</div>
                </div>
                """, unsafe_allow_html=True)

        # Gráficos linha 1
        gc1, gc2 = st.columns([1, 1], gap="medium")

        with gc1:
            # Churn por segmento
            seg_stats = (
                df_plot.groupby("segmento")["churn"]
                .agg(churned="sum", total="count", taxa="mean")
                .assign(taxa_pct=lambda x: (x["taxa"]*100).round(1))
                .drop(columns="taxa")
                .sort_values("taxa_pct", ascending=False)
                .reset_index()
            )
            media_geral = df_plot["churn"].mean() * 100
            clrs = [SEG_COLORS.get(s, "#8b95b0") for s in seg_stats["segmento"]]

            fig_seg = go.Figure(go.Bar(
                x=seg_stats["segmento"],
                y=seg_stats["taxa_pct"],
                text=[f"{v}%" for v in seg_stats["taxa_pct"]],
                textposition="outside",
                marker_color=clrs,
                marker_line_color="#2d3250",
                marker_line_width=1
            ))
            fig_seg.add_shape(type="line", x0=-0.5, x1=3.5,
                              y0=media_geral, y1=media_geral,
                              line=dict(color="#8b95b0", dash="dot", width=1.5))
            fig_seg.add_annotation(x=3.4, y=media_geral + 1.5,
                                   text=f"Média: {media_geral:.1f}%",
                                   showarrow=False, font=dict(color="#8b95b0", size=11), xanchor="right")
            fig_seg.update_layout(
                title="Taxa de Churn por Segmento",
                **PLOTLY_DARK,
                height=340,
                xaxis_title="Segmento",
                yaxis_title="Churn (%)",
                yaxis_range=[0, seg_stats["taxa_pct"].max() + 10]
            )
            st.plotly_chart(fig_seg, use_container_width=True, key="seg_chart")

        with gc2:
            # Distribuição de risco
            risco_counts = df_plot["risco"].value_counts().reindex(["Alto", "Médio", "Baixo"])
            fig_risco = go.Figure(go.Pie(
                labels=["🔴 Alto", "🟡 Médio", "🟢 Baixo"],
                values=risco_counts.values,
                hole=0.55,
                marker=dict(colors=["#f03e3e", "#ffd43b", "#51cf66"],
                            line=dict(color="#1a1f2e", width=2)),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Clientes: %{value}<br>%{percent}<extra></extra>"
            ))
            fig_risco.add_annotation(
                text=f"<b>{total}</b><br>clientes",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#e8ecf4")
            )
            fig_risco.update_layout(
                title="Distribuição de Risco (modelo)",
                **PLOTLY_DARK,
                height=340,
                showlegend=True,
                legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center")
            )
            st.plotly_chart(fig_risco, use_container_width=True, key="risco_pie")

        # Gráficos linha 2
        gc3, gc4 = st.columns([1.2, 0.8], gap="medium")

        with gc3:
            # Scatter: dias sem contato (early-warning) vs prob_churn
            fig_scatter = px.scatter(
                df_plot,
                x="dias_desde_ultimo_contato",
                y="prob_churn",
                color="segmento",
                color_discrete_map=SEG_COLORS,
                size="auc_milhoes",
                size_max=22,
                hover_data=["cliente_id", "variacao_freq_contato_3m", "auc_milhoes"],
                opacity=0.7,
                labels={"dias_desde_ultimo_contato": "Dias desde o último contato",
                        "prob_churn": "Prob. Churn", "segmento": "Segmento"}
            )
            fig_scatter.update_layout(
                title="Early-warning: dias sem contato vs prob. de churn (tamanho = AuC)",
                **PLOTLY_DARK,
                height=340,
                xaxis_title="Dias desde o último contato",
                yaxis_title="Prob. Churn"
            )
            st.plotly_chart(fig_scatter, use_container_width=True, key="scatter")

        with gc4:
            top_risco = (
                df_plot[df_plot["risco"] == "Alto"]
                .sort_values("prob_churn", ascending=False)
                .head(10)[["cliente_id", "segmento", "prob_churn", "auc_milhoes"]]
            )
            st.markdown("<div style='color:#8b95b0; font-size:12px; font-weight:600; letter-spacing:0.8px; text-transform:uppercase; margin-bottom:10px;'>🔴 Top 10 relações — Maior Risco</div>", unsafe_allow_html=True)
            if len(top_risco) > 0:
                top_risco_display = top_risco.copy()
                top_risco_display["prob_churn"] = top_risco_display["prob_churn"].apply(lambda x: f"{x*100:.1f}%")
                top_risco_display["auc_milhoes"] = top_risco_display["auc_milhoes"].apply(lambda x: f"R${x:.0f} mi")
                top_risco_display.columns = ["ID", "Segmento", "Prob. Churn", "AuC"]
                st.dataframe(
                    top_risco_display.reset_index(drop=True),
                    use_container_width=True,
                    height=280
                )
            else:
                st.info("Nenhum cliente com alto risco identificado.")


# ==============================================================
# TAB 3 — PERFORMANCE DO MODELO
# ==============================================================
with tab3:
    st.markdown('<p class="section-header">Performance e Avaliação do Modelo v2</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Early-warning comportamental (ADR-0001) contra a baseline reativa v1, no mesmo split. Dado sintético — as métricas medem a coerência do pipeline, não desempenho de produção. Números lidos de <code>comparacao_v1_v2.csv</code>, <code>feature_importance_v2.csv</code>, <code>cv_scores_v2.csv</code>.</p>', unsafe_allow_html=True)

    if not artifacts_ok:
        st.error("⚠️ Dados não encontrados. Execute `python pipeline.py` primeiro.")
    else:
        _, imp, cmp, cv_df = load_data()
        thr_map = load_thresholds_v2()
        thr_tbl = load_thresholds_table()

        v1 = cmp[cmp["modelo"] == "v1_baseline_reativa"].iloc[0]
        v2 = cmp[cmp["modelo"] == "v2_early_warning_advisor"].iloc[0]
        n_teste = int(v2["n_teste"])
        comport = ["variacao_freq_contato_3m", "dias_desde_ultimo_contato", "tempo_resposta_medio_horas"]
        imp_comport = imp[imp["feature"].isin(comport)]["importance"].sum()
        cv_mean = cv_df["recall_churn"].mean()
        cv_std  = cv_df["recall_churn"].std()
        n_seg = int((thr_tbl["origem_threshold"] == "segmento").sum())

        m1, m2, m3, m4 = st.columns(4)
        mets = [
            ("ROC-AUC v2", f"{v2['roc_auc']:.4f}", f"v1 baseline: {v1['roc_auc']:.4f}"),
            ("Recall churn v2", f"{v2['recall_churn']*100:.1f}%", f"split de teste n={n_teste}, ~28 eventos"),
            ("Importância comportamental", f"{imp_comport*100:.1f}%", "3 sinais de early-warning"),
            ("Thresholds calibrados", f"{n_seg}/4 por segmento", "resto: fallback global"),
        ]
        for col, (label, val, note) in zip([m1, m2, m3, m4], mets):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value' style='color:#74c0fc;'>{val}</div>
                    <div style='font-size:11px; color:#6b7590; margin-top:4px;'>{note}</div>
                </div>
                """, unsafe_allow_html=True)

        row1, row2 = st.columns([1, 1], gap="medium")

        with row1:
            metricas = ["recall_churn", "f1_churn", "roc_auc"]
            rotulos  = ["Recall (churn)", "F1 (churn)", "ROC-AUC"]
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name="v1 reativa", x=rotulos, y=[v1[m] for m in metricas],
                                     text=[f"{v1[m]:.3f}" for m in metricas], textposition="outside",
                                     marker_color="#8b95b0"))
            fig_cmp.add_trace(go.Bar(name="v2 early-warning", x=rotulos, y=[v2[m] for m in metricas],
                                     text=[f"{v2[m]:.3f}" for m in metricas], textposition="outside",
                                     marker_color="#63e6be"))
            fig_cmp.update_layout(title=f"v1 reativa vs v2 early-warning (mesmo split, n={n_teste})",
                                  barmode="group", **PLOTLY_DARK, height=360,
                                  legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
                                  yaxis_range=[0, 1.0], xaxis_title="Métrica", yaxis_title="Valor")
            st.plotly_chart(fig_cmp, use_container_width=True, key="cmp_chart")

        with row2:
            cv_mean_pct = cv_mean * 100
            fig_cv = go.Figure(go.Bar(
                x=cv_df["fold"], y=cv_df["recall_churn"] * 100,
                text=[f"{v*100:.1f}%" for v in cv_df["recall_churn"]], textposition="outside",
                marker_color="#74c0fc", marker_line_color="#4dabf7", marker_line_width=1.5,
            ))
            fig_cv.add_hline(y=cv_mean_pct, line_dash="dash", line_color="#1971c2",
                             annotation_text=f"Média={cv_mean_pct:.1f}% ±{cv_std*100:.1f} p.p.",
                             annotation_position="top left")
            fig_cv.update_layout(title="Cross-Validation 5-Fold v2 — Recall (churn)",
                                 **PLOTLY_DARK, height=360, xaxis_title="Fold", yaxis_title="Recall (%)",
                                 yaxis_range=[0, cv_df["recall_churn"].max() * 100 + 8])
            st.plotly_chart(fig_cv, use_container_width=True, key="cv_chart")

        # Feature importance v2
        imp_plot = imp.copy()
        imp_plot["label"] = imp_plot["feature"].map(FEAT_LABELS).fillna(imp_plot["feature"])
        imp_plot = imp_plot.sort_values("importance")
        fig_imp = go.Figure(go.Bar(
            x=imp_plot["importance"], y=imp_plot["label"], orientation="h",
            text=[f"{v:.3f}" for v in imp_plot["importance"]], textposition="outside", cliponaxis=False,
            marker_color=["#63e6be" if f in comport else "#74c0fc" for f in imp_plot["feature"]],
            marker_line_color="#2d3250", marker_line_width=1,
        ))
        fig_imp.update_layout(
            title="Feature Importance — modelo v2 (verde = early-warning comportamental)",
            **PLOTLY_DARK, height=440,
            xaxis_title="Importância (impurity-based)",
            xaxis_range=[0, imp_plot["importance"].max() * 1.35],
        )
        fig_imp.update_layout(margin=dict(t=60, b=40, l=220, r=70))
        st.plotly_chart(fig_imp, use_container_width=True, key="imp_chart")

        # Thresholds calibrados
        st.markdown("<hr style='border-color:#2d3250; margin:20px 0 12px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='color:#8b95b0; font-size:12px; font-weight:600; letter-spacing:0.8px; text-transform:uppercase; margin-bottom:10px;'>Thresholds por segmento — <code>reports/thresholds_v2.md</code></div>", unsafe_allow_html=True)
        st.dataframe(thr_tbl, use_container_width=True, hide_index=True)

        # Leitura honesta (substitui a antiga seção de "ROI estimado")
        diff_pp = (v2["recall_churn"] - v1["recall_churn"]) * 100
        st.markdown(f"""
        <div class='insight-box' style='margin-top:14px;'>
            <div style='font-weight:600; color:#e8ecf4; margin-bottom:6px;'>O que este projeto demonstra</div>
            <div style='line-height:1.7;'>
            Contrato de dados temporal, separação de produtos analíticos (churn de cliente vs. exposição de carteira),
            calibração de threshold por custo assimétrico e serving com schema versionado.
            A v2 supera a v1 em recall no mesmo split (+{diff_pp:.1f} p.p.), mas com ~28 eventos de churn no teste
            o IC95% da diferença inclui zero — <b>não é um ganho robusto</b>, e o dado é sintético.
            Nenhum número aqui é efeito de negócio observado.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================
# TAB 4 — CARTEIRA EXPOSTA POR ASSESSOR  (Direção B, ADR-0001)
# ==============================================================
with tab4:
    st.markdown('<p class="section-header">Carteira Exposta por Assessor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Tabela <b>descritiva</b> de priorização, não predição. '
        'Responde: "se ESTE assessor deixar a firma, quanto AuC da carteira dele tende a migrar junto?". '
        'Insumo para retenção de assessor — <b>não</b> é feature do classificador de churn de cliente '
        '(importância de <code>auc_exposto</code> = 1,3%, correlação com churn individual −0,04). '
        'Ver §6 do ADR-0001.</p>',
        unsafe_allow_html=True
    )

    CARTEIRA_PATH = "output/data/carteira_exposta_por_assessor.csv"
    if not os.path.exists(CARTEIRA_PATH):
        st.error("⚠️ `carteira_exposta_por_assessor.csv` não encontrado. Execute `python pipeline.py` primeiro.")
    else:
        carteira = pd.read_csv(CARTEIRA_PATH)

        fc1, fc2 = st.columns([1, 1])
        with fc1:
            canais = ["Todos"] + sorted(carteira["canal"].dropna().unique().tolist())
            canal_sel = st.selectbox("Canal", canais, key="cart_canal")
        with fc2:
            so_risco = st.checkbox("Só assessores com risco de saída (risco_saida = 1)", key="cart_risco")

        view = carteira.copy()
        if canal_sel != "Todos":
            view = view[view["canal"] == canal_sel]
        if so_risco:
            view = view[view["risco_saida"] == 1]
        view = view.sort_values("auc_exposto_total", ascending=False)

        k1, k2, k3, k4 = st.columns(4)
        kpis = [
            ("Assessores", f"{len(view)}", "#74c0fc"),
            ("AuC exposto total", f"R$ {view['auc_exposto_total'].sum()/1000:.1f} bi", "#ff6b6b"),
            ("Em risco de saída", f"{int((view['risco_saida'] == 1).sum())}", "#ffd43b"),
            ("Clientes cobertos", f"{int(view['qtd_clientes'].sum())}", "#63e6be"),
        ]
        for col, (label, val, color) in zip([k1, k2, k3, k4], kpis):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value' style='font-size:22px; color:{color};'>{val}</div>
                </div>
                """, unsafe_allow_html=True)

        gcol1, gcol2 = st.columns([1.2, 0.8], gap="medium")
        with gcol1:
            top = view.head(15).sort_values("auc_exposto_total")
            top_bi = top["auc_exposto_total"] / 1000
            fig_top = go.Figure(go.Bar(
                x=top_bi,
                y=top["assessor_id"],
                orientation="h",
                text=[f"R$ {v:.2f} bi" for v in top_bi],
                textposition="outside",
                cliponaxis=False,
                marker_color=["#f03e3e" if r == 1 else "#74c0fc" for r in top["risco_saida"]],
                marker_line_color="#2d3250",
                marker_line_width=1,
            ))
            fig_top.update_layout(
                title="Top 15 — AuC exposto (vermelho = risco de saída)",
                **PLOTLY_DARK,
                height=420,
                xaxis_title="AuC exposto (R$ bi)",
                xaxis_range=[0, top_bi.max() * 1.25] if len(top) else [0, 1],
            )
            fig_top.update_layout(margin=dict(t=60, b=40, l=90, r=90))
            st.plotly_chart(fig_top, use_container_width=True, key="cart_top")

        with gcol2:
            fig_sc = go.Figure(go.Scatter(
                x=view["anos_de_casa"],
                y=view["pct_carteira_exposta"] * 100,
                mode="markers",
                marker=dict(
                    size=(view["auc_exposto_total"] / view["auc_exposto_total"].max() * 34 + 6)
                    if view["auc_exposto_total"].max() > 0 else 8,
                    color=["#f03e3e" if r == 1 else "#74c0fc" for r in view["risco_saida"]],
                    line=dict(color="#2d3250", width=1),
                ),
                text=view["assessor_id"],
            ))
            fig_sc.update_layout(
                title="Anos de casa × % da carteira exposta (tamanho = AuC)",
                **PLOTLY_DARK,
                height=420,
                xaxis_title="Anos de casa",
                yaxis_title="% carteira exposta",
            )
            st.plotly_chart(fig_sc, use_container_width=True, key="cart_scatter")

        st.markdown("<div style='color:#8b95b0; font-size:12px; font-weight:600; letter-spacing:0.8px; text-transform:uppercase; margin:8px 0 10px 0;'>Detalhe por assessor</div>", unsafe_allow_html=True)
        tbl = view.copy()
        tbl["auc_total_carteira"] = tbl["auc_total_carteira"].apply(lambda v: f"R$ {v/1000:.2f} bi")
        tbl["auc_exposto_total"]  = tbl["auc_exposto_total"].apply(lambda v: f"R$ {v/1000:.2f} bi")
        tbl["pct_carteira_exposta"] = tbl["pct_carteira_exposta"].apply(lambda v: f"{v*100:.1f}%")
        tbl["risco_saida"] = tbl["risco_saida"].map({1: "🔴 sim", 0: "—"})
        tbl.columns = ["Assessor", "Clientes", "AuC carteira", "AuC exposto", "% exposta", "Canal", "Anos de casa", "Risco saída"]
        st.dataframe(tbl.reset_index(drop=True), use_container_width=True, height=340)


# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#6b7590; font-size:12px; padding:8px 0;'>
    Pipeline Churn Finance · v2 early-warning comportamental · dado sintético · 2026
</div>
""", unsafe_allow_html=True)
