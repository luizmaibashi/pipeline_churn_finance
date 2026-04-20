# =============================================================
# app.py — Dashboard Streamlit: Churn Finance Pipeline
# Uso: streamlit run app.py
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

from transformers import FeatureEngineer  # Necessário para unpicklear o Pipeline

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
FEATURES = [
    "meses_cliente", "qtd_produtos", "retorno_12m_pct",
    "freq_contato_mes", "saldo_bi",
    "engajamento_score", "retorno_relativo", "flag_risco",
    "intensidade_rel", "segmento_enc"
]

FEAT_LABELS = {
    "saldo_bi":           "Saldo (R$ bi)",
    "intensidade_rel":    "Intensidade Relacionamento",
    "meses_cliente":      "Meses como Cliente",
    "retorno_relativo":   "Retorno Relativo à Média",
    "segmento_enc":       "Segmento",
    "retorno_12m_pct":    "Retorno 12m (%)",
    "engajamento_score":  "Score de Engajamento",
    "qtd_produtos":       "Qtd. Produtos",
    "freq_contato_mes":   "Freq. Contato/Mês",
    "flag_risco":         "Flag de Risco"
}

SEG_COLORS = {
    "Varejo":      "#f03e3e",
    "Alta Renda":  "#ffd43b",
    "Wealth":      "#74c0fc",
    "Corporate":   "#63e6be"
}

PLOTLY_DARK = dict(
    paper_bgcolor="#1a1f2e",
    plot_bgcolor="#1a1f2e",
    font=dict(color="#c1c8de", family="Inter"),
    margin=dict(t=60, b=40, l=50, r=20),
)


@st.cache_resource
def load_artifacts():
    """Carrega o Pipeline formal do scikit-learn."""
    model  = joblib.load("output/models/gb_pipeline.pkl")
    return model


@st.cache_data
def load_data():
    """Carrega os CSVs gerados pelo pipeline."""
    df_fe = pd.read_csv("output/data/base_feature_eng.csv")
    imp   = pd.read_csv("output/data/feature_importance.csv")
    bench = pd.read_csv("output/data/benchmark_results.csv")
    cv    = pd.read_csv("output/data/cv_scores.csv")
    cm_df = pd.read_csv("output/data/confusion_matrix.csv")
    return df_fe, imp, bench, cv, cm_df





def risk_badge(prob):
    if prob >= 0.55:
        return '<span class="badge-alto">🔴 Alto Risco</span>'
    elif prob >= 0.30:
        return '<span class="badge-medio">🟡 Médio Risco</span>'
    else:
        return '<span class="badge-baixo">🟢 Baixo Risco</span>'


def gauge_chart(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 38, "color": "#e8ecf4"}},
        delta={"reference": 20, "increasing": {"color": "#ff6b6b"},
               "decreasing": {"color": "#51cf66"}},
        gauge={
            "axis":    {"range": [0, 100], "tickcolor": "#8b95b0", "tickfont": {"color": "#8b95b0"}},
            "bar":     {"color": "#f03e3e" if prob >= 0.55 else "#ffd43b" if prob >= 0.30 else "#51cf66", "thickness": 0.28},
            "bgcolor": "#252a3d",
            "bordercolor": "#2d3250",
            "steps": [
                {"range": [0,  30],  "color": "rgba(81,207,102,0.08)"},
                {"range": [30, 55],  "color": "rgba(255,212,59,0.08)"},
                {"range": [55, 100], "color": "rgba(240,62,62,0.08)"},
            ],
            "threshold": {"line": {"color": "#a9b4d0", "width": 2}, "value": 20}
        },
        title={"text": "Probabilidade de Churn", "font": {"size": 14, "color": "#8b95b0"}}
    ))
    fig.update_layout(height=280, paper_bgcolor="#1a1f2e", margin=dict(t=20, b=20, l=40, r=40))
    return fig


# ── Verificação de artefatos ──────────────────────────────────
artifacts_ok = all(os.path.exists(p) for p in [
    "output/models/gb_pipeline.pkl",
    "output/data/base_feature_eng.csv",
    "output/data/feature_importance.csv",
    "output/data/benchmark_results.csv",
    "output/data/cv_scores.csv",
    "output/data/confusion_matrix.csv",
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
        "Pipeline Final":    "output/models/gb_pipeline.pkl",
        "Dados (FE)":        "output/data/base_feature_eng.csv",
        "Feature Importance":"output/data/feature_importance.csv",
        "Benchmark":         "output/data/benchmark_results.csv",
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
        🏢 Ecossistema financeiro<br>
        💰 ~R$75bi sob custódia<br>
        👥 ~12.000 clientes<br>
        📈 4 segmentos de carteira<br>
        ⚠️ 12% churn atual
    </div>
    """, unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div style='padding: 8px 0 24px 0;'>
    <h1 style='color:#e8ecf4; font-size:30px; font-weight:700; margin:0; line-height:1.2;'>
        Pipeline de Predição de Churn
    </h1>
    <p style='color:#8b95b0; margin:6px 0 0 0; font-size:14px;'>
        Ecossistema Financeiro — Gradient Boosting com Feature Engineering
    </p>
</div>
""", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🎯  Predição Individual",
    "📈  Análise da Carteira",
    "🔬  Performance do Modelo"
])


# ==============================================================
# TAB 1 — PREDIÇÃO INDIVIDUAL
# ==============================================================
with tab1:
    st.markdown('<p class="section-header">Preditor de Churn — Cliente Individual</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Insira o perfil do cliente para calcular a probabilidade de churn nos próximos 30 dias.</p>', unsafe_allow_html=True)

    if not artifacts_ok:
        st.error("⚠️ Artefatos de modelo não encontrados. Execute `python pipeline.py` primeiro.")
    else:
        model = load_artifacts()
        df_fe, _, _, _, _ = load_data()
        media_retorno = df_fe["retorno_12m_pct"].mean()

        col_form, col_result = st.columns([1, 1], gap="large")

        with col_form:
            st.markdown("<div style='color:#8b95b0; font-size:12px; font-weight:600; letter-spacing:0.8px; text-transform:uppercase; margin-bottom:14px;'>Perfil do Cliente</div>", unsafe_allow_html=True)

            with st.container():
                segmento = st.selectbox(
                    "Segmento",
                    ["Varejo", "Alta Renda", "Wealth", "Corporate"],
                    key="pred_segmento"
                )
                col_a, col_b = st.columns(2)
                with col_a:
                    meses = st.slider("Tempo como cliente (meses)", 1, 144, 36, key="pred_meses")
                    qtd_prod = st.slider("Nº de produtos", 1, 8, 3, key="pred_qtd")
                with col_b:
                    retorno = st.slider("Retorno 12m (%)", 0.0, 30.0, 11.5, 0.1, key="pred_retorno")
                    freq = st.slider("Contatos/mês com assessor", 0, 15, 3, key="pred_freq")

                saldo = st.slider("Saldo sob custódia (R$ bi)", 0.01, 5.0, 0.5, 0.01, key="pred_saldo",
                                  format="R$ %.2f bi")

                # Predição em tempo real c/ Pipeline (elimina Training-Serving Skew)
                X_new = pd.DataFrame([{
                    "segmento": segmento,
                    "meses_cliente": meses,
                    "qtd_produtos": qtd_prod,
                    "retorno_12m_pct": retorno,
                    "freq_contato_mes": freq,
                    "saldo_bi": saldo
                }])
                prob = model.predict_proba(X_new)[0][1]
                pred = int(prob >= 0.5)

        with col_result:
            st.markdown("<div style='color:#8b95b0; font-size:12px; font-weight:600; letter-spacing:0.8px; text-transform:uppercase; margin-bottom:14px;'>Resultado da Predição</div>", unsafe_allow_html=True)

            # Gauge
            st.plotly_chart(gauge_chart(prob), use_container_width=True, key="gauge")

            # Badge de risco
            st.markdown(
                f"<div style='text-align:center; margin: -10px 0 16px 0;'>{risk_badge(prob)}</div>",
                unsafe_allow_html=True
            )

            # Insights
            insights = []
            if freq == 0:
                insights.append("⚠️ Nenhum contato com assessor — maior fator de abandono")
            if retorno < media_retorno:
                insights.append(f"⚠️ Retorno abaixo da média da carteira ({media_retorno:.1f}%)")
            if qtd_prod == 1:
                insights.append("⚠️ Monoproduto — menor fidelização do cliente")
            if meses < 12:
                insights.append("⚠️ Cliente recente (<12 meses) — maior risco de saída")
            if saldo < 0.1:
                insights.append("⚠️ Saldo baixo — menor custo de saída para o cliente")
            if segmento == "Varejo":
                insights.append("📌 Segmento Varejo — maior taxa histórica de churn (25%+)")

            if insights:
                st.markdown("<div class='alert-box'><b style='color:#ffa8a8; font-size:13px;'>Fatores de Risco Identificados</b><br><div style='font-size:12px; color:#ffcdd2; line-height:2; margin-top:6px;'>" + "<br>".join(insights) + "</div></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='alert-box-green'><b style='color:#a9e34b; font-size:13px;'>✅ Perfil de baixo risco</b><br><span style='font-size:12px; color:#cbf078;'>Nenhum fator de risco crítico identificado para este cliente.</span></div>", unsafe_allow_html=True)

            # ROI rápido
            auc_risco = saldo * 0.012
            st.markdown(f"""
            <div class='insight-box' style='margin-top:12px;'>
                <div style='font-weight:600; color:#e8ecf4; margin-bottom:6px;'>💼 Impacto financeiro estimado</div>
                <div>Saldo sob custódia: <b style='color:#74c0fc;'>R$ {saldo:.2f} bi</b></div>
                <div>Receita anual em risco: <b style='color:{'#ff6b6b' if prob >= 0.55 else '#ffd43b' if prob >= 0.30 else '#51cf66'};'>R$ {auc_risco*1000:.1f}M/ano</b></div>
                <div style='margin-top:6px; font-size:11px; color:#6b7590;'>* 1,2% do AuC como proxy de receita anual</div>
            </div>
            """, unsafe_allow_html=True)

        # Perfis rápidos
        st.markdown("<hr style='border-color:#2d3250; margin:28px 0 16px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='color:#8b95b0; font-size:12px; font-weight:600; letter-spacing:0.8px; text-transform:uppercase; margin-bottom:14px;'>Simule Perfis Pré-definidos</div>", unsafe_allow_html=True)

        p1, p2, p3 = st.columns(3)
        profiles = [
            {
                "label": "🔴 Alto Risco",
                "desc": "Varejo | 3 meses | 1 produto | Retorno 6% | 0 contatos | R$0.05bi",
                "color": "#ff6b6b",
                "vals": {"segmento":"Varejo", "meses_cliente":3, "qtd_produtos":1, "retorno_12m_pct":6.0, "freq_contato_mes":0, "saldo_bi":0.05}
            },
            {
                "label": "🟡 Médio Risco",
                "desc": "Alta Renda | 24 meses | 3 produtos | Retorno 10% | 2 contatos | R$0.8bi",
                "color": "#ffd43b",
                "vals": {"segmento":"Alta Renda", "meses_cliente":24, "qtd_produtos":3, "retorno_12m_pct":10.0, "freq_contato_mes":2, "saldo_bi":0.8}
            },
            {
                "label": "🟢 Baixo Risco",
                "desc": "Wealth | 72 meses | 7 produtos | Retorno 15% | 5 contatos | R$3bi",
                "color": "#51cf66",
                "vals": {"segmento":"Wealth", "meses_cliente":72, "qtd_produtos":7, "retorno_12m_pct":15.0, "freq_contato_mes":5, "saldo_bi":3.0}
            }
        ]
        for col, prof in zip([p1, p2, p3], profiles):
            with col:
                x = pd.DataFrame([prof["vals"]])
                p_val = model.predict_proba(x)[0][1]
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
        df_fe, imp, bench, cv_df, cm_df = load_data()
        media_retorno_base = df_fe["retorno_12m_pct"].mean()

        FEATURES_BASE = [
            "segmento", "meses_cliente", "qtd_produtos", 
            "retorno_12m_pct", "freq_contato_mes", "saldo_bi"
        ]
        
        # Calcular probabilidades para toda a carteira
        X_all = df_fe[FEATURES_BASE]
        df_plot = df_fe.copy()
        df_plot["prob_churn"] = model.predict_proba(X_all)[:, 1]
        df_plot["risco"] = pd.cut(
            df_plot["prob_churn"],
            bins=[0, 0.30, 0.55, 1.0],
            labels=["Baixo", "Médio", "Alto"]
        )

        # KPIs
        total = len(df_plot)
        alto_risco = (df_plot["risco"] == "Alto").sum()
        medio_risco = (df_plot["risco"] == "Médio").sum()
        churn_real = df_plot["churn"].sum()
        auc_risco = df_plot.loc[df_plot["risco"] == "Alto", "saldo_bi"].sum()

        k1, k2, k3, k4 = st.columns(4)
        kpis = [
            (k1, "Total Clientes", f"{total:,}", ""),
            (k2, "Alto Risco (modelo)", f"{alto_risco}", f"{alto_risco/total*100:.1f}% da carteira"),
            (k3, "Churn Real (base)", f"{churn_real}", f"{churn_real/total*100:.1f}% da base"),
            (k4, "AuC em Risco (alto)", f"R${auc_risco:.1f}bi", "clientes flag alto risco"),
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
            # Scatter: saldo vs prob_churn
            fig_scatter = px.scatter(
                df_plot,
                x="saldo_bi",
                y="prob_churn",
                color="segmento",
                color_discrete_map=SEG_COLORS,
                hover_data=["cliente_id", "meses_cliente", "qtd_produtos"],
                opacity=0.7,
                labels={"saldo_bi": "Saldo (R$ bi)", "prob_churn": "Prob. Churn", "segmento": "Segmento"}
            )
            fig_scatter.add_hline(y=0.55, line_dash="dot", line_color="#ff6b6b",
                                  annotation_text="Limiar alto risco (55%)")
            fig_scatter.add_hline(y=0.30, line_dash="dot", line_color="#ffd43b",
                                  annotation_text="Limiar médio risco (30%)")
            fig_scatter.update_layout(
                title="Saldo vs Probabilidade de Churn",
                **PLOTLY_DARK,
                height=340,
                xaxis=dict(type="log", title="Saldo (R$ bi) — escala log"),
                yaxis_title="Prob. Churn"
            )
            st.plotly_chart(fig_scatter, use_container_width=True, key="scatter")

        with gc4:
            # Top 15 clientes em risco
            top_risco = (
                df_plot[df_plot["risco"] == "Alto"]
                .sort_values("prob_churn", ascending=False)
                .head(10)[["cliente_id", "segmento", "prob_churn", "saldo_bi"]]
            )
            st.markdown("<div style='color:#8b95b0; font-size:12px; font-weight:600; letter-spacing:0.8px; text-transform:uppercase; margin-bottom:10px;'>🔴 Top 10 Clientes — Maior Risco</div>", unsafe_allow_html=True)
            if len(top_risco) > 0:
                top_risco_display = top_risco.copy()
                top_risco_display["prob_churn"] = top_risco_display["prob_churn"].apply(lambda x: f"{x*100:.1f}%")
                top_risco_display["saldo_bi"]   = top_risco_display["saldo_bi"].apply(lambda x: f"R${x:.2f}bi")
                top_risco_display.columns = ["ID", "Segmento", "Prob. Churn", "Saldo"]
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
    st.markdown('<p class="section-header">Performance e Avaliação do Modelo</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Gradient Boosting treinado com 5-Fold Cross-Validation estratificado. Meta: F1-macro ≥ 0.55 | ROC-AUC ≥ 0.70</p>', unsafe_allow_html=True)

    if not artifacts_ok:
        st.error("⚠️ Dados não encontrados. Execute `python pipeline.py` primeiro.")
    else:
        _, imp, bench, cv_df, cm_df = load_data()

        # KPIs do modelo final
        gb_row = bench[bench["modelo"] == "Gradient Boosting"].iloc[0]
        tn, fp, fn, tp = int(cm_df["tn"][0]), int(cm_df["fp"][0]), int(cm_df["fn"][0]), int(cm_df["tp"][0])
        recall_churn = tp / (tp + fn) if (tp + fn) > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        mets = [
            ("F1-macro (GB)", f"{gb_row['f1_macro']:.4f}", "Meta ≥ 0.55", gb_row['f1_macro'] >= 0.55),
            ("ROC-AUC (GB)",  f"{gb_row['roc_auc']:.4f}",  "Meta ≥ 0.70", gb_row['roc_auc'] >= 0.70),
            ("Recall Churn",  f"{recall_churn*100:.1f}%",   f"TP={tp} de {tp+fn} churns", True),
            ("Acurácia",      f"{gb_row['acuracia']:.4f}",  "⚠️ Não é métrica principal", False),
        ]
        for col, (label, val, note, ok) in zip([m1, m2, m3, m4], mets):
            with col:
                color = "#51cf66" if ok else "#ffd43b"
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value' style='color:{color};'>{val}</div>
                    <div style='font-size:11px; color:#6b7590; margin-top:4px;'>{note}</div>
                </div>
                """, unsafe_allow_html=True)

        row1, row2 = st.columns([1.3, 0.7], gap="medium")

        with row1:
            # Benchmark de modelos
            cores = ["#adb5bd", "#74c0fc", "#63e6be", "#ffd43b", "#f03e3e"]
            fig_bench = go.Figure()
            fig_bench.add_trace(go.Bar(
                name="F1-macro",
                x=bench["modelo"],
                y=bench["f1_macro"],
                text=[f"{v:.3f}" for v in bench["f1_macro"]],
                textposition="outside",
                marker_color=cores
            ))
            fig_bench.add_trace(go.Bar(
                name="ROC-AUC",
                x=bench["modelo"],
                y=bench["roc_auc"],
                text=[f"{v:.3f}" for v in bench["roc_auc"]],
                textposition="outside",
                marker_color=["rgba(0,0,0,0.2)"] * 5,
                marker_line_color=cores,
                marker_line_width=2
            ))
            fig_bench.add_hline(y=0.55, line_dash="dot", line_color="#ff6b6b",
                                annotation_text="Meta F1 ≥ 0.55")
            fig_bench.add_hline(y=0.70, line_dash="dot", line_color="#ffd43b",
                                annotation_text="Meta ROC ≥ 0.70")
            fig_bench.update_layout(
                title="Benchmark de Modelos — F1-macro vs ROC-AUC",
                barmode="group",
                **PLOTLY_DARK,
                height=360,
                legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                yaxis_range=[0, 0.82],
                xaxis_title="Modelo",
                yaxis_title="Score"
            )
            st.plotly_chart(fig_bench, use_container_width=True, key="bench_chart")

        with row2:
            # Confusion Matrix
            cm_arr = np.array([[tn, fp], [fn, tp]])
            fig_cm = go.Figure(go.Heatmap(
                z=cm_arr,
                x=["Ficou (0)", "Churnou (1)"],
                y=["Ficou (0)", "Churnou (1)"],
                text=[[str(v) for v in row] for row in cm_arr],
                texttemplate="<b>%{text}</b>",
                textfont=dict(size=28, color="white"),
                colorscale=[[0, "#1e2130"], [0.5, "#1971c2"], [1, "#1864ab"]],
                showscale=False
            ))
            fig_cm.update_layout(
                title=f"Matriz de Confusão (Gradient Boosting)",
                **PLOTLY_DARK,
                height=360,
                xaxis_title="Predito",
                yaxis_title="Real"
            )
            st.plotly_chart(fig_cm, use_container_width=True, key="cm_chart")

        row3, row4 = st.columns([1, 1], gap="medium")

        with row3:
            # Feature Importance
            imp_plot = imp.copy()
            imp_plot["label"] = imp_plot["feature"].map(FEAT_LABELS)
            imp_plot = imp_plot.sort_values("importance")

            fig_imp = go.Figure(go.Bar(
                x=imp_plot["importance"],
                y=imp_plot["label"],
                orientation="h",
                text=[f"{v:.3f}" for v in imp_plot["importance"]],
                textposition="outside",
                cliponaxis=False,
                marker_color=["#f03e3e" if v > 0.15 else "#ffd43b" if v > 0.08 else "#74c0fc"
                              for v in imp_plot["importance"]],
                marker_line_color="#2d3250",
                marker_line_width=1
            ))
            fig_imp.update_layout(
                title="Feature Importance — Gradient Boosting",
                **PLOTLY_DARK,
                height=380,
                margin=dict(t=60, b=40, l=140, r=70),
                xaxis_title="Importância",
                xaxis_range=[0, imp_plot["importance"].max() * 1.35]
            )
            st.plotly_chart(fig_imp, use_container_width=True, key="imp_chart")

        with row4:
            # Cross-Validation 5-Fold
            fig_cv = go.Figure(go.Bar(
                x=cv_df["fold"],
                y=cv_df["f1_macro"],
                text=[f"{v:.4f}" for v in cv_df["f1_macro"]],
                textposition="outside",
                marker_color="#74c0fc",
                marker_line_color="#4dabf7",
                marker_line_width=1.5
            ))
            mean_cv = cv_df["f1_macro"].mean()
            std_cv  = cv_df["f1_macro"].std()
            fig_cv.add_hline(y=mean_cv, line_dash="dash", line_color="#1971c2",
                             annotation_text=f"Média={mean_cv:.4f} ±{std_cv:.4f}",
                             annotation_position="top left")
            fig_cv.update_layout(
                title="Cross-Validation 5-Fold — F1-macro",
                **PLOTLY_DARK,
                height=380,
                xaxis_title="Fold",
                yaxis_title="F1-macro",
                yaxis_range=[cv_df["f1_macro"].min() - 0.04, cv_df["f1_macro"].max() + 0.06]
            )
            st.plotly_chart(fig_cv, use_container_width=True, key="cv_chart")

        # ROI financeiro
        st.markdown("<hr style='border-color:#2d3250; margin:24px 0 16px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='color:#8b95b0; font-size:12px; font-weight:600; letter-spacing:0.8px; text-transform:uppercase; margin-bottom:14px;'>💰 Análise de ROI Financeiro</div>", unsafe_allow_html=True)

        ticket_medio_mi = 6.7
        custo_contato   = 0.5       # R$ mil
        receita_retida  = 0.012     # 1.2% AuC

        alertados       = tp + fp
        auc_risco_roi   = tp * ticket_medio_mi
        receita_prot    = auc_risco_roi * receita_retida
        custo_total     = alertados * custo_contato / 1000
        roi_mi          = receita_prot - custo_total

        r1, r2, r3, r4 = st.columns(4)
        roi_kpis = [
            ("Clientes Alertados", str(alertados), "#74c0fc"),
            ("Churns Capturados", f"{tp} / {tp+fn}", "#ffd43b"),
            ("Receita Protegida/ano", f"R${receita_prot:.2f}mi", "#51cf66"),
            ("ROI Líquido Estimado", f"R${roi_mi:.2f}mi", "#51cf66"),
        ]
        for col, (label, val, color) in zip([r1, r2, r3, r4], roi_kpis):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value' style='font-size:22px; color:{color};'>{val}</div>
                </div>
                """, unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#6b7590; font-size:12px; padding:8px 0;'>
    Pipeline Churn Finance · Gradient Boosting · Tech Challenge · 2025
</div>
""", unsafe_allow_html=True)
