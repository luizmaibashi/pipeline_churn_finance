# =============================================================
# api.py — FastAPI: Churn Finance Prediction Service
# Fase 2 do Roadmap: O mercado consome APIs
#
# Uso:
#   uvicorn api:app --reload --port 8000
#
# Documentação interativa:
#   http://localhost:8000/docs   (Swagger UI)
#   http://localhost:8000/redoc  (ReDoc)
#
# Endpoints:
#   GET  /                        → health check + status do modelo
#   GET  /model/info              → metadados da versão em produção
#   POST /predict                 → previsão individual (1 cliente)
#   POST /predict/batch           → previsão em lote (N clientes)
#   GET  /monitor/latest          → último relatório de drift
#   GET  /clients/high-risk       → lista clientes de alto risco
# =============================================================

from __future__ import annotations

import os
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from contextlib import asynccontextmanager

# ── Importa o transformer customizado (necessário para unpickle)
from transformers import FeatureEngineer   # noqa: F401

# ── Constantes do PROBLEM.md ─────────────────────────────────
SEGMENTOS_VALIDOS = ["Varejo", "Alta Renda", "Wealth", "Corporate"]
THRESHOLD_MAP = {
    "Varejo"    : 0.40,
    "Alta Renda": 0.50,
    "Wealth"    : 0.60,
    "Corporate" : 0.60,
}
FEATURES_BASE = [
    "segmento", "meses_cliente", "qtd_produtos",
    "retorno_12m_pct", "freq_contato_mes", "saldo_bi"
]
SHAP_CSV  = os.path.join("output", "shap", "client_explanations.csv")
DATA_CSV  = os.path.join("output", "data", "base_clientes.csv")
MODELS_DIR    = os.path.join("output", "models")
POINTER_FILE  = os.path.join(MODELS_DIR, "current_version.txt")
MONITOR_DIR   = os.path.join("output", "monitor")

TRADUCAO = {
    "retorno_12m_pct":   "Retorno da carteira (12m)",
    "freq_contato_mes":  "Frequência de contato com assessor",
    "retorno_relativo":  "Retorno relativo ao benchmark",
    "engajamento_score": "Score de engajamento",
    "saldo_bi":          "Saldo sob custódia",
    "qtd_produtos":      "Quantidade de produtos",
    "meses_cliente":     "Tempo como cliente (meses)",
    "flag_risco":        "Flag de risco comportamental",
    "intensidade_rel":   "Intensidade de relacionamento",
    "segmento_enc":      "Segmento do cliente",
}


# ── Estado global do app ─────────────────────────────────────
_state: dict = {}


def _load_model() -> object:
    """Carrega o modelo da versão atual de produção."""
    if os.path.exists(POINTER_FILE):
        with open(POINTER_FILE) as f:
            version = f.read().strip()
        pkl = os.path.join(MODELS_DIR, version, "gb_pipeline.pkl")
        meta_path = os.path.join(MODELS_DIR, version, "metadata.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
    else:
        pkl     = os.path.join(MODELS_DIR, "gb_pipeline.pkl")
        version = "flat"
        meta    = {}

    if not os.path.exists(pkl):
        raise FileNotFoundError(
            f"Modelo não encontrado em {pkl}. Execute 'python pipeline.py' primeiro."
        )

    model = joblib.load(pkl)
    return model, version, meta


def _load_shap_explanations() -> pd.DataFrame | None:
    if os.path.exists(SHAP_CSV):
        return pd.read_csv(SHAP_CSV)
    return None


def _latest_monitor_report() -> dict | None:
    """Lê o relatório de drift mais recente."""
    if not os.path.exists(MONITOR_DIR):
        return None
    reports = sorted([
        f for f in os.listdir(MONITOR_DIR)
        if f.startswith("drift_report_") and f.endswith(".json")
    ], reverse=True)
    if not reports:
        return None
    with open(os.path.join(MONITOR_DIR, reports[0]), encoding="utf-8") as f:
        return json.load(f)


# ── Lifespan: carrega artefatos na inicialização ─────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo e artefatos uma vez, no startup."""
    try:
        model, version, meta = _load_model()
        _state["model"]   = model
        _state["version"] = version
        _state["meta"]    = meta
        _state["shap_df"] = _load_shap_explanations()
        _state["started_at"] = datetime.datetime.now().isoformat()
        print(f"[OK] Modelo v{version} carregado.")
    except FileNotFoundError as e:
        print(f"[WARN] {e} — API iniciada sem modelo. Execute pipeline.py.")
        _state["model"]   = None
        _state["version"] = "N/A"
        _state["meta"]    = {}
        _state["shap_df"] = None
        _state["started_at"] = datetime.datetime.now().isoformat()
    yield
    _state.clear()


# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="Churn Finance — Prediction API",
    description=(
        "API de predição de churn para o ecossistema financeiro. "
        "Baseada em PROBLEM.md v1.0 — define thresholds por segmento, "
        "janela temporal de 30 dias e explicabilidade LGPD-ready via SHAP."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas Pydantic ─────────────────────────────────────────

class ClienteInput(BaseModel):
    """Perfil de um cliente para predição de churn."""

    cliente_id: str = Field(
        default="CLI_NOVO",
        description="Identificador único do cliente",
        examples=["CLI00001"]
    )
    segmento: Literal["Varejo", "Alta Renda", "Wealth", "Corporate"] = Field(
        description="Segmento de investimento do cliente",
        examples=["Wealth"]
    )
    meses_cliente: int = Field(
        ge=1, le=600,
        description="Tempo como cliente em meses (janela de observação: últimos 90 dias)",
        examples=[36]
    )
    qtd_produtos: int = Field(
        ge=1, le=20,
        description="Quantidade de produtos financeiros ativos",
        examples=[3]
    )
    retorno_12m_pct: float = Field(
        ge=0.0, le=100.0,
        description="Retorno da carteira nos últimos 12 meses (%)",
        examples=[11.5]
    )
    freq_contato_mes: int = Field(
        ge=0, le=60,
        description="Número de contatos com assessor no último mês",
        examples=[2]
    )
    saldo_bi: float = Field(
        gt=0.0,
        description="Saldo sob custódia em R$ bilhões",
        examples=[0.5]
    )

    @field_validator("segmento")
    @classmethod
    def segmento_valido(cls, v):
        if v not in SEGMENTOS_VALIDOS:
            raise ValueError(f"Segmento deve ser um de: {SEGMENTOS_VALIDOS}")
        return v


class PredictionResult(BaseModel):
    """Resultado da predição de churn para um cliente."""
    cliente_id: str
    segmento: str
    churn_probability: float = Field(description="Probabilidade de churn [0.0, 1.0]")
    churn_probability_pct: str = Field(description="Probabilidade formatada (ex: '78.3%')")
    risk_level: Literal["BAIXO", "MEDIO", "ALTO"]
    threshold_used: float = Field(description="Threshold do segmento usado para classificação")
    churn_predicted: bool = Field(description="True se prob >= threshold do segmento")
    top3_reasons: list[dict] = Field(description="Top 3 fatores SHAP (se disponível)")
    recommended_action: str
    auc_at_risk_MM: float = Field(description="Receita anual estimada em risco (R$ milhões)")
    flow: str = Field(description="Fluxo operacional: AUTO→CRM ou REVISAO_HUMANA")
    scored_at: str


class BatchInput(BaseModel):
    """Payload para predição em lote."""
    clientes: list[ClienteInput] = Field(
        min_length=1, max_length=1000,
        description="Lista de clientes para scoring (máx. 1000 por request)"
    )


class BatchResult(BaseModel):
    """Resultado de predição em lote."""
    total: int
    scored_at: str
    model_version: str
    results: list[PredictionResult]
    summary: dict


# ── Funções de negócio ────────────────────────────────────────

def _risk_level(prob: float, segmento: str) -> str:
    threshold = THRESHOLD_MAP.get(segmento, 0.50)
    if prob >= 0.60:
        return "ALTO"
    elif prob >= 0.35:
        return "MEDIO"
    return "BAIXO"


def _recommended_action(risk: str, prob: float, features: dict) -> str:
    if risk == "BAIXO":
        return "Monitoramento rotineiro. Nenhuma acao imediata necessaria."

    actions = []
    if features.get("retorno_12m_pct", 99) < 9.0:
        actions.append("Apresentar portfólio com maior CDI+ e produtos de renda variável diversificada")
    if features.get("freq_contato_mes", 99) == 0:
        actions.append("Agendar call consultiva com assessor — cliente sem contato recente")
    if features.get("qtd_produtos", 99) == 1:
        actions.append("Oferecer diversificação de produtos — cliente monoproduto")
    if features.get("saldo_bi", 99) < 0.1:
        actions.append("Avaliar incentivo de aporte mínimo ou campanha de fidelização")

    if not actions:
        actions.append("Contato proativo pelo assessor para entender necessidades atuais")

    return " | ".join(actions)


def _flow(segmento: str, saldo_bi: float) -> str:
    """Define o fluxo operacional conforme PROBLEM.md Seção 5.2."""
    if segmento == "Wealth" or saldo_bi >= 0.5:
        return "REVISAO_HUMANA (especialista)"
    return "AUTO → CRM"


def _get_shap_reasons(cliente_id: str) -> list[dict]:
    """Busca top-3 razões SHAP do CSV pré-computado."""
    shap_df = _state.get("shap_df")
    if shap_df is None:
        return []
    row = shap_df[shap_df["cliente_id"] == cliente_id]
    if row.empty:
        return []
    texto = str(row.iloc[0].get("explicacao", ""))
    reasons = []
    for line in texto.split("\n"):
        line = line.strip()
        if line.startswith("•") or line.startswith("-"):
            reasons.append({"descricao": line.lstrip("• -").strip()})
    return reasons[:3]


def _predict_one(cliente: ClienteInput) -> PredictionResult:
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado. Execute pipeline.py.")

    X = pd.DataFrame([{
        "segmento":        cliente.segmento,
        "meses_cliente":   cliente.meses_cliente,
        "qtd_produtos":    cliente.qtd_produtos,
        "retorno_12m_pct": cliente.retorno_12m_pct,
        "freq_contato_mes":cliente.freq_contato_mes,
        "saldo_bi":        cliente.saldo_bi,
    }])

    prob = float(model.predict_proba(X)[0][1])
    prob = round(prob, 4)

    risk        = _risk_level(prob, cliente.segmento)
    threshold   = THRESHOLD_MAP.get(cliente.segmento, 0.50)
    predicted   = prob >= threshold
    action      = _recommended_action(risk, prob, cliente.dict())
    flow_str    = _flow(cliente.segmento, cliente.saldo_bi)
    reasons     = _get_shap_reasons(cliente.cliente_id)
    auc_risk_mm = round(cliente.saldo_bi * 1000 * 0.012 * prob, 2)  # 1.2% do AuC × prob

    return PredictionResult(
        cliente_id            = cliente.cliente_id,
        segmento              = cliente.segmento,
        churn_probability     = prob,
        churn_probability_pct = f"{prob*100:.1f}%",
        risk_level            = risk,
        threshold_used        = threshold,
        churn_predicted       = predicted,
        top3_reasons          = reasons,
        recommended_action    = action,
        auc_at_risk_MM        = auc_risk_mm,
        flow                  = flow_str,
        scored_at             = datetime.datetime.now().isoformat(),
    )


# ── Endpoints ────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def health_check():
    """
    Health check da API.
    Retorna status, versão do modelo e uptime.
    """
    model_ok = _state.get("model") is not None
    return {
        "status"       : "ok" if model_ok else "degraded",
        "api_version"  : "1.0.0",
        "model_version": _state.get("version", "N/A"),
        "model_loaded" : model_ok,
        "started_at"   : _state.get("started_at"),
        "timestamp"    : datetime.datetime.now().isoformat(),
        "docs"         : "/docs",
    }


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Retorna os metadados completos do modelo em produção:
    versão, métricas, perfil de dados e data de promoção.
    """
    meta = _state.get("meta", {})
    if not meta:
        return {"message": "Modelo carregado sem metadata (versão flat). Execute version_manager.py para versionar."}
    return meta


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(cliente: ClienteInput):
    """
    Predição de churn para **um único cliente**.

    - Aplica threshold específico por segmento (PROBLEM.md Seção 4.3)
    - Retorna probabilidade + nível de risco + ação recomendada + fluxo operacional
    - Busca explicações SHAP pré-computadas se disponíveis

    **Thresholds por segmento:**
    - Varejo: ≥ 0.40 → AUTO → CRM
    - Alta Renda: ≥ 0.50 → CRM + Notificação Assessor
    - Wealth / Corporate: ≥ 0.60 → Revisão humana obrigatória
    """
    return _predict_one(cliente)


@app.post("/predict/batch", response_model=BatchResult, tags=["Prediction"])
async def predict_batch(payload: BatchInput):
    """
    Predição de churn em **lote** (até 1.000 clientes por request).

    Retorna resultados individuais + resumo agregado:
    - total por nível de risco
    - AuC total em risco
    - clientes que precisam de revisão humana
    """
    results = [_predict_one(c) for c in payload.clientes]

    alto   = [r for r in results if r.risk_level == "ALTO"]
    medio  = [r for r in results if r.risk_level == "MEDIO"]
    baixo  = [r for r in results if r.risk_level == "BAIXO"]
    humano = [r for r in results if "REVISAO" in r.flow]

    total_auc_risk = round(sum(r.auc_at_risk_MM for r in alto), 2)

    summary = {
        "total_clientes"         : len(results),
        "alto_risco"             : len(alto),
        "medio_risco"            : len(medio),
        "baixo_risco"            : len(baixo),
        "revisao_humana"         : len(humano),
        "auc_total_em_risco_MM"  : total_auc_risk,
        "pct_alto_risco"         : f"{len(alto)/len(results)*100:.1f}%",
    }

    return BatchResult(
        total         = len(results),
        scored_at     = datetime.datetime.now().isoformat(),
        model_version = _state.get("version", "N/A"),
        results       = results,
        summary       = summary,
    )


@app.get("/monitor/latest", tags=["MLOps"])
async def monitor_latest():
    """
    Retorna o **último relatório de Data Drift** gerado pelo `monitor.py`.

    Inclui status geral (OK / ATENÇÃO / CRÍTICO), alertas detectados
    e recomendação de re-treino.
    """
    report = _latest_monitor_report()
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="Nenhum relatório de drift encontrado. Execute 'python monitor.py'."
        )
    return report


@app.get("/clients/high-risk", tags=["Analytics"])
async def high_risk_clients(
    limit: int = Query(default=20, ge=1, le=200, description="Número máximo de clientes retornados"),
    segmento: Optional[str] = Query(default=None, description="Filtrar por segmento (ex: Wealth)")
):
    """
    Retorna a lista de clientes de **alto risco** com suas explicações SHAP.

    - Ordenados por probabilidade de churn (desc)
    - Inclui ação recomendada e fluxo operacional
    - Alimenta o CRM e as filas de assessores
    """
    shap_df = _state.get("shap_df")
    if shap_df is None:
        raise HTTPException(
            status_code=404,
            detail="Explicações SHAP não disponíveis. Execute 'python shap_analysis.py'."
        )

    df = shap_df[shap_df["churn_prob"] >= 0.35].copy()

    # Adiciona dados de segmento do dataset base
    if os.path.exists(DATA_CSV):
        base = pd.read_csv(DATA_CSV)[["cliente_id", "segmento", "saldo_bi"]]
        df   = df.merge(base, on="cliente_id", how="left")

    # Filtra por segmento apenas se a coluna existir após o merge
    if segmento and "segmento" in df.columns:
        df = df[df["segmento"] == segmento]

    df = df.sort_values("churn_prob", ascending=False).head(limit)

    clientes = []
    for _, row in df.iterrows():
        prob    = float(row["churn_prob"])
        seg     = str(row.get("segmento", "N/A"))
        saldo   = float(row.get("saldo_bi", 0))
        risk    = _risk_level(prob, seg)
        flow    = _flow(seg, saldo)
        clientes.append({
            "cliente_id"          : row["cliente_id"],
            "segmento"            : seg,
            "churn_probability"   : round(prob, 4),
            "churn_probability_pct": f"{prob*100:.1f}%",
            "risk_level"          : risk,
            "churn_real"          : int(row.get("churn_real", -1)),
            "auc_at_risk_MM"      : round(saldo * 1000 * 0.012 * prob, 2),
            "flow"                : flow,
            "explicacao"          : str(row.get("explicacao", "")),
        })

    return {
        "total"         : len(clientes),
        "filter_segment": segmento or "all",
        "model_version" : _state.get("version", "N/A"),
        "generated_at"  : datetime.datetime.now().isoformat(),
        "clientes"      : clientes,
    }
