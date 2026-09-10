# =============================================================
# api.py — FastAPI: Churn Finance Prediction Service
# Fase 2 do Roadmap: O mercado consome APIs
#
# Uso (a partir da raiz do projeto):
#   uvicorn src.api:app --reload --port 8000
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

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from contextlib import asynccontextmanager

# ── Importa a fila de processamento assíncrono e os transformers
# (FeatureEngineer + StructuralNullImputer são necessários para unpicklear o
# Pipeline v2 — ADR-0001, Direção A early-warning).
from src.job_queue import JobQueue
from src.transformers import FeatureEngineer, StructuralNullImputer   # noqa: F401
from src.serving_contract import (
    SEGMENTOS_VALIDOS, FEATURES_V2_BASE,
    load_threshold_map, risk_level, operational_flow, auc_at_risk_mm, risk_factors,
)

# ── Inicializa a fila global de Jobs
job_queue = JobQueue()

# ── Constantes de serving (contrato v2 — ADR-0003, serving_contract.py) ───────
# `sem_historico_12m` / `cliente_novo_sem_contato_hist` são derivados da ausência
# de `retorno_12m_pct` / `dias_desde_ultimo_contato` — o payload não os recebe.
SHAP_CSV  = os.path.join("output", "shap", "v2", "client_explanations.csv")
DATA_CSV  = os.path.join("output", "data", "base_clientes_v2_limpo.csv")
CARTEIRA_CSV  = os.path.join("output", "data", "carteira_exposta_por_assessor.csv")
MODELS_DIR    = os.path.join("output", "models")
MODEL_V2_PKL  = os.path.join(MODELS_DIR, "gb_pipeline_v2.pkl")
MONITOR_DIR   = os.path.join("output", "monitor")

TRADUCAO = {
    "retorno_12m_pct":   "Retorno da carteira (12m)",
    "freq_contato_mes":  "Frequência de contato com assessor",
    "retorno_relativo":  "Retorno relativo ao benchmark",
    "engajamento_score": "Score de engajamento",
    "auc_milhoes":       "AuC sob custódia (R$ milhões)",
    "qtd_produtos":      "Quantidade de produtos",
    "meses_cliente":     "Tempo como cliente (meses)",
    "flag_risco":        "Flag de risco comportamental",
    "intensidade_rel":   "Intensidade de relacionamento",
    "segmento_enc":      "Segmento do cliente",
}


# ── Estado global do app ─────────────────────────────────────
_state: dict = {}


def _load_model() -> object:
    """Carrega o Pipeline v2 (early-warning comportamental, ADR-0001).

    A API expõe só a v2 — decisão travada na sessão de 2026-09-09: substitui a
    baseline reativa v1, não roda os dois lado a lado.
    """
    if not os.path.exists(MODEL_V2_PKL):
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_V2_PKL}. Execute 'python src/pipeline.py' primeiro."
        )

    meta = {
        "version": "v2",
        "algorithm": "GradientBoostingClassifier",
        "adr": "docs/adr/0001-refatoracao-early-warning-advisor-attrition.md",
        "features": FEATURES_V2_BASE,
        "notes": (
            "Direção A — early-warning comportamental. Métricas citáveis nos CSVs "
            "regeneráveis: comparacao_v1_v2.csv, cv_scores_v2.csv, feature_importance_v2.csv. "
            "Dado sintético — não é desempenho de produção."
        ),
    }
    model = joblib.load(MODEL_V2_PKL)
    return model, "v2", meta


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
        _state["threshold_map"] = load_threshold_map()
        _state["shap_df"] = _load_shap_explanations()
        _state["started_at"] = datetime.datetime.now().isoformat()
        print(f"[OK] Modelo v{version} carregado.")
    except FileNotFoundError as e:
        print(f"[WARN] {e} — API iniciada sem modelo. Execute 'python src/pipeline.py'.")
        _state["model"]   = None
        _state["version"] = "N/A"
        _state["meta"]    = {}
        _state["threshold_map"] = {}
        _state["shap_df"] = None
        _state["started_at"] = datetime.datetime.now().isoformat()
    yield
    _state.clear()


# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="Churn Finance — Prediction API",
    description=(
        "API de predição de churn de uma gestora de wealth. Modelo v2 "
        "(early-warning comportamental, ADR-0001 / PROBLEM.md v2.0): thresholds "
        "calibrados por segmento em reports/thresholds_v2.md e explicabilidade "
        "LGPD-ready via SHAP. Dado sintético."
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
    segmento: Literal["Alta Renda", "Private", "Wealth", "Family Office"] = Field(
        description="Segmento de investimento do cliente",
        examples=["Wealth"]
    )
    meses_cliente: int = Field(
        ge=6, le=600,
        description="Tempo como cliente em meses (janela de observação: últimos 90 dias)",
        examples=[36]
    )
    qtd_produtos: int = Field(
        ge=1, le=20,
        description="Quantidade de produtos financeiros ativos",
        examples=[3]
    )
    retorno_12m_pct: Optional[float] = Field(
        default=None, ge=-50.0, le=100.0,
        description=(
            "Retorno da carteira nos últimos 12 meses (%). Pode ser negativo. "
            "null = cliente sem histórico de 12m → sem_historico_12m=1 e "
            "imputação por mediana do treino."
        ),
        examples=[11.5]
    )
    freq_contato_mes: float = Field(
        ge=0.0, le=60.0,
        description="Número de contatos com assessor no último mês",
        examples=[2]
    )
    auc_milhoes: float = Field(
        gt=0.0, le=2000.0,
        description="AuC sob custódia em R$ milhões (máximo R$ 2 bi por relação)",
        examples=[120.0]
    )
    dias_desde_ultimo_contato: Optional[float] = Field(
        default=None, ge=0.0, le=400.0,
        description=(
            "Early-warning: dias desde o último contato cliente-assessor. "
            "null = cliente novo sem histórico → cliente_novo_sem_contato_hist=1."
        ),
        examples=[16.8]
    )
    variacao_freq_contato_3m: float = Field(
        ge=-1.0, le=3.0,
        description=(
            "Early-warning: variação relativa da cadência de contato nos "
            "últimos 3 meses (−0,3 = caiu 30%)."
        ),
        examples=[-0.03]
    )
    tempo_resposta_medio_horas: Optional[float] = Field(
        default=None, ge=0.0, le=400.0,
        description=(
            "Early-warning: latência média de resposta do cliente ao assessor "
            "(horas). null → imputação por mediana do treino."
        ),
        examples=[16.9]
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

def _threshold_map_or_503() -> dict[str, float]:
    thr_map = _state.get("threshold_map") or {}
    if not thr_map:
        raise HTTPException(status_code=503, detail="Thresholds v2 não carregados. Execute 'python src/pipeline.py'.")
    return thr_map


def _risk_level(prob: float, segmento: str) -> str:
    return risk_level(prob, segmento, _threshold_map_or_503())


def _threshold_for(segmento: str) -> float:
    return _threshold_map_or_503()[segmento]


def _recommended_action(risk: str, features: dict) -> str:
    if risk == "BAIXO":
        return "Monitoramento rotineiro. Nenhuma acao imediata necessaria."

    # Limiares e disparo vêm do contrato de serving (serving_contract.risk_factors);
    # aqui só o texto da ação para cada código.
    FRASES = {
        "retorno_baixo":       "Apresentar portfólio com maior CDI+ e produtos de renda variável diversificada",
        "sem_contato_recente": "Agendar call consultiva com assessor — cliente sem contato recente",
        "cadencia_caindo":     "Cadência de contato caindo — early-warning: acionar assessor antes da próxima régua",
        "monoproduto":         "Oferecer diversificação de produtos — cliente monoproduto",
        "auc_baixo":           "Avaliar incentivo de aporte mínimo ou campanha de fidelização",
    }
    actions = [FRASES[cod] for cod in risk_factors(features) if cod in FRASES]

    if not actions:
        actions.append("Contato proativo pelo assessor para entender necessidades atuais")

    return " | ".join(actions)


def _flow(segmento: str, auc_milhoes: float) -> str:
    """Fluxo operacional (PROBLEM.md §5.2) — delega ao contrato de serving."""
    return operational_flow(segmento, auc_milhoes)


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
        raise HTTPException(status_code=503, detail="Modelo não carregado. Execute 'python src/pipeline.py'.")

    # Flags de nulo estrutural derivadas da ausência do valor bruto — mesma
    # regra do treino (src/data_processing/nodes.py:316-317). O StructuralNullImputer
    # do Pipeline v2 completa retorno/dias/tempo com a mediana aprendida no treino.
    sem_historico = int(cliente.retorno_12m_pct is None)
    cliente_novo  = int(cliente.dias_desde_ultimo_contato is None)

    X = pd.DataFrame([{
        "segmento":                    cliente.segmento,
        "meses_cliente":               cliente.meses_cliente,
        "qtd_produtos":                cliente.qtd_produtos,
        "retorno_12m_pct":             cliente.retorno_12m_pct,
        "freq_contato_mes":            cliente.freq_contato_mes,
        "auc_milhoes":                 cliente.auc_milhoes,
        "dias_desde_ultimo_contato":   cliente.dias_desde_ultimo_contato,
        "variacao_freq_contato_3m":    cliente.variacao_freq_contato_3m,
        "tempo_resposta_medio_horas":  cliente.tempo_resposta_medio_horas,
        "sem_historico_12m":           sem_historico,
        "cliente_novo_sem_contato_hist": cliente_novo,
    }])[FEATURES_V2_BASE]

    prob = float(model.predict_proba(X)[0][1])
    prob = round(prob, 4)

    risk        = _risk_level(prob, cliente.segmento)
    threshold   = _threshold_for(cliente.segmento)
    predicted   = prob >= threshold
    action      = _recommended_action(risk, cliente.model_dump())
    flow_str    = _flow(cliente.segmento, cliente.auc_milhoes)
    reasons     = _get_shap_reasons(cliente.cliente_id)
    auc_risk_mm = auc_at_risk_mm(cliente.auc_milhoes, prob)

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
        return {"message": "Modelo não carregado. Execute 'python src/pipeline.py'."}
    return meta


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(cliente: ClienteInput):
    """
    Predição de churn para **um único cliente** (schema v2 early-warning).

    - Aplica o threshold calibrado do segmento (reports/thresholds_v2.md)
    - Retorna probabilidade + nível de risco + ação recomendada + fluxo operacional
    - Busca explicações SHAP pré-computadas se disponíveis

    Segmentos: Alta Renda, Private, Wealth, Family Office. Wealth / Family Office
    (ou AuC ≥ R$ 250 mi) vão para revisão humana.
    """
    return _predict_one(cliente)


def run_local_worker_task(job_id: str):
    """Worker local simplificado que roda em thread de background caso o Redis esteja off."""
    import sqlite3
    try:
        job_queue.update_job(job_id, "PROCESSING")
        
        # Carrega os dados do banco
        status_data = job_queue.get_status(job_id)
        if not status_data:
            return
            
        conn = sqlite3.connect(job_queue.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT payload FROM jobs WHERE id = ?", (job_id,))
        payload_str = cursor.fetchone()[0]
        conn.close()
        
        payload_dict = json.loads(payload_str)
        clientes_list = payload_dict.get("clientes", [])
        
        # Roda as predições usando as funções internas do api.py
        results = []
        for c_dict in clientes_list:
            cliente_obj = ClienteInput(**c_dict)
            pred_res = _predict_one(cliente_obj)
            results.append(pred_res)
            
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
        
        batch_result = {
            "total": len(results),
            "scored_at": datetime.datetime.now().isoformat(),
            "model_version": _state.get("version", "N/A"),
            "results": [r.model_dump() for r in results],
            "summary": summary
        }
        
        job_queue.update_job(job_id, "COMPLETED", result=batch_result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        job_queue.update_job(job_id, "FAILED", error=str(e))


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[BatchResult] = None


@app.post("/predict/batch", status_code=202, tags=["Prediction"])
async def predict_batch(payload: BatchInput, background_tasks: BackgroundTasks):
    """
    Inicia predição de churn em **lote** de forma assíncrona (até 1.000 clientes).

    Retorna um `job_id` com status `202 Accepted` imediatamente.
    Para obter os resultados, consulte o endpoint `/predict/batch/status/{job_id}`.
    """
    job_id = job_queue.enqueue(payload.model_dump())
    
    # Se não houver Redis conectado, usa o worker local integrado via BackgroundTasks do FastAPI
    if job_queue.redis_client is None:
        background_tasks.add_task(run_local_worker_task, job_id)
        
    return {"job_id": job_id, "status": "PENDING"}


@app.get("/predict/batch/status/{job_id}", response_model=JobStatusResponse, tags=["Prediction"])
async def get_batch_status(job_id: str):
    """
    Consulta o status e o resultado de uma predição em lote de forma assíncrona.

    Retorna status: `PENDING`, `PROCESSING`, `COMPLETED` ou `FAILED`.
    Caso status seja `COMPLETED`, inclui o campo `result` com o `BatchResult` consolidado.
    """
    job_data = job_queue.get_status(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} não encontrado.")
        
    return JobStatusResponse(
        job_id=job_id,
        status=job_data["status"],
        created_at=job_data.get("created_at"),
        completed_at=job_data.get("completed_at"),
        error=job_data.get("error"),
        result=job_data.get("result")
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
            detail="Nenhum relatório de drift encontrado. Execute 'python src/monitor.py'."
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
            detail="Explicações SHAP não disponíveis. Execute 'python src/shap_analysis_v2.py'."
        )

    df = shap_df[shap_df["churn_prob"] >= 0.35].copy()

    # O client_explanations v2 já traz `segmento`; do dataset base só falta AuC.
    if os.path.exists(DATA_CSV):
        cols = ["cliente_id", "auc_milhoes"]
        if "segmento" not in df.columns:
            cols.append("segmento")
        base = pd.read_csv(DATA_CSV)[cols]
        df   = df.merge(base, on="cliente_id", how="left")

    # Filtra por segmento apenas se a coluna existir após o merge
    if segmento and "segmento" in df.columns:
        df = df[df["segmento"] == segmento]

    df = df.sort_values("churn_prob", ascending=False).head(limit)

    clientes = []
    for _, row in df.iterrows():
        prob    = float(row["churn_prob"])
        seg     = str(row.get("segmento", "N/A"))
        auc_milhoes = float(row.get("auc_milhoes", 0))
        risk    = _risk_level(prob, seg)
        flow    = _flow(seg, auc_milhoes)
        clientes.append({
            "cliente_id"          : row["cliente_id"],
            "segmento"            : seg,
            "churn_probability"   : round(prob, 4),
            "churn_probability_pct": f"{prob*100:.1f}%",
            "risk_level"          : risk,
            "churn_real"          : int(row.get("churn_real", -1)),
            "auc_at_risk_MM"      : auc_at_risk_mm(auc_milhoes, prob),
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


@app.get("/advisors/exposed-portfolio", tags=["Analytics"])
async def advisor_exposed_portfolio(
    limit: int = Query(default=20, ge=1, le=500, description="Número máximo de assessores retornados"),
    canal: Optional[str] = Query(default=None, description="Filtrar por canal (ex: Wirehouse)"),
    apenas_risco: bool = Query(default=False, description="Só assessores com risco_saida=1"),
):
    """
    **Direção B do ADR-0001 — carteira exposta por assessor.**

    NÃO é predição de churn de cliente. É uma tabela descritiva de priorização:
    "se ESTE assessor deixar a firma, quanto AuC da carteira dele tende a
    migrar junto". Insumo para retenção de assessor, não para o classificador
    (feature importance de `auc_exposto` era 1,3%, corr com churn individual
    −0,04 — ver §6 do ADR).

    Ordenado por AuC exposto (desc). Fonte: `carteira_exposta_por_assessor.csv`,
    regenerado a cada `python src/pipeline.py`.
    """
    if not os.path.exists(CARTEIRA_CSV):
        raise HTTPException(
            status_code=404,
            detail="carteira_exposta_por_assessor.csv não encontrado. Execute 'python src/pipeline.py'.",
        )

    df = pd.read_csv(CARTEIRA_CSV)
    if canal and "canal" in df.columns:
        df = df[df["canal"] == canal]
    if apenas_risco and "risco_saida" in df.columns:
        df = df[df["risco_saida"] == 1]

    df = df.sort_values("auc_exposto_total", ascending=False).head(limit)

    assessores = [
        {
            "assessor_id"          : row["assessor_id"],
            "canal"                : str(row.get("canal", "N/A")),
            "anos_de_casa"         : float(row.get("anos_de_casa", 0)),
            "risco_saida"          : int(row.get("risco_saida", 0)),
            "qtd_clientes"         : int(row.get("qtd_clientes", 0)),
            "auc_total_carteira_milhoes": round(float(row["auc_total_carteira"]), 3),
            "auc_exposto_total_milhoes" : round(float(row["auc_exposto_total"]), 3),
            "pct_carteira_exposta" : round(float(row["pct_carteira_exposta"]), 4),
        }
        for _, row in df.iterrows()
    ]

    return {
        "total"          : len(assessores),
        "filter_canal"   : canal or "all",
        "apenas_risco"   : apenas_risco,
        "auc_exposto_total_milhoes": round(sum(a["auc_exposto_total_milhoes"] for a in assessores), 3),
        "generated_at"   : datetime.datetime.now().isoformat(),
        "assessores"     : assessores,
    }
