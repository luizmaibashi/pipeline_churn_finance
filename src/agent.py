# =============================================================
# agent.py — Data Agent Core: Churn Finance
# Fase 3 do Roadmap: Visão Agêntica
#
# Arquitetura:
#   agent_chat.py (UI) → agent.py (LLM + Tools) → api.py (FastAPI) → Modelo
#
# Funciona em dois modos:
#   - FULL MODE: OpenAI function calling (requer OPENAI_API_KEY no .env)
#   - DEMO MODE: Intent classifier + tools reais (sem API key, zero custo)
#
# Setup: copie .env.example para .env e adicione OPENAI_API_KEY
# =============================================================

import os
import json
import datetime
import requests
from functools import lru_cache

import pandas as pd

from src.serving_contract import (
    SEGMENTOS_VALIDOS as SEGMENTOS_V2, load_threshold_map, risk_level,
    operational_flow, auc_at_risk_mm,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Configuração ─────────────────────────────────────────────
API_BASE       = os.getenv("API_BASE",       "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL      = os.getenv("LLM_MODEL",      "gpt-4o-mini")
DEMO_MODE      = not bool(OPENAI_API_KEY)

# Contrato de scoring v2 em serving_contract.py (ADR-0003). Aqui só os caminhos
# offline que reproduzem, sobre o disco, o que a API faria.
_SEG_ENUM      = SEGMENTOS_V2 + ["todos"]   # enums das ferramentas que aceitam "carteira toda"
SHAP_V2_CSV    = os.path.join("output", "shap", "v2", "client_explanations.csv")
BASE_V2_CSV    = os.path.join("output", "data", "base_clientes_v2_limpo.csv")
COMPARACAO_CSV = os.path.join("output", "data", "comparacao_v1_v2.csv")


@lru_cache(maxsize=1)
def _threshold_map() -> dict:
    """{segmento: threshold} calibrado — lido uma vez por processo (cache), pois
    os fallbacks offline pontuam linha a linha. Vazio se o pipeline não rodou."""
    try:
        return load_threshold_map()
    except (FileNotFoundError, ValueError) as e:
        print(f"[Agent] thresholds v2 indisponíveis ({e}). Rode 'python src/pipeline.py'.")
        return {}


def _risco_offline(prob: float, seg: str, thr_map: dict) -> str:
    return risk_level(prob, seg, thr_map) if seg in thr_map else "N/A"


def _carregar_base_risco_offline(segmento: str) -> "pd.DataFrame | None":
    """Fallback quando a API está fora: SHAP v2 + AuC do book limpo, filtrado por
    `churn_prob >= 0.35` e por segmento. Usado por `consultar_auc_segmento` e
    `listar_clientes_prioritarios` — mantém a leitura de disco num lugar só."""
    if not os.path.exists(SHAP_V2_CSV):
        return None
    df = pd.read_csv(SHAP_V2_CSV)
    df = df[df["churn_prob"] >= 0.35].copy()
    if os.path.exists(BASE_V2_CSV):
        cols = ["cliente_id", "auc_milhoes"] + (["segmento"] if "segmento" not in df.columns else [])
        df = df.merge(pd.read_csv(BASE_V2_CSV, usecols=cols), on="cliente_id", how="left")
    if segmento and segmento.lower() != "todos" and "segmento" in df.columns:
        df = df[df["segmento"] == segmento]
    return df

# ── Helpers HTTP ─────────────────────────────────────────────

def _get(endpoint: str, params: dict = None) -> dict:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "detail": "API offline. Execute: uvicorn src.api:app --port 8000"}


def _post(endpoint: str, body: dict) -> dict:
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=body, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "detail": "API offline. Execute: uvicorn src.api:app --port 8000"}


# ═══════════════════════════════════════════════════════════
# FERRAMENTAS DO AGENTE
# Cada função é um "tool" que o LLM pode chamar
# ═══════════════════════════════════════════════════════════

def consultar_auc_segmento(segmento: str) -> dict:
    """
    Consulta o AuC (Assets under Custody) total em risco de churn
    em um segmento específico ou em toda a carteira.
    """
    params = {"limit": 500}
    if segmento and segmento.lower() != "todos":
        params["segmento"] = segmento

    result = _get("/clients/high-risk", params=params)
    
    # Fallback offline para leitura direta do disco local
    if "error" in result:
        print("[Agent Fallback] API indisponível. Carregando dados de AuC diretamente do disco local...")
        df = _carregar_base_risco_offline(segmento)
        thr_map = _threshold_map()
        if df is None or not thr_map:
            return result
        clientes = [
            {
                "cliente_id"    : row["cliente_id"],
                "risk_level"    : _risco_offline(float(row["churn_prob"]),
                                                 str(row.get("segmento", "N/A")), thr_map),
                "auc_at_risk_MM": auc_at_risk_mm(float(row.get("auc_milhoes", 0)),
                                                 float(row["churn_prob"])),
            }
            for _, row in df.iterrows()
        ]
        result = {"clientes": clientes}

    clientes = result.get("clientes", [])
    alto  = [c for c in clientes if c.get("risk_level") == "ALTO"]
    medio = [c for c in clientes if c.get("risk_level") == "MEDIO"]

    auc_total = sum(c.get("auc_at_risk_MM", 0) for c in clientes)
    auc_alto  = sum(c.get("auc_at_risk_MM", 0) for c in alto)

    return {
        "segmento"               : segmento,
        "total_clientes_em_risco": len(clientes),
        "clientes_alto_risco"    : len(alto),
        "clientes_medio_risco"   : len(medio),
        "auc_total_em_risco_MM"  : round(auc_total, 2),
        "auc_alto_risco_MM"      : round(auc_alto,  2),
        "top3_clientes"          : clientes[:3],
    }


def listar_clientes_prioritarios(segmento: str = "todos", limit: int = 10) -> dict:
    """
    Lista os clientes mais em risco de churn, em ordem de prioridade.
    Inclui probabilidade, AuC em risco, fluxo operacional e ação recomendada.
    """
    params = {"limit": limit}
    if segmento and segmento.lower() != "todos":
        params["segmento"] = segmento
    result = _get("/clients/high-risk", params=params)
    
    # Fallback offline para leitura direta do disco local
    if "error" in result:
        print("[Agent Fallback] API indisponível. Carregando lista de prioridade diretamente do disco local...")
        df = _carregar_base_risco_offline(segmento)
        thr_map = _threshold_map()
        if df is None or not thr_map:
            return result
        df = df.sort_values("churn_prob", ascending=False).head(limit)
        clientes = []
        for _, row in df.iterrows():
            prob = float(row["churn_prob"])
            seg = str(row.get("segmento", "N/A"))
            auc_milhoes = float(row.get("auc_milhoes", 0))
            clientes.append({
                "cliente_id"          : row["cliente_id"],
                "segmento"            : seg,
                "churn_probability"   : round(prob, 4),
                "churn_probability_pct": f"{prob*100:.1f}%",
                "risk_level"          : _risco_offline(prob, seg, thr_map),
                "churn_real"          : int(row.get("churn_real", -1)),
                "auc_at_risk_MM"      : auc_at_risk_mm(auc_milhoes, prob),
                "flow"                : operational_flow(seg, auc_milhoes),
                "explicacao"          : str(row.get("explicacao", "")),
            })
        return {
            "total"         : len(clientes),
            "filter_segment": segmento,
            "model_version" : "flat (offline)",
            "generated_at"  : datetime.datetime.now().isoformat(),
            "clientes"      : clientes,
        }
    return result


def prever_churn_cliente(
    segmento: str,
    meses_cliente: int,
    qtd_produtos: int,
    freq_contato_mes: float,
    auc_milhoes: float,
    variacao_freq_contato_3m: float,
    retorno_12m_pct: float | None = None,
    dias_desde_ultimo_contato: float | None = None,
    tempo_resposta_medio_horas: float | None = None,
) -> dict:
    """
    Prevê a probabilidade de churn de um cliente com o perfil fornecido (schema v2).
    `retorno_12m_pct`, `dias_desde_ultimo_contato` e `tempo_resposta_medio_horas`
    são opcionais (nulo estrutural = cliente sem histórico).
    Retorna: probabilidade, nível de risco, ação recomendada e fluxo operacional.
    """
    return _post("/predict", {
        "cliente_id"                : "AGENT_QUERY",
        "segmento"                  : segmento,
        "meses_cliente"             : meses_cliente,
        "qtd_produtos"              : qtd_produtos,
        "retorno_12m_pct"           : retorno_12m_pct,
        "freq_contato_mes"          : freq_contato_mes,
        "auc_milhoes"               : auc_milhoes,
        "dias_desde_ultimo_contato" : dias_desde_ultimo_contato,
        "variacao_freq_contato_3m"  : variacao_freq_contato_3m,
        "tempo_resposta_medio_horas": tempo_resposta_medio_horas,
    })


def status_modelo() -> dict:
    """
    Retorna as métricas e metadados do modelo v2 em produção: versão, ROC-AUC,
    recall de churn e F1 de churn no split de teste (lidos de comparacao_v1_v2.csv).
    """
    result = _get("/model/info")
    
    # Fallback offline: lê as métricas v2 do CSV regenerável (comparacao_v1_v2.csv),
    # nunca números hardcoded — acceptance do Bloco 4 da spec 0002.
    if "error" in result:
        print("[Agent Fallback] API indisponível. Carregando metadados do modelo diretamente do disco local...")
        try:
            cmp = pd.read_csv(COMPARACAO_CSV)
            v2 = cmp[cmp["modelo"] == "v2_early_warning_advisor"].iloc[0]
            return {
                "message": "Modelo v2 (early-warning comportamental, ADR-0001) — leitura offline.",
                "version": "v2",
                "metrics": {
                    "roc_auc": round(float(v2["roc_auc"]), 4),
                    "recall_churn": round(float(v2["recall_churn"]), 4),
                    "f1_churn": round(float(v2["f1_churn"]), 4),
                    "n_teste": int(v2["n_teste"]),
                },
            }
        except Exception as e:
            return {"error": f"comparacao_v1_v2.csv indisponível: {e}. Execute python src/pipeline.py."}
    return result


def alertas_drift() -> dict:
    """
    Verifica o status de Data Drift — se o modelo está 'envelhecendo'
    e precisa ser re-treinado. Retorna: status (OK/ATENÇÃO/CRÍTICO) e alertas.
    """
    r = _get("/monitor/latest")
    
    # Fallback offline para leitura direta do disco local
    if "error" in r:
        print("[Agent Fallback] API indisponível. Carregando alertas de drift diretamente do disco local...")
        monitor_dir = os.path.join("output", "monitor")
        if os.path.exists(monitor_dir):
            import glob
            reports = sorted([
                f for f in os.listdir(monitor_dir)
                if f.startswith("drift_report_") and f.endswith(".json")
            ], reverse=True)
            if reports:
                with open(os.path.join(monitor_dir, reports[0]), encoding="utf-8") as f:
                    r = json.load(f)
            else:
                return r
        else:
            return r

    return {
        "status"           : r.get("summary", {}).get("status"),
        "alertas"          : r.get("summary", {}).get("alerts", []),
        "n_alertas"        : r.get("summary", {}).get("n_alerts", 0),
        "retraining_needed": r.get("summary", {}).get("retraining_needed", False),
        "recomendacao"     : r.get("summary", {}).get("recommendation"),
        "run_at"           : r.get("run_at"),
    }


def economia_auc_segmento(segmento: str, taxa_retencao_pct: float = 15.0) -> dict:
    """
    Calculadora what-if: aplica uma taxa de retenção HIPOTÉTICA (fornecida pelo
    usuário) sobre o AuC em risco para dar ordem de grandeza. NÃO é um efeito
    medido — o dado é sintético e o projeto não estima retenção real
    (ADR-0001 §5, PROBLEM.md v2.0 §7).
    """
    dados = consultar_auc_segmento(segmento)
    if "error" in dados:
        return dados

    total_auc = dados["auc_total_em_risco_MM"]
    economia  = round(total_auc * taxa_retencao_pct / 100, 2)

    return {
        "segmento"              : segmento,
        "auc_em_risco_MM"       : total_auc,
        "taxa_retencao_hipotetica": f"{taxa_retencao_pct}%",
        "auc_salvo_hipotetico_MM" : economia,
        "clientes_em_risco"     : dados["total_clientes_em_risco"],
        "clientes_alto_risco"   : dados["clientes_alto_risco"],
        "nota"                  : "Cenário ilustrativo sobre dado sintético; não é retenção observada.",
    }


# ── Mapa de ferramentas ───────────────────────────────────────
TOOL_FUNCTIONS = {
    "consultar_auc_segmento"    : consultar_auc_segmento,
    "listar_clientes_prioritarios": listar_clientes_prioritarios,
    "prever_churn_cliente"      : prever_churn_cliente,
    "status_modelo"             : status_modelo,
    "alertas_drift"             : alertas_drift,
    "economia_auc_segmento"     : economia_auc_segmento,
}

# ── Schemas para OpenAI function calling ─────────────────────
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "consultar_auc_segmento",
            "description": "Consulta o AuC (Assets under Custody) total em risco de churn em um segmento ou na carteira toda.",
            "parameters": {
                "type": "object",
                "properties": {
                    "segmento": {
                        "type": "string",
                        "enum": _SEG_ENUM,
                        "description": "Segmento a consultar. Use 'todos' para carteira completa."
                    }
                },
                "required": ["segmento"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "listar_clientes_prioritarios",
            "description": "Lista os clientes mais em risco de churn em ordem de prioridade para ação do assessor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "segmento": {
                        "type": "string",
                        "enum": _SEG_ENUM,
                        "description": "Filtrar por segmento. Use 'todos' para todos os segmentos."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Quantos clientes retornar (máx. 50)",
                        "default": 10
                    }
                },
                "required": ["segmento"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "prever_churn_cliente",
            "description": "Prevê a probabilidade de churn de um cliente (schema v2 early-warning). Use quando o usuário descrever um cliente hipotético.",
            "parameters": {
                "type": "object",
                "properties": {
                    "segmento"        : {"type": "string", "enum": SEGMENTOS_V2},
                    "meses_cliente"   : {"type": "integer", "description": "Tempo como cliente em meses (mínimo 6)"},
                    "qtd_produtos"    : {"type": "integer", "description": "Quantidade de produtos ativos"},
                    "freq_contato_mes": {"type": "number",  "description": "Contatos com assessor no último mês"},
                    "auc_milhoes"     : {"type": "number",  "description": "AuC sob custódia em R$ milhões"},
                    "variacao_freq_contato_3m": {"type": "number", "description": "Early-warning: variação relativa da cadência de contato nos últimos 3 meses (-0,3 = caiu 30%)"},
                    "retorno_12m_pct" : {"type": "number",  "description": "Retorno da carteira em 12 meses (%). Omitir se o cliente não tem histórico de 12m"},
                    "dias_desde_ultimo_contato": {"type": "number", "description": "Early-warning: dias desde o último contato cliente-assessor. Omitir se cliente novo"},
                    "tempo_resposta_medio_horas": {"type": "number", "description": "Early-warning: latência média de resposta do cliente ao assessor (horas). Opcional"},
                },
                "required": ["segmento","meses_cliente","qtd_produtos","freq_contato_mes","auc_milhoes","variacao_freq_contato_3m"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "status_modelo",
            "description": "Retorna métricas e metadados do modelo ML em produção: versão, F1-macro, ROC-AUC, data de treino.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "alertas_drift",
            "description": "Verifica se o modelo está envelhecendo (Data Drift). Retorna status de alerta e recomendação de re-treino.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "economia_auc_segmento",
            "description": "Cenário what-if ILUSTRATIVO: aplica uma taxa de retenção hipotética (fornecida pelo usuário) sobre o AuC em risco de um segmento. Não é retenção observada — o dado é sintético.",
            "parameters": {
                "type": "object",
                "properties": {
                    "segmento": {
                        "type": "string",
                        "enum": _SEG_ENUM
                    },
                    "taxa_retencao_pct": {
                        "type": "number",
                        "description": "Taxa de retenção hipotética (%) para o cenário what-if. Default: 15",
                        "default": 15.0
                    }
                },
                "required": ["segmento"]
            }
        }
    },
]

# ── System Prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """Você é o **Data Agent** de uma gestora de investimentos.
Sua função é responder perguntas estratégicas de Diretores e Gestores sobre churn de clientes, risco de AuC e performance do modelo de ML.

Você tem acesso a um modelo Gradient Boosting v2 (early-warning comportamental, ADR-0001) que
prioriza sinais de relacionamento — dias desde o último contato, variação de cadência e latência
de resposta — antes da queda de AuC. O dado é sintético; as métricas medem a coerência do
pipeline, não desempenho em produção. Consulte `status_modelo` para os números atuais (ROC-AUC e
recall vêm do CSV regenerável, nunca de memória).

**Regras de comportamento:**
1. SEMPRE chame as ferramentas para buscar dados atualizados antes de responder — nunca invente números.
2. Quantifique em R$ (milhões de AuC) sempre que possível. Fale o idioma do negócio, não de data science.
3. Seja direto e executivo. Responda como um Chief Data Officer em reunião de diretoria.
4. Quando identificar clientes em risco, sugira a ação de retenção adequada ao segmento.
5. Se a pergunta não puder ser respondida com as ferramentas disponíveis, informe claramente.
6. Não apresente cenários what-if de retenção como efeito medido — o projeto não estima retenção real.
7. Use formatação markdown: **negrito** para números importantes, tabelas quando útil.
8. Responda em Português do Brasil."""


# ═══════════════════════════════════════════════════════════
# MODO DEMO — funciona sem OpenAI API Key
# Usa intent classifier + ferramentas reais
# ═══════════════════════════════════════════════════════════

def _detect_intent(question: str) -> tuple[str, dict]:
    """Classifica a intenção da pergunta para o modo demo."""
    q = question.lower()

    seg = "todos"
    if "family office" in q or "family-office" in q: seg = "Family Office"
    elif "wealth"     in q: seg = "Wealth"
    elif "private"    in q: seg = "Private"
    elif "alta renda" in q: seg = "Alta Renda"

    if any(w in q for w in ["salvar", "salvando", "economi", "reter", "retenção"]):
        return "economia", {"segmento": seg}
    if any(w in q for w in ["auc", "custódia", "custody", "ativo"]):
        return "auc", {"segmento": seg}
    if any(w in q for w in ["lista", "clientes", "priorit", "fila"]):
        return "listar", {"segmento": seg, "limit": 5}
    if any(w in q for w in ["drift", "envelhecen", "alerta", "re-treino", "retreino"]):
        return "drift", {}
    if any(w in q for w in ["modelo", "performance", "acurácia", "roc", "f1", "métrica"]):
        return "modelo", {}
    return "auc", {"segmento": seg}


def _format_demo_response(intent: str, args: dict, result: dict) -> str:
    """Gera resposta em linguagem natural para o modo demo."""
    if "error" in result:
        return f"Não consegui conectar à API. {result.get('detail', '')}"

    seg_label = args.get("segmento", "todos")
    seg_text  = f"no segmento **{seg_label}**" if seg_label != "todos" else "em toda a carteira"

    if intent == "economia":
        return (
            f"Cenário **ilustrativo** {seg_text} (dado sintético, não é retenção observada):\n\n"
            f"- **AuC total em risco:** R$ {result.get('auc_em_risco_MM', 0):.1f}M\n"
            f"- **Clientes em risco:** {result.get('clientes_em_risco', 0)}\n"
            f"- **AuC salvo se a retenção fosse {result.get('taxa_retencao_hipotetica','15%')}** "
            f"(hipótese, não medição): **R$ {result.get('auc_salvo_hipotetico_MM', 0):.1f}M**\n\n"
            f"> Recomendação: priorizar contato dos assessores com os "
            f"**{result.get('clientes_alto_risco', 0)} clientes de alto risco**."
        )
    elif intent == "auc":
        top3 = result.get("top3_clientes", [])
        tabela = ""
        if top3:
            tabela = "\n\n**Top 3 clientes mais críticos:**\n"
            tabela += "| Cliente | Prob. Churn | AuC em Risco | Fluxo |\n"
            tabela += "|---------|-------------|--------------|-------|\n"
            for c in top3:
                tabela += (f"| {c.get('cliente_id')} | {c.get('churn_probability_pct','?')} "
                           f"| R${c.get('auc_at_risk_MM',0):.1f}M | {c.get('flow','?')} |\n")
        return (
            f"Situação atual {seg_text}:\n\n"
            f"- **{result.get('total_clientes_em_risco', 0)}** clientes em risco de churn\n"
            f"- **{result.get('clientes_alto_risco', 0)}** em alto risco (acima do threshold)\n"
            f"- **AuC em risco total:** R$ {result.get('auc_total_em_risco_MM', 0):.1f}M\n"
            f"- **AuC em alto risco:** R$ {result.get('auc_alto_risco_MM', 0):.1f}M"
            + tabela
        )
    elif intent == "listar":
        clientes = result.get("clientes", [])
        if not clientes:
            return f"Nenhum cliente em risco identificado {seg_text}."
        linhas = [f"**{len(clientes)} clientes prioritários** {seg_text}:\n"]
        for i, c in enumerate(clientes, 1):
            linhas.append(
                f"{i}. **{c.get('cliente_id')}** | {c.get('churn_probability_pct','?')} | "
                f"R${c.get('auc_at_risk_MM',0):.1f}M em risco | _{c.get('flow','?')}_"
            )
        return "\n".join(linhas)
    elif intent == "drift":
        status = result.get("status", "N/A")
        alertas = result.get("alertas", [])
        emoji = {"OK": "✅", "ATENCAO": "⚠️", "CRITICO": "🔴"}.get(status, "❓")
        resp = f"{emoji} **Status do modelo: {status}**\n\n"
        if alertas:
            resp += "**Alertas detectados:**\n"
            for a in alertas:
                resp += f"- {a}\n"
        resp += f"\n**Recomendação:** {result.get('recomendacao', 'N/A')}"
        return resp
    elif intent == "modelo":
        m = result.get("metrics", {})
        n_teste = m.get("n_teste", "?")
        return (
            f"**Modelo em produção: {result.get('version','?')}** "
            f"(early-warning comportamental, ADR-0001)\n\n"
            f"| Métrica (split de teste) | Valor |\n"
            f"|---------|-------|\n"
            f"| ROC-AUC | **{m.get('roc_auc','?')}** |\n"
            f"| Recall (churn) | **{m.get('recall_churn','?')}** |\n"
            f"| F1 (churn) | {m.get('f1_churn','?')} |\n\n"
            f"- Base de teste: n={n_teste} relações. Dado sintético — a métrica mede a "
            f"coerência do pipeline, não desempenho em produção.\n"
            f"- Thresholds calibrados por segmento em `reports/thresholds_v2.md`."
        )
    return json.dumps(result, indent=2, ensure_ascii=False)


def run_demo_turn(question: str) -> tuple[str, list[dict]]:
    """Executa uma volta do agente em DEMO MODE (sem OpenAI)."""
    intent, args = _detect_intent(question)

    tool_calls_log = []

    if intent in ("auc", "economia"):
        fn_name  = "consultar_auc_segmento" if intent == "auc" else "economia_auc_segmento"
        fn_args  = {"segmento": args.get("segmento", "todos")}
        result   = TOOL_FUNCTIONS[fn_name](**fn_args)
    elif intent == "listar":
        fn_name  = "listar_clientes_prioritarios"
        fn_args  = args
        result   = TOOL_FUNCTIONS[fn_name](**fn_args)
    elif intent == "drift":
        fn_name  = "alertas_drift"
        fn_args  = {}
        result   = TOOL_FUNCTIONS[fn_name]()
    else:
        fn_name  = "status_modelo"
        fn_args  = {}
        result   = TOOL_FUNCTIONS[fn_name]()

    tool_calls_log.append({
        "tool"  : fn_name,
        "args"  : fn_args,
        "result": result,
    })

    response = _format_demo_response(intent, fn_args, result)
    return response, tool_calls_log


# ═══════════════════════════════════════════════════════════
# MODO FULL — OpenAI function calling
# ═══════════════════════════════════════════════════════════

def run_llm_turn(messages: list) -> tuple[str, list[dict]]:
    """Executa uma volta do agente com OpenAI function calling."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    tool_calls_log = []
    current_messages = list(messages)

    while True:
        response = client.chat.completions.create(
            model    = LLM_MODEL,
            messages = current_messages,
            tools    = TOOLS_SCHEMA,
            tool_choice = "auto",
        )

        msg = response.choices[0].message

        # Sem tool calls → resposta final
        if not msg.tool_calls:
            return msg.content or "", tool_calls_log

        # Processa cada tool call
        current_messages.append(msg)

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            fn = TOOL_FUNCTIONS.get(fn_name)
            if fn is None:
                result = {"error": f"Ferramenta '{fn_name}' não encontrada"}
            else:
                result = fn(**fn_args)

            tool_calls_log.append({
                "tool"  : fn_name,
                "args"  : fn_args,
                "result": result,
            })

            current_messages.append({
                "role"        : "tool",
                "tool_call_id": tc.id,
                "content"     : json.dumps(result, ensure_ascii=False),
            })


def run_agent(question: str, history: list = None) -> tuple[str, list[dict]]:
    """
    Ponto de entrada principal do agente.
    Seleciona automaticamente FULL ou DEMO MODE.

    Args:
        question: Pergunta do usuário
        history:  Histórico de mensagens [{"role": ..., "content": ...}]

    Returns:
        (resposta_texto, tool_calls_log)
    """
    if DEMO_MODE:
        return run_demo_turn(question)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    return run_llm_turn(messages)
