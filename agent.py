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
import requests
from typing import Generator

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

# ── Helpers HTTP ─────────────────────────────────────────────

def _get(endpoint: str, params: dict = None) -> dict:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "detail": "API offline. Execute: uvicorn api:app --port 8000"}


def _post(endpoint: str, body: dict) -> dict:
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=body, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "detail": "API offline. Execute: uvicorn api:app --port 8000"}


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
    if "error" in result:
        return result

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
    return _get("/clients/high-risk", params=params)


def prever_churn_cliente(
    segmento: str,
    meses_cliente: int,
    qtd_produtos: int,
    retorno_12m_pct: float,
    freq_contato_mes: int,
    saldo_bi: float,
) -> dict:
    """
    Prevê a probabilidade de churn de um cliente com o perfil fornecido.
    Retorna: probabilidade, nível de risco, ação recomendada e fluxo operacional.
    """
    return _post("/predict", {
        "cliente_id"      : "AGENT_QUERY",
        "segmento"        : segmento,
        "meses_cliente"   : meses_cliente,
        "qtd_produtos"    : qtd_produtos,
        "retorno_12m_pct" : retorno_12m_pct,
        "freq_contato_mes": freq_contato_mes,
        "saldo_bi"        : saldo_bi,
    })


def status_modelo() -> dict:
    """
    Retorna as métricas e metadados do modelo em produção:
    versão, F1-macro, ROC-AUC, taxa de churn da base e feature mais importante.
    """
    return _get("/model/info")


def alertas_drift() -> dict:
    """
    Verifica o status de Data Drift — se o modelo está 'envelhecendo'
    e precisa ser re-treinado. Retorna: status (OK/ATENÇÃO/CRÍTICO) e alertas.
    """
    r = _get("/monitor/latest")
    if "error" in r:
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
    Estima o AuC que pode ser salvo com ações de retenção.
    Meta do PROBLEM.md: 15% de AuC retido em 90 dias vs. grupo de controle.
    """
    dados = consultar_auc_segmento(segmento)
    if "error" in dados:
        return dados

    total_auc = dados["auc_total_em_risco_MM"]
    economia  = round(total_auc * taxa_retencao_pct / 100, 2)

    return {
        "segmento"              : segmento,
        "auc_em_risco_MM"       : total_auc,
        "taxa_retencao_aplicada": f"{taxa_retencao_pct}%",
        "auc_salvo_estimado_MM" : economia,
        "clientes_em_risco"     : dados["total_clientes_em_risco"],
        "clientes_alto_risco"   : dados["clientes_alto_risco"],
        "baseline_contratual"   : "PROBLEM.md Seção 6.2: +15% AuC retido em 90 dias vs. controle",
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
                        "enum": ["Varejo", "Alta Renda", "Wealth", "Corporate", "todos"],
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
                        "enum": ["Varejo", "Alta Renda", "Wealth", "Corporate", "todos"],
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
            "description": "Prevê a probabilidade de churn de um cliente com o perfil informado. Use quando o usuário descrever um cliente hipotético.",
            "parameters": {
                "type": "object",
                "properties": {
                    "segmento"        : {"type": "string", "enum": ["Varejo", "Alta Renda", "Wealth", "Corporate"]},
                    "meses_cliente"   : {"type": "integer",  "description": "Tempo como cliente em meses"},
                    "qtd_produtos"    : {"type": "integer",  "description": "Quantidade de produtos ativos"},
                    "retorno_12m_pct" : {"type": "number",   "description": "Retorno da carteira em 12 meses (%)"},
                    "freq_contato_mes": {"type": "integer",  "description": "Contatos com assessor por mês"},
                    "saldo_bi"        : {"type": "number",   "description": "Saldo em R$ bilhões"},
                },
                "required": ["segmento","meses_cliente","qtd_produtos","retorno_12m_pct","freq_contato_mes","saldo_bi"]
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
            "description": "Estima o AuC que pode ser salvo com ações de retenção proativas em um segmento.",
            "parameters": {
                "type": "object",
                "properties": {
                    "segmento": {
                        "type": "string",
                        "enum": ["Varejo", "Alta Renda", "Wealth", "Corporate", "todos"]
                    },
                    "taxa_retencao_pct": {
                        "type": "number",
                        "description": "Taxa de retenção esperada (%). Default: 15 conforme PROBLEM.md",
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

Você tem acesso a um modelo Gradient Boosting em produção (ROC-AUC 0.93) que monitora clientes com base em comportamento de carteira, contato com assessores e retorno relativo.

**Regras de comportamento:**
1. SEMPRE chame as ferramentas para buscar dados atualizados antes de responder — nunca invente números.
2. Quantifique em R$ sempre que possível. Fale o idioma do negócio, não de data science.
3. Seja direto e executivo. Responda como um Chief Data Officer em reunião de diretoria.
4. Quando identificar clientes em risco, sugira a ação de retenção adequada ao segmento.
5. Se a pergunta não puder ser respondida com as ferramentas disponíveis, informe claramente.
6. Use formatação markdown: **negrito** para números importantes, tabelas quando útil.
7. Responda em Português do Brasil."""


# ═══════════════════════════════════════════════════════════
# MODO DEMO — funciona sem OpenAI API Key
# Usa intent classifier + ferramentas reais
# ═══════════════════════════════════════════════════════════

def _detect_intent(question: str) -> tuple[str, dict]:
    """Classifica a intenção da pergunta para o modo demo."""
    q = question.lower()

    seg = "todos"
    if "wealth"      in q: seg = "Wealth"
    elif "alta renda" in q: seg = "Alta Renda"
    elif "varejo"     in q: seg = "Varejo"
    elif "corporate"  in q: seg = "Corporate"

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
            f"Com base nos dados atuais {seg_text}:\n\n"
            f"- **AuC total em risco:** R$ {result.get('auc_em_risco_MM', 0):.1f}M\n"
            f"- **Clientes em risco:** {result.get('clientes_em_risco', 0)}\n"
            f"- **Estimativa de AuC salvo** (meta {result.get('taxa_retencao_aplicada','15%')}): "
            f"**R$ {result.get('auc_salvo_estimado_MM', 0):.1f}M**\n\n"
            f"Esta estimativa segue o critério contratual do PROBLEM.md: "
            f"+15% de AuC retido em 90 dias vs. grupo de controle (A/B Test).\n\n"
            f"> Recomendação: Acione os assessores para os **{result.get('clientes_alto_risco', 0)} "
            f"clientes de alto risco** imediatamente."
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
        d = result.get("data_profile", {})
        return (
            f"**Modelo em produção: v{result.get('version','?')}**\n\n"
            f"| Métrica | Valor | Meta |\n"
            f"|---------|-------|------|\n"
            f"| F1-macro | **{m.get('f1_macro','?')}** | ≥ 0.55 |\n"
            f"| ROC-AUC | **{m.get('roc_auc','?')}** | ≥ 0.80 |\n"
            f"| Threshold | {m.get('threshold','?')} | — |\n\n"
            f"- **Amostras de treino:** {d.get('n_samples','?')} clientes\n"
            f"- **Taxa de churn base:** {d.get('churn_rate_pct','?')}%\n"
            f"- **Feature mais importante (SHAP):** `{d.get('top_shap_feature','?')}`\n"
            f"- **Promovido em:** {result.get('promoted_at','?')[:10]}"
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
