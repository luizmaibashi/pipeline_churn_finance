# =============================================================
# agent_chat.py — Interface de Chat do Data Agent
# Fase 3 do Roadmap: Visão Agêntica
#
# Uso: streamlit run agent_chat.py
#
# Modos:
#   DEMO MODE: sem API key — usa intent classifier + dados reais
#   FULL MODE: com OPENAI_API_KEY no .env — LLM completo
# =============================================================

import streamlit as st
import sys
import os

# ── Configuração da página ────────────────────────────────────
st.set_page_config(
    page_title="Data Agent | Churn Finance",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Premium ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0f1117; }

/* Chat container */
.chat-wrap {
    max-width: 820px;
    margin: 0 auto;
    padding: 0 8px;
}

/* Mensagem do usuário */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 16px 0 4px 0;
}
.bubble-user {
    background: linear-gradient(135deg, #4c1d95, #6d28d9);
    color: #ede9fe;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    max-width: 78%;
    font-size: 14px;
    line-height: 1.6;
    box-shadow: 0 2px 12px rgba(109,40,217,0.3);
}

/* Mensagem do agente */
.msg-agent {
    display: flex;
    justify-content: flex-start;
    margin: 4px 0 16px 0;
    gap: 10px;
    align-items: flex-start;
}
.agent-avatar {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #1e3a5f, #1971c2);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0; margin-top: 2px;
}
.bubble-agent {
    background: #1e2130;
    border: 1px solid #2d3250;
    color: #c1c8de;
    border-radius: 4px 18px 18px 18px;
    padding: 14px 18px;
    max-width: 82%;
    font-size: 14px;
    line-height: 1.7;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.bubble-agent strong { color: #e8ecf4; }
.bubble-agent code {
    background: #0f1117;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 12px;
    color: #74c0fc;
}
.bubble-agent table {
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0;
    font-size: 13px;
}
.bubble-agent th {
    background: #252a3d;
    color: #8b95b0;
    padding: 6px 12px;
    text-align: left;
    font-weight: 500;
}
.bubble-agent td {
    padding: 5px 12px;
    border-bottom: 1px solid #2d3250;
    color: #c1c8de;
}
.bubble-agent blockquote {
    border-left: 3px solid #4c6ef5;
    margin: 8px 0;
    padding: 4px 12px;
    color: #8b95b0;
    font-size: 13px;
}

/* Tool call badge */
.tool-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(76,110,245,0.1);
    border: 1px solid rgba(76,110,245,0.3);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    color: #74c0fc;
    margin: 2px 4px 2px 0;
    font-family: 'Courier New', monospace;
}

/* Thinking indicator */
.thinking-box {
    background: rgba(76,110,245,0.06);
    border: 1px solid rgba(76,110,245,0.2);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-size: 12px;
    color: #6b7590;
}

/* Quick question chips */
.chip {
    display: inline-block;
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    color: #8b95b0;
    cursor: pointer;
    margin: 3px 0;
    transition: all 0.2s;
    width: 100%;
    text-align: left;
}
.chip:hover { border-color: #4c6ef5; color: #e8ecf4; }

/* Mode badge */
.mode-badge-demo {
    display: inline-block;
    background: rgba(255,212,59,0.1);
    border: 1px solid rgba(255,212,59,0.4);
    color: #ffd43b;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
}
.mode-badge-full {
    display: inline-block;
    background: rgba(81,207,102,0.1);
    border: 1px solid rgba(81,207,102,0.4);
    color: #51cf66;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141824 0%, #1a1f2e 100%);
    border-right: 1px solid #2d3250;
}

/* Input box */
.stChatInput > div {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Imports do agente ─────────────────────────────────────────
try:
    from agent import run_agent, DEMO_MODE, consultar_auc_segmento, alertas_drift
    AGENT_OK = True
except ImportError as e:
    AGENT_OK = False
    IMPORT_ERROR = str(e)


# ── Session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_logs" not in st.session_state:
    st.session_state.tool_logs = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    # Header
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px 0;'>
        <div style='font-size:42px;'>🤖</div>
        <div style='font-size:17px; font-weight:700; color:#e8ecf4; margin-top:8px;'>Data Agent</div>
        <div style='font-size:12px; color:#8b95b0; margin-top:4px;'>Churn Finance — Fase 3</div>
    </div>
    <hr style='border-color:#2d3250; margin:14px 0;'>
    """, unsafe_allow_html=True)

    # Modo do agente
    if AGENT_OK:
        if DEMO_MODE:
            st.markdown("""
            <div style='margin-bottom:8px;'>
                <span class='mode-badge-demo'>⚡ DEMO MODE</span>
            </div>
            <div style='font-size:11px; color:#8b95b0; line-height:1.6;'>
                Intent classifier + dados reais.<br>
                Adicione <code>OPENAI_API_KEY</code> no <code>.env</code> para modo completo.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='margin-bottom:8px;'>
                <span class='mode-badge-full'>✅ FULL MODE</span>
            </div>
            <div style='font-size:11px; color:#8b95b0;'>OpenAI function calling ativo.</div>
            """, unsafe_allow_html=True)
    else:
        st.error("agent.py não encontrado.")

    st.markdown("<hr style='border-color:#2d3250; margin:14px 0;'>", unsafe_allow_html=True)

    # Status do modelo (widget)
    st.markdown("<div style='color:#8b95b0; font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;'>Status do Modelo</div>", unsafe_allow_html=True)
    try:
        import requests
        info = requests.get("http://localhost:8000/model/info", timeout=3).json()
        drift = requests.get("http://localhost:8000/monitor/latest", timeout=3).json()
        drift_status = drift.get("summary", {}).get("status", "N/A")
        drift_color  = {"OK": "#51cf66", "ATENCAO": "#ffd43b", "CRITICO": "#ff6b6b"}.get(drift_status, "#8b95b0")
        st.markdown(f"""
        <div style='background:#1e2130; border:1px solid #2d3250; border-radius:10px; padding:12px 14px; font-size:12px;'>
            <div style='color:#e8ecf4; font-weight:600;'>v{info.get('version','?')}</div>
            <div style='color:#8b95b0; margin-top:4px;'>ROC-AUC: <b style='color:#74c0fc;'>{info.get('metrics',{{}}).get('roc_auc','?')}</b></div>
            <div style='color:#8b95b0;'>F1-macro: <b style='color:#74c0fc;'>{info.get('metrics',{{}}).get('f1_macro','?')}</b></div>
            <div style='margin-top:6px;'>Drift: <b style='color:{drift_color};'>{drift_status}</b></div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.markdown("<div style='color:#ff6b6b; font-size:12px;'>API offline. Execute uvicorn api:app --port 8000</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2d3250; margin:14px 0;'>", unsafe_allow_html=True)

    # Perguntas rápidas
    st.markdown("<div style='color:#8b95b0; font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;'>Perguntas Rápidas</div>", unsafe_allow_html=True)

    QUICK_QUESTIONS = [
        "Quanto de AuC estamos salvando no segmento Wealth?",
        "Quais são os 5 clientes mais críticos da carteira?",
        "Como está o status do modelo em produção?",
        "Há alertas de Data Drift esta semana?",
        "Qual o risco de churn para um cliente Varejo com retorno de 6%?",
        "Liste os clientes prioritários do segmento Alta Renda.",
    ]

    for q in QUICK_QUESTIONS:
        if st.button(q, key=f"quick_{q[:20]}", use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("<hr style='border-color:#2d3250; margin:14px 0;'>", unsafe_allow_html=True)

    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.session_state.tool_logs = []
        st.rerun()


# ── MAIN CONTENT ──────────────────────────────────────────────
st.markdown("""
<div style='padding:8px 0 24px 0;'>
    <h1 style='color:#e8ecf4; font-size:26px; font-weight:700; margin:0;'>
        🤖 Data Agent — Churn Finance
    </h1>
    <p style='color:#8b95b0; margin:6px 0 0 0; font-size:13px;'>
        Pergunte sobre AuC em risco, clientes prioritários, performance do modelo ou alertas de drift.
    </p>
</div>
""", unsafe_allow_html=True)

if not AGENT_OK:
    st.error(f"Erro ao carregar agent.py: {IMPORT_ERROR}")
    st.stop()


# ── Renderiza histórico de mensagens ──────────────────────────
def render_tool_calls(tool_logs: list):
    """Renderiza as chamadas de ferramentas do agente como seção colapsável."""
    if not tool_logs:
        return
    with st.expander(f"🔧 {len(tool_logs)} ferramenta(s) consultada(s)", expanded=False):
        for tc in tool_logs:
            tool_name = tc.get("tool", "?")
            args      = tc.get("args", {})
            result    = tc.get("result", {})

            st.markdown(f"""
            <div class='thinking-box'>
                <span class='tool-badge'>⚙ {tool_name}</span>
                <code style='font-size:11px; color:#8b95b0;'>{args}</code>
            </div>
            """, unsafe_allow_html=True)

            # Exibe resultado resumido
            if isinstance(result, dict) and "error" not in result:
                with st.container():
                    import json as _json
                    st.code(_json.dumps(result, indent=2, ensure_ascii=False)[:600] + "...", language="json")


def render_message(role: str, content: str, tool_logs: list = None):
    """Renderiza uma mensagem do chat."""
    import re

    # Converte markdown simples para HTML
    def md_to_html(text):
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        # Code inline
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        # Blockquote
        text = re.sub(r'^> (.+)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
        # Headers
        text = re.sub(r'^### (.+)$', r'<h4 style="color:#e8ecf4;margin:8px 0 4px 0;">\1</h4>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$',  r'<h3 style="color:#e8ecf4;margin:10px 0 6px 0;">\1</h3>', text, flags=re.MULTILINE)
        # List items
        text = re.sub(r'^- (.+)$',   r'<li style="margin:2px 0;">\1</li>', text, flags=re.MULTILINE)
        # Table (simple)
        lines = text.split('\n')
        result_lines = []
        i = 0
        while i < len(lines):
            if '|' in lines[i] and i + 1 < len(lines) and '---' in lines[i + 1]:
                headers = [h.strip() for h in lines[i].split('|') if h.strip()]
                i += 2  # skip separator
                table_html = '<table><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>'
                while i < len(lines) and '|' in lines[i]:
                    cells = [c.strip() for c in lines[i].split('|') if c.strip()]
                    table_html += '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'
                    i += 1
                table_html += '</table>'
                result_lines.append(table_html)
            else:
                result_lines.append(lines[i])
                i += 1
        text = '\n'.join(result_lines)
        # Line breaks
        text = text.replace('\n', '<br>')
        return text

    if role == "user":
        st.markdown(f"""
        <div class='msg-user'>
            <div class='bubble-user'>{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        html_content = md_to_html(content)
        st.markdown(f"""
        <div class='msg-agent'>
            <div class='agent-avatar'>🤖</div>
            <div class='bubble-agent'>{html_content}</div>
        </div>
        """, unsafe_allow_html=True)
        if tool_logs:
            render_tool_calls(tool_logs)


# Renderiza histórico
for i, msg in enumerate(st.session_state.messages):
    logs = st.session_state.tool_logs[i] if i < len(st.session_state.tool_logs) else []
    render_message(msg["role"], msg["content"], logs if msg["role"] == "assistant" else None)


# ── Mensagem de boas-vindas se chat vazio ─────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center; padding:48px 20px; color:#8b95b0;'>
        <div style='font-size:48px; margin-bottom:16px;'>🤖</div>
        <div style='font-size:18px; font-weight:600; color:#c1c8de; margin-bottom:8px;'>
            Olá, Diretor.
        </div>
        <div style='font-size:14px; color:#6b7590; max-width:480px; margin:0 auto; line-height:1.7;'>
            Sou seu Data Agent. Tenho acesso ao modelo de churn em produção e posso responder 
            perguntas sobre risco de AuC, clientes prioritários e alertas de drift em tempo real.
        </div>
        <div style='margin-top:20px; font-size:12px; color:#4a5270;'>
            Use as perguntas rápidas na sidebar ou digite sua pergunta abaixo.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Processa perguntas rápidas ────────────────────────────────
question_to_process = None

if st.session_state.pending_question:
    question_to_process = st.session_state.pending_question
    st.session_state.pending_question = None


# ── Input do usuário ──────────────────────────────────────────
user_input = st.chat_input("Pergunte sobre a carteira, risco de AuC, clientes ou o modelo...")

if user_input:
    question_to_process = user_input


# ── Executa o agente ──────────────────────────────────────────
if question_to_process:
    # Adiciona mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": question_to_process})
    render_message("user", question_to_process)

    # Constrói histórico para contexto (últimas 6 trocas)
    history = []
    recent = st.session_state.messages[-13:-1]  # exclui a que acabamos de adicionar
    for msg in recent:
        history.append({"role": msg["role"], "content": msg["content"]})

    # Mostra "pensando..."
    with st.spinner("🤖 Consultando dados..."):
        try:
            response_text, tool_logs = run_agent(question_to_process, history)
        except Exception as e:
            response_text = f"Erro ao executar o agente: {str(e)}\n\nVerifique se a API está rodando: `uvicorn api:app --port 8000`"
            tool_logs = []

    # Adiciona resposta ao histórico
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.tool_logs.append(tool_logs)

    # Renderiza resposta
    render_message("assistant", response_text, tool_logs)
    st.rerun()
