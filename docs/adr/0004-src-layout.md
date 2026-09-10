# ADR-0004: Src-layout — todo código Python em `src/`

**Data:** 2026-09-10
**Status:** Accepted
**Proposto por:** Luiz Maibashi
**Estende / revê:** [ADR-0003](0003-modulo-serving-contract.md) §2–§4 (posição de `serving_contract.py`)

---

## 1. CONTEXTO (O Quê?)

A raiz do repositório acumulava **11 arquivos `.py`** soltos (`agent.py`, `agent_chat.py`,
`api.py`, `app.py`, `monitor.py`, `orchestrator.py`, `pipeline.py`, `serving_contract.py`,
`shap_analysis_v2.py`, `transformers.py`, `worker.py`) ao lado de `src/`, que já continha
`kedro_runner.py`, `job_queue.py`, `data_processing/` e `model_training/`.

Efeitos:

- Duas "casas" de código sem critério — um leitor não sabe se `transformers.py` é biblioteca
  ou script, nem por que `nodes.py` mora em `src/` e `api.py` não.
- Imports mistos: uns `from src.model_training.nodes import`, outros `from transformers import`,
  dependendo só de o CWD ser a raiz.
- A auditoria de conformidade de 2026-09-10 marcou isso como Caos Funcional; a decisão
  na hora foi **não mexer** (risco vs. ganho cosmético). Revista a pedido do Luiz:
  o ganho de legibilidade do repositório de portfólio vale o custo único da migração.

O ADR-0003 §3 registrou como "inconsistência de layout deliberada" `serving_contract.py`
ficar na raiz e não em `src/`, com o receio de `src/` puxar `sklearn` transitivamente.
Verificado: **`src/__init__.py` vazio não importa nada**; `import src.serving_contract`
não arrasta `sklearn`. O receio não se sustentava.

## 2. DECISÃO (Por Quê?)

**Src-layout padrão:** todo `.py` do projeto vive em `src/`, importado como pacote `src.*`.

- `src/__init__.py` vazio (pacote regular, previsível).
- Imports internos sempre `from src.<módulo> import …`.
- **Entry points** (`pipeline.py`, `monitor.py`, `shap_analysis_v2.py`, `worker.py`,
  `orchestrator.py`, `app.py`, `agent_chat.py`) trazem um shim de 1 linha antes dos
  imports `src.*`:
  ```python
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  ```
  necessário porque o Streamlit e o `python arquivo.py` injetam o diretório do script
  (`src/`) no path, não a raiz.
- **Testes:** `conftest.py` na raiz insere a raiz no `sys.path`; imports passam a `from src…`.
- **Comandos** (sempre a partir da raiz do projeto):
  - `python src/pipeline.py` · `python src/shap_analysis_v2.py` · `python src/monitor.py`
  - `python -m pytest -q`
  - `uvicorn src.api:app --port 8000`
  - `streamlit run src/app.py` · `streamlit run src/agent_chat.py`
- **Docker:** `CMD ["uvicorn", "api:app", "--app-dir", "src", …]`; `command: python src/worker.py`.
- Os caminhos relativos de saída (`output/…`, `conf/…`, `reports/…`) continuam relativos ao
  CWD = raiz do projeto. Rodar de outro diretório sempre foi e segue sendo não suportado.

**Revê o ADR-0003:** `serving_contract.py` passa a morar em `src/` como os demais. A
alternativa "dentro de `src/`" que o ADR-0003 §4 rejeitou é a adotada aqui; o motivo da
rejeição (`sklearn` transitivo) foi verificado como falso.

## 3. CONSEQUÊNCIAS

**Positivas:**
- Raiz do repo fica só com config, docs e pastas — legível num relance.
- Uma casa só para código; imports uniformes (`src.*`).
- `src/__init__.py` explícito remove a dependência de namespace-package implícito.

**Negativas / custo pago uma vez:**
- ~20 sites de import reescritos + `conftest.py` + shims nos 7 entry points.
- Dockerfile, `docker-compose.yml` e `.devcontainer` ajustados.
- Os `.pkl` versionados mudam (o módulo das classes `FeatureEngineer`/`StructuralNullImputer`
  passa de `transformers` para `src.transformers`) — regenerados no mesmo commit.
- Shim de `sys.path` nos entry points é boilerplate; é o preço de suportar
  Streamlit + `python arquivo.py` + `uvicorn -m` + pytest no mesmo projeto.

**Não-objetivo:** dividir `src/` em subpacotes por responsabilidade (`serving/`, `agentic/`).
Fica plano; o projeto é pequeno e o `serving_contract` já é a fronteira que importa.

## 4. ALTERNATIVAS DESCARTADAS

| Opção | Por quê não |
|---|---|
| Deixar os 11 na raiz | O que a auditoria marcou como Caos Funcional; decisão revista. |
| Manter `src.` só nos submódulos e o resto sem prefixo (CWD-dependente) | É o estado que gerava imports mistos e confusão sobre o que é lib vs. script. |
| Subpacotes `serving/` + `agentic/` | Mais movimento e mais imports para ganho de organização marginal num projeto deste tamanho. |
| `pyproject.toml` + `pip install -e .` (package instalável) | Peso de empacotamento que um projeto de portfólio de 1 modelo não paga. O shim resolve. |

## 5. LINKS

- [[0003-modulo-serving-contract]] — §2–§4 revistos aqui
- Auditoria de conformidade 2026-09-10 (`brain/sessions/SESSAO_10_09_2026_pipeline_churn_auditoria.md`)
