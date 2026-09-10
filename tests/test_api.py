import pytest
from fastapi.testclient import TestClient
from api import app

# Payload base v2 (ADR-0001, Direção A — early-warning comportamental).
# `retorno_12m_pct`, `dias_desde_ultimo_contato` e `tempo_resposta_medio_horas`
# são opcionais (nulo estrutural → flag derivada + imputação por mediana do treino).
PAYLOAD_V2 = {
    "cliente_id": "CLI99999",
    "segmento": "Wealth",
    "meses_cliente": 24,
    "qtd_produtos": 3,
    "retorno_12m_pct": 12.5,
    "freq_contato_mes": 2,
    "saldo_bi": 0.25,
    "dias_desde_ultimo_contato": 18.0,
    "variacao_freq_contato_3m": -0.05,
    "tempo_resposta_medio_horas": 20.0,
}


@pytest.fixture
def client():
    """Fixture que fornece o TestClient executando os eventos de lifespan da API."""
    with TestClient(app) as c:
        yield c


def test_api_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] in ["ok", "degraded"]
    assert "api_version" in json_data


def test_api_model_info_is_v2(client):
    """O endpoint de metadados deve reportar a v2 (early-warning), não a baseline v1."""
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json().get("version") == "v2"


def test_predict_single_client_success(client):
    response = client.post("/predict", json=PAYLOAD_V2)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["cliente_id"] == "CLI99999"
    assert 0.0 <= json_data["churn_probability"] <= 1.0
    assert json_data["risk_level"] in ("BAIXO", "MEDIO", "ALTO")
    assert "recommended_action" in json_data
    assert "flow" in json_data


def test_predict_accepts_null_early_warning_fields(client):
    """Nulo estrutural em retorno/dias/tempo é contrato válido, não erro 422."""
    payload = {**PAYLOAD_V2, "retorno_12m_pct": None,
               "dias_desde_ultimo_contato": None, "tempo_resposta_medio_horas": None}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert 0.0 <= response.json()["churn_probability"] <= 1.0


def test_predict_accepts_negative_return(client):
    """v2 permite retorno negativo — a baseline v1 travava em ge=0."""
    response = client.post("/predict", json={**PAYLOAD_V2, "retorno_12m_pct": -3.5})
    assert response.status_code == 200


@pytest.mark.parametrize("invalid_field,value", [
    ("segmento", "Investimento"),
    ("meses_cliente", 0),
    ("meses_cliente", 601),
    ("qtd_produtos", 0),
    ("qtd_produtos", 21),
    ("retorno_12m_pct", -60.0),         # abaixo de ge=-50
    ("retorno_12m_pct", 100.1),
    ("freq_contato_mes", -1),
    ("freq_contato_mes", 61),
    ("saldo_bi", 0.0),
    ("saldo_bi", -0.05),
    ("variacao_freq_contato_3m", -2.0),  # abaixo de ge=-1
    ("variacao_freq_contato_3m", 5.0),   # acima de le=3
    ("dias_desde_ultimo_contato", 500.0),
])
def test_predict_single_client_boundary_violations(client, invalid_field, value):
    response = client.post("/predict", json={**PAYLOAD_V2, invalid_field: value})
    assert response.status_code == 422


def test_predict_missing_required_early_warning_field(client):
    """`variacao_freq_contato_3m` não tem default — ausência é 422."""
    payload = {k: v for k, v in PAYLOAD_V2.items() if k != "variacao_freq_contato_3m"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_advisor_exposed_portfolio(client):
    """Direção B: tabela descritiva de AuC exposto por assessor (não é predição)."""
    response = client.get("/advisors/exposed-portfolio?limit=5")
    assert response.status_code == 200
    json_data = response.json()
    assert "assessores" in json_data
    assert len(json_data["assessores"]) <= 5
    if json_data["assessores"]:
        a = json_data["assessores"][0]
        assert {"assessor_id", "auc_exposto_total_bi", "pct_carteira_exposta"} <= a.keys()
        # ordenado por AuC exposto desc
        vals = [x["auc_exposto_total_bi"] for x in json_data["assessores"]]
        assert vals == sorted(vals, reverse=True)


def test_predict_batch_async_success(client):
    import time
    payload = {"clientes": [
        {**PAYLOAD_V2, "cliente_id": "CLI00001", "segmento": "Varejo", "saldo_bi": 0.05},
        {**PAYLOAD_V2, "cliente_id": "CLI00002", "saldo_bi": 1.2},
    ]}

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 202
    json_data = response.json()
    assert "job_id" in json_data
    job_id = json_data["job_id"]

    for _ in range(10):
        status_resp = client.get(f"/predict/batch/status/{job_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        if status_data["status"] == "COMPLETED":
            assert status_data["result"]["total"] == 2
            assert len(status_data["result"]["results"]) == 2
            assert "summary" in status_data["result"]
            break
        elif status_data["status"] == "FAILED":
            pytest.fail(f"Job falhou: {status_data['error']}")
        time.sleep(0.5)
    else:
        pytest.fail("O processamento em background do lote expirou.")
