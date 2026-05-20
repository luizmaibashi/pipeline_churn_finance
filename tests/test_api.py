import pytest
from fastapi.testclient import TestClient
from api import app

@pytest.fixture
def client():
    """Fixture que fornece o TestClient executando os eventos de lifespan da API."""
    with TestClient(app) as c:
        yield c

def test_api_health_check(client):
    """Testa o endpoint de health check da API."""
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] in ["ok", "degraded"]
    assert "api_version" in json_data


def test_api_model_info(client):
    """Testa o endpoint de metadados do modelo."""
    response = client.get("/model/info")
    assert response.status_code == 200
    # O endpoint retorna metadados completos ou uma mensagem se for versão flat
    assert "message" in response.json() or "version" in response.json()


def test_predict_single_client_success(client):
    """Testa predição bem-sucedida para um único cliente com dados válidos."""
    payload = {
        "cliente_id": "CLI99999",
        "segmento": "Wealth",
        "meses_cliente": 24,
        "qtd_produtos": 3,
        "retorno_12m_pct": 12.5,
        "freq_contato_mes": 2,
        "saldo_bi": 0.25
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["cliente_id"] == "CLI99999"
    assert "churn_probability" in json_data
    assert "risk_level" in json_data
    assert "recommended_action" in json_data
    assert "flow" in json_data


@pytest.mark.parametrize("invalid_field,value", [
    ("segmento", "Investimento"),      # Segmento inválido
    ("meses_cliente", 0),             # Meses fora do range [1, 600]
    ("meses_cliente", 601),           # Meses fora do range [1, 600]
    ("qtd_produtos", 0),              # Qtd produtos fora do range [1, 20]
    ("qtd_produtos", 21),             # Qtd produtos fora do range [1, 20]
    ("retorno_12m_pct", -0.1),         # Retorno fora do range [0.0, 100.0]
    ("retorno_12m_pct", 100.1),        # Retorno fora do range [0.0, 100.0]
    ("freq_contato_mes", -1),         # Contatos fora do range [0, 60]
    ("freq_contato_mes", 61),          # Contatos fora do range [0, 60]
    ("saldo_bi", 0.0),                # Saldo deve ser gt=0.0
    ("saldo_bi", -0.05)               # Saldo deve ser gt=0.0
])
def test_predict_single_client_boundary_violations(client, invalid_field, value):
    """Testa se violações de limites dos esquemas Pydantic retornam HTTP 422."""
    valid_payload = {
        "cliente_id": "CLI99999",
        "segmento": "Wealth",
        "meses_cliente": 24,
        "qtd_produtos": 3,
        "retorno_12m_pct": 12.5,
        "freq_contato_mes": 2,
        "saldo_bi": 0.25
    }
    
    # Substitui campo válido pelo valor inválido paramétrico
    valid_payload[invalid_field] = value
    
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_batch_async_success(client):
    """Testa predição assíncrona em lote e consulta de status."""
    import time
    payload = {
        "clientes": [
            {
                "cliente_id": "CLI00001",
                "segmento": "Varejo",
                "meses_cliente": 12,
                "qtd_produtos": 2,
                "retorno_12m_pct": 8.5,
                "freq_contato_mes": 1,
                "saldo_bi": 0.05
            },
            {
                "cliente_id": "CLI00002",
                "segmento": "Wealth",
                "meses_cliente": 48,
                "qtd_produtos": 4,
                "retorno_12m_pct": 14.2,
                "freq_contato_mes": 5,
                "saldo_bi": 1.2
            }
        ]
    }
    
    # Envia lote
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 202
    json_data = response.json()
    assert "job_id" in json_data
    assert json_data["status"] == "PENDING"
    
    job_id = json_data["job_id"]
    
    # Aguarda o processamento em background
    max_retries = 10
    completed = False
    for _ in range(max_retries):
        status_resp = client.get(f"/predict/batch/status/{job_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        if status_data["status"] == "COMPLETED":
            completed = True
            assert "result" in status_data
            assert status_data["result"]["total"] == 2
            assert len(status_data["result"]["results"]) == 2
            assert "summary" in status_data["result"]
            break
        elif status_data["status"] == "FAILED":
            pytest.fail(f"Job falhou: {status_data['error']}")
        time.sleep(0.5)
        
    assert completed, "O processamento em background do lote expirou."
