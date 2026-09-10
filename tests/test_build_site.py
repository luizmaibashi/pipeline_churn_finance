from pathlib import Path

from tools.build_site import MAX_MODEL_GZIP_BYTES, gzip_size_bytes


def test_model_publicado_cabe_no_orcamento_de_transferencia():
    path = Path("docs/model.json")
    assert gzip_size_bytes(path) <= MAX_MODEL_GZIP_BYTES


def test_build_publica_snapshot_de_explicabilidade():
    assert Path("docs/data/client_explanations.json").exists()
