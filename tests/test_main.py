from fastapi.testclient import TestClient

from oasis_generation.config import get_settings
from oasis_generation.main import app

client = TestClient(app)


def setup_function() -> None:
    get_settings.cache_clear()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_models_lists_only_enabled():
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "gemma-4-12b-it" in ids
    # Community fine-tunes stay hidden until GEN-002 clears them.
    assert "gemma-4-12b-coder" not in ids
    assert "gemma-4-12b-agentic" not in ids


def test_unknown_model_404s():
    resp = client.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert resp.status_code == 404


def test_auth_enforced_when_tokens_configured(monkeypatch):
    monkeypatch.setenv("OASIS_GENERATION_SERVICE_TOKENS", "sekrit")
    get_settings.cache_clear()
    assert client.get("/v1/models").status_code == 401
    ok = client.get("/v1/models", headers={"Authorization": "Bearer sekrit"})
    assert ok.status_code == 200
