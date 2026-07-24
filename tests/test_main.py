from fastapi.testclient import TestClient

from oasis_generation import bedrock
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
    # GEN-002 pass 2026-07-12 cleared both community fine-tunes for
    # personal/fleet use; baseline is disabled until re-pulled or cloud-hosted.
    assert "gemma-4-12b-coder" in ids
    assert "gemma-4-12b-agentic" in ids
    assert "gemma-4-12b-it" not in ids
    # Self-hosted higher tiers stay hidden until their runners exist...
    assert "glm-5.2" not in ids
    # ...but the Bedrock passthrough (GEN-003) is live now (roster trimmed 2026-07-24).
    assert "claude-opus-4-8" in ids
    assert "claude-sonnet-5" in ids
    assert "glm-5" in ids
    # Roster trim: these were removed 2026-07-24.
    for removed in ("claude-sonnet-4-6", "claude-haiku-4-5", "gpt-oss-120b",
                    "deepseek-v3.2", "llama-4-maverick"):
        assert removed not in ids
    # Disabled templates/placeholders stay hidden.
    assert "gpt-5" not in ids          # openai_compat template, no key
    assert "gpt-5.6-sol" not in ids    # bedrock_mantle backend not built yet


def test_models_expose_capability_descriptors():
    data = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    # Adaptive Claude: thinking + effort controllable, temperature hidden.
    opus = data["claude-opus-4-8"]["capabilities"]
    assert opus["supports_thinking"] is True
    assert opus["supports_effort"] is True
    assert opus["effort_levels"] == ["low", "high", "xhigh"]  # fast/balanced/deep order
    assert opus["supports_temperature"] is False
    assert opus["max_output_tokens_ceiling"] == 32000
    assert opus["profiles"] == ["fast", "balanced", "deep"]
    assert opus["default_profile"] == "balanced"
    # Native model (GLM-5): no gateway thinking control, temperature allowed.
    glm = data["glm-5"]["capabilities"]
    assert glm["supports_thinking"] is False
    assert glm["supports_effort"] is False
    assert glm["supports_temperature"] is True
    assert glm["effort_levels"] == []
    # Local Gemma: native, small ceiling.
    gemma = data["gemma-4-12b-coder"]["capabilities"]
    assert gemma["supports_thinking"] is False
    assert gemma["max_output_tokens_ceiling"] == 8192


def test_unknown_model_404s():
    resp = client.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert resp.status_code == 404


def test_auth_enforced_when_tokens_configured(monkeypatch):
    monkeypatch.setenv("OASIS_GENERATION_SERVICE_TOKENS", "sekrit")
    get_settings.cache_clear()
    assert client.get("/v1/models").status_code == 401
    ok = client.get("/v1/models", headers={"Authorization": "Bearer sekrit"})
    assert ok.status_code == 200


class _FakeBedrockClient:
    def converse(self, **kwargs):
        self.seen = kwargs
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "pong"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 4, "outputTokens": 1, "totalTokens": 5},
        }


def test_bedrock_route_returns_openai_shape(monkeypatch):
    fake = _FakeBedrockClient()
    monkeypatch.setattr(bedrock, "_client", lambda region: fake)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "claude-sonnet-5"
    assert data["choices"][0]["message"]["content"] == "pong"
    # The public id was translated to the Bedrock inference-profile id upstream.
    assert fake.seen["modelId"] == "us.anthropic.claude-sonnet-5"
    # The model's inference policy was applied: adaptive thinking + balanced
    # effort injected, and no temperature leaked into inferenceConfig.
    amrf = fake.seen.get("additionalModelRequestFields") or {}
    assert amrf.get("thinking") == {"type": "adaptive"}
    assert amrf.get("output_config") == {"effort": "high"}  # balanced default
    assert "temperature" not in fake.seen["inferenceConfig"]
    assert fake.seen["inferenceConfig"]["maxTokens"] == 16000  # balanced profile


def test_openai_compat_missing_key_502(monkeypatch):
    # gpt-5 is an openai_compat template; enable it in a throwaway copy so the
    # route is exercised without a real OpenAI key present.
    from oasis_generation import catalog

    entry = next(m for m in catalog.CATALOG if m.public_id == "gpt-5")
    monkeypatch.setattr(entry, "enabled", True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 502
    assert "OPENAI_API_KEY" in resp.json()["detail"]
