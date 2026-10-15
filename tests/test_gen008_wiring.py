"""Integration: routed /v1/chat/completions, /v1/x/decide, /v1/gate (GEN-008)."""
import yaml
from fastapi.testclient import TestClient

from oasis_generation import main, wiring
from oasis_generation.config import get_settings
from oasis_generation.main import app

client = TestClient(app)
ROUTES = {
    "pools": {"eu": {"base_url": "http://eu/v1", "region": "eu"},
              "us": {"base_url": "http://us/v1", "region": "us"}},
    "tenants": {"strict": {"residency": "eu", "fallback": "never"}},
    "routes": [{"tenant": "*", "model": "gemma-4-12b-agentic", "targets": ["own:us", "own:eu"]}],
}


def _setup(monkeypatch, tmp_path, routes=ROUTES, tokens="strict:tk,loose:tl"):
    p = tmp_path / "routes.yaml"
    p.write_text(yaml.safe_dump(routes))
    monkeypatch.setenv("OASIS_GENERATION_ROUTES_FILE", str(p))
    monkeypatch.setenv("OASIS_GENERATION_SERVICE_TOKENS", tokens)
    monkeypatch.setenv("OASIS_GENERATION_USAGE_LOG", str(tmp_path / "u.jsonl"))
    get_settings.cache_clear(); wiring.reset()


def _teardown():
    get_settings.cache_clear(); wiring.reset()


def _fake_proxy(record, fail=()):
    async def proxy(body, entry, base_url, settings, headers, request=None):
        record.append(base_url)
        if base_url in fail:
            raise main.HTTPException(502, "down")
        return main.JSONResponse({"ok": base_url})
    return proxy


def test_routed_failover_in_order(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    seen = []
    monkeypatch.setattr(main, "_proxy_openai_compatible", _fake_proxy(seen, fail={"http://us/v1"}))
    r = client.post("/v1/chat/completions", headers={"Authorization": "Bearer tl"},
                    json={"model": "gemma-4-12b-agentic", "messages": []})
    _teardown()
    assert r.status_code == 200 and seen == ["http://us/v1", "http://eu/v1"]


def test_eu_tenant_never_reaches_us(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    seen = []
    monkeypatch.setattr(main, "_proxy_openai_compatible", _fake_proxy(seen))
    r = client.post("/v1/chat/completions", headers={"Authorization": "Bearer tk"},
                    json={"model": "gemma-4-12b-agentic", "messages": []})
    _teardown()
    assert r.status_code == 200 and seen == ["http://eu/v1"]


def test_eu_tenant_with_eu_down_fails_not_leaks(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    seen = []
    monkeypatch.setattr(main, "_proxy_openai_compatible", _fake_proxy(seen, fail={"http://eu/v1"}))
    r = client.post("/v1/chat/completions", headers={"Authorization": "Bearer tk"},
                    json={"model": "gemma-4-12b-agentic", "messages": []})
    _teardown()
    assert r.status_code == 502 and "http://us/v1" not in seen


def test_no_route_is_422(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    r = client.post("/v1/chat/completions", headers={"Authorization": "Bearer tl"},
                    json={"model": "unrouted", "messages": []})
    _teardown()
    assert r.status_code == 422


def test_legacy_path_when_routes_unset(monkeypatch):
    monkeypatch.delenv("OASIS_GENERATION_ROUTES_FILE", raising=False)
    get_settings.cache_clear(); wiring.reset()
    r = client.post("/v1/chat/completions", json={"model": "no-such-model", "messages": []})
    _teardown()
    assert r.status_code == 404


def test_gate_and_decide_endpoints(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    svc = wiring.xservice(get_settings())

    async def caller(url, body, t):
        return {"decVerdictProbs": {"allow": 0.98, "escalate": 0.01, "deny": 0.01}}
    svc._call = caller
    h = {"Authorization": "Bearer tl"}
    g = client.post("/v1/gate", headers=h, json={"model": "m", "action": {"tool": "x"}, "baseline": "allow"})
    assert g.status_code == 200 and g.json()["verdict"] == "allow"
    g = client.post("/v1/gate", headers=h, json={"model": "m", "action": {}})
    assert g.json()["verdict"] == "escalate"          # no baseline -> cannot pass a bare allow
    assert client.post("/v1/gate", headers=h, json={"model": "m", "action": {}, "baseline": "zz"}).status_code == 422
    d = client.post("/v1/x/decide", headers=h, json={"model": "m", "labelset_id": "gate.v1", "ctx": {}})
    assert d.status_code == 200 and d.json()["chosen"] == "allow"
    assert client.post("/v1/x/decide", headers=h, json={"model": "m", "labelset_id": "nope"}).status_code == 404
    assert client.post("/v1/gate", json={"model": "m", "action": {}}).status_code == 401   # auth enforced
    _teardown()


def test_decider_down_gate_escalates_via_http(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    svc = wiring.xservice(get_settings())

    async def caller(url, body, t):
        raise ConnectionError("down")
    svc._call = caller
    g = client.post("/v1/gate", headers={"Authorization": "Bearer tl"},
                    json={"model": "m", "action": {}, "baseline": "allow"})
    _teardown()
    assert g.status_code == 200 and g.json()["verdict"] == "escalate"


def test_example_routes_file_is_valid():
    from oasis_generation.routes import RouteTable
    RouteTable.from_dict(yaml.safe_load(open("docs/routes.example.yaml")))
