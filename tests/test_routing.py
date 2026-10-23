"""GEN-008 per-client route switch (remote.py): local vs the remote gateway."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from oasis_generation import main
from oasis_generation.config import get_settings
from oasis_generation.main import app

client = TestClient(app)

BOT = {"Authorization": "Bearer tok-house"}
OTHER = {"Authorization": "Bearer tok-yesman"}
ADMIN = {"Authorization": "Bearer admin-secret"}


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "admin-token").write_text("admin-secret\n")
    (secrets / "remote-token").write_text("fleet-secret\n")
    monkeypatch.setenv("OASIS_GENERATION_SERVICE_TOKENS", "house:tok-house,yesman:tok-yesman")
    monkeypatch.setenv("OASIS_GENERATION_ADMIN_TOKEN_FILE", str(secrets / "admin-token"))
    monkeypatch.setenv("OASIS_GENERATION_REMOTE_TOKEN_FILE", str(secrets / "remote-token"))
    monkeypatch.setenv("OASIS_GENERATION_REMOTE_BASE_URL", "https://gen.remote.test/v1")
    monkeypatch.setenv("OASIS_GENERATION_ROUTE_STATE_FILE", str(tmp_path / "state" / "routing.json"))
    monkeypatch.setenv("OASIS_GENERATION_SELF_SWITCH_CLIENTS", "house")
    # USAGE_LOG is read once at import, so patch the module constant.
    monkeypatch.setattr(main, "USAGE_LOG", str(tmp_path / "usage" / "requests.jsonl"))
    get_settings.cache_clear()
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "remote-only"}]})
        body = json.loads(req.content)
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "from remote"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        })

    monkeypatch.setattr(main, "_remote_transport", httpx.MockTransport(handler))
    yield {"seen": seen, "tmp": tmp_path, "secrets": secrets}
    get_settings.cache_clear()


def _chat(headers, extra=None):
    return client.post("/v1/chat/completions", headers={**headers, **(extra or {})},
                       json={"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "hi"}]})


def test_default_is_local_and_says_so(env):
    resp = client.get("/v1/models", headers=BOT)
    assert resp.status_code == 200
    assert resp.headers["X-Oasis-Served-By"] == "local"
    assert "claude-sonnet-5" in [m["id"] for m in resp.json()["data"]]
    assert env["seen"] == []


def test_admin_switch_forwards_only_that_client(env):
    resp = client.put("/admin/routing/house", headers=ADMIN, json={"target": "remote"})
    assert resp.status_code == 200 and resp.json() == {"client": "house", "from": "local", "to": "remote"}

    resp = _chat(BOT)
    assert resp.status_code == 200
    assert resp.headers["X-Oasis-Served-By"] == "remote"
    assert resp.json()["choices"][0]["message"]["content"] == "from remote"
    req = env["seen"][-1]
    assert str(req.url) == "https://gen.remote.test/v1/chat/completions"
    # The remote token goes upstream; the bot's own token never does.
    assert req.headers["authorization"] == "Bearer fleet-secret"
    assert req.headers["x-oasis-origin-client"] == "house"

    # /v1/models follows the switch, so /genconfig shows the remote catalog.
    assert client.get("/v1/models", headers=BOT).json()["data"] == [{"id": "remote-only"}]

    # Another bot is untouched (negative control).
    n = len(env["seen"])
    assert client.get("/v1/models", headers=OTHER).headers["X-Oasis-Served-By"] == "local"
    assert len(env["seen"]) == n

    rows = [json.loads(l) for l in (env["tmp"] / "usage" / "requests.jsonl").read_text().splitlines()]
    assert rows[-1]["backend"] == "remote" and rows[-1]["client"] == "house"
    assert rows[-1]["prompt_tokens"] == 3

    audit = (env["tmp"] / "state" / "routing-audit.jsonl").read_text().splitlines()
    assert json.loads(audit[-1])["actor"] == "admin"


def test_switch_back_to_local(env):
    client.put("/admin/routing/house", headers=ADMIN, json={"target": "remote"})
    client.put("/admin/routing/house", headers=ADMIN, json={"target": "local"})
    assert client.get("/v1/models", headers=BOT).headers["X-Oasis-Served-By"] == "local"
    state = json.loads((env["tmp"] / "state" / "routing.json").read_text())
    assert state["clients"] == {}


def test_admin_requires_admin_token(env):
    assert client.get("/admin/routing", headers=BOT).status_code == 401  # a bot token is not admin
    assert client.get("/admin/routing").status_code == 401
    assert client.put("/admin/routing/house", headers=BOT, json={"target": "remote"}).status_code == 401
    view = client.get("/admin/routing", headers=ADMIN).json()
    assert view["clients"] == {"house": "local", "yesman": "local"}
    assert view["remote_configured"] is True


def test_admin_disabled_without_token_file(env):
    (env["secrets"] / "admin-token").unlink()
    assert client.get("/admin/routing", headers=ADMIN).status_code == 503


def test_admin_rejects_unknown_client_and_target(env):
    assert client.put("/admin/routing/nobody", headers=ADMIN, json={"target": "remote"}).status_code == 404
    assert client.put("/admin/routing/house", headers=ADMIN, json={"target": "elsewhere"}).status_code == 400


def test_remote_refused_when_not_configured(env, monkeypatch):
    monkeypatch.setenv("OASIS_GENERATION_REMOTE_BASE_URL", "")
    get_settings.cache_clear()
    assert client.put("/admin/routing/house", headers=ADMIN, json={"target": "remote"}).status_code == 409
    monkeypatch.setenv("OASIS_GENERATION_REMOTE_BASE_URL", "http://plain.test/v1")  # not https
    get_settings.cache_clear()
    assert client.put("/admin/routing/house", headers=ADMIN, json={"target": "remote"}).status_code == 409


def test_self_switch_only_self_and_only_if_allowed(env):
    resp = client.put("/v1/routing/self", headers=BOT, json={"target": "remote"})
    assert resp.status_code == 200 and resp.json()["to"] == "remote"
    assert client.get("/v1/routing/self", headers=BOT).json()["target"] == "remote"
    # yesman is not in SELF_SWITCH_CLIENTS.
    assert client.put("/v1/routing/self", headers=OTHER, json={"target": "remote"}).status_code == 403
    assert client.get("/v1/routing/self", headers=OTHER).json()["may_switch"] is False
    audit = (env["tmp"] / "state" / "routing-audit.jsonl").read_text().splitlines()
    assert json.loads(audit[-1])["actor"] == "self:house"


def test_self_switch_star_means_every_token_client(env, monkeypatch):
    monkeypatch.setenv("OASIS_GENERATION_SELF_SWITCH_CLIENTS", "*")
    get_settings.cache_clear()
    assert client.put("/v1/routing/self", headers=OTHER, json={"target": "remote"}).status_code == 200


def test_no_silent_fallback_when_remote_down(env, monkeypatch):
    def down(req):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(main, "_remote_transport", httpx.MockTransport(down))
    client.put("/admin/routing/house", headers=ADMIN, json={"target": "remote"})
    resp = _chat(BOT)
    assert resp.status_code == 502
    assert "remote gateway unreachable" in resp.json()["detail"]


def test_remote_token_removed_after_switch_gives_502(env):
    client.put("/admin/routing/house", headers=ADMIN, json={"target": "remote"})
    (env["secrets"] / "remote-token").unlink()
    resp = _chat(BOT)
    assert resp.status_code == 502 and "Switch it back to local" in resp.json()["detail"]


def test_forwarded_call_is_never_forwarded_again(env):
    client.put("/admin/routing/house", headers=ADMIN, json={"target": "remote"})
    resp = client.get("/v1/models", headers={**BOT, "X-Oasis-Origin-Client": "kolmogorov"})
    assert resp.headers["X-Oasis-Served-By"] == "local"


def test_corrupt_state_file_means_local(env):
    path = env["tmp"] / "state" / "routing.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert client.get("/v1/models", headers=BOT).headers["X-Oasis-Served-By"] == "local"


def test_require_auth_without_tokens_refuses(monkeypatch):
    """GEN-008: an internet-facing gateway with no tokens must not serve anyone."""
    monkeypatch.setenv("OASIS_GENERATION_SERVICE_TOKENS", "")
    monkeypatch.setenv("OASIS_GENERATION_REQUIRE_AUTH", "1")
    get_settings.cache_clear()
    assert client.get("/v1/models").status_code == 503
    with pytest.raises(RuntimeError, match="refusing to start"):
        main._refuse_open_deployment()
    # Negative control: without REQUIRE_AUTH the local-dev behaviour is unchanged.
    monkeypatch.delenv("OASIS_GENERATION_REQUIRE_AUTH")
    get_settings.cache_clear()
    assert client.get("/v1/models").status_code == 200
    main._refuse_open_deployment()
