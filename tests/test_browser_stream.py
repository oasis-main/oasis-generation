"""ORG-171: POST /v1/chat/stream, GET /v1/models in capability mode, auth modes.

Test keys are generated here. No network: the JWKS fetcher is a fake, the usage
publisher is a fake, Bedrock is a fake boto client, mantle.complete is patched,
and the OpenAI-compatible upstream is an httpx.MockTransport.
"""
import asyncio
import json
import threading
import time
import uuid

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from oasis_generation import bedrock, browser_stream as bs, catalog, main, mantle
from oasis_generation.capability import b64u
from oasis_generation.config import get_settings

ISS = "https://auth.test.cloud.oasis-x.io"
KID = "test-kid-1"
KEY = Ed25519PrivateKey.generate()
OTHER_KEY = Ed25519PrivateKey.generate()
JWKS = {"keys": [{"kty": "OKP", "crv": "Ed25519", "kid": KID, "alg": "EdDSA",
                  "x": b64u(KEY.public_key().public_bytes(serialization.Encoding.Raw,
                                                          serialization.PublicFormat.Raw))}]}
HOST = "ai.test.cloud.oasis-x.io"

client = TestClient(main.app, base_url=f"https://{HOST}")


def token(key=KEY, **over) -> str:
    now = int(time.time())
    p = {"iss": ISS, "aud": "oasis-gen", "sub": "user_1", "tier": "personal",
         "billing_state": "active", "scopes": ["gen:chat", "gen:models"],
         "allowance_ref": "allow_user_1_2026-10", "jti": uuid.uuid4().hex,
         "iat": now, "exp": now + 600}
    p.update(over)
    h = b64u(json.dumps({"alg": "EdDSA", "typ": "oasis-cap+jwt", "kid": KID}).encode())
    b = b64u(json.dumps(p).encode())
    sig = key.sign(f"{h}.{b}".encode())
    return f"{h}.{b}.{b64u(sig)}"


def auth(tok=None):
    return {"Authorization": f"Bearer {tok or token()}"}


class FakePublisher:
    def __init__(self):
        self.ok = True
        self.events = []

    def publish(self, topic, payload, qos):
        if not self.ok:
            return False
        assert topic == "oasis/system/gen/usage/output" and qos == 1
        self.events.append(json.loads(payload))
        return True


@pytest.fixture
def cap(monkeypatch, tmp_path):
    monkeypatch.setenv("OASIS_GENERATION_AUTH_MODE", "capability")
    monkeypatch.setenv("CAPABILITY_ISSUER", ISS)
    monkeypatch.setenv("CAPABILITY_JWKS_URL", "https://auth.test.cloud.oasis-x.io/.well-known/jwks.json")
    monkeypatch.setenv("USAGE_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.delenv("OASIS_GENERATION_SERVICE_TOKENS", raising=False)
    get_settings.cache_clear()
    pub = FakePublisher()

    async def fetch(url):
        return JWKS

    rt = bs.CapabilityRuntime(get_settings(), fetch=fetch, publisher=pub)
    bs.set_runtime(rt)
    yield rt, pub
    bs.set_runtime(None)
    get_settings.cache_clear()


def frames(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        if all(ln.startswith(":") for ln in lines):
            continue
        ev = next(ln[7:] for ln in lines if ln.startswith("event: "))
        data = json.loads(next(ln[6:] for ln in lines if ln.startswith("data: ")))
        out.append((ev, data))
    return out


def body(model="claude-sonnet-5", **over):
    b = {"model": model, "messages": [{"role": "user", "content": "hi"}],
         "client_request_id": str(uuid.uuid4())}
    b.update(over)
    return b


class FakeEventStream:
    def __init__(self, events, gate=None):
        self.events, self.gate, self.closed = events, gate, False

    def __iter__(self):
        for i, e in enumerate(self.events):
            if self.gate is not None and i == 1:
                self.gate.wait(5)
            yield e

    def close(self):
        self.closed = True


class FakeBedrock:
    def __init__(self, events=None, exc=None, gate=None):
        self.stream = FakeEventStream(events or [], gate)
        self.exc = exc
        self.seen = None

    def converse_stream(self, **kw):
        self.seen = kw
        if self.exc:
            raise self.exc
        return {"stream": self.stream}


BEDROCK_EVENTS = [
    {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "think"}}, "contentBlockIndex": 0}},
    {"contentBlockDelta": {"delta": {"text": "Hel"}, "contentBlockIndex": 1}},
    {"contentBlockDelta": {"delta": {"text": "lo"}, "contentBlockIndex": 1}},
    {"messageStop": {"stopReason": "end_turn"}},
    {"metadata": {"usage": {"inputTokens": 7, "outputTokens": 3, "totalTokens": 30,
                            "cacheReadInputTokens": 20, "cacheWriteInputTokens": 0}}},
]


# ---- happy paths per backend ------------------------------------------------------

def test_bedrock_stream_events_seq_and_usage_event(cap, monkeypatch):
    rt, pub = cap
    fake = FakeBedrock(BEDROCK_EVENTS)
    monkeypatch.setattr(bedrock, "_client", lambda region: fake)
    b = body()
    r = client.post("/v1/chat/stream", json=b, headers=auth())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert not any(h.lower().startswith("access-control-") for h in r.headers)
    fr = frames(r.text)
    kinds = [e for e, _ in fr]
    assert kinds == ["start", "thinking", "text", "text", "usage", "done"]
    assert [d["seq"] for _, d in fr] == list(range(len(fr)))
    rid = fr[0][1]["request_id"]
    assert rid.startswith("req_") and all(d["request_id"] == rid for _, d in fr)
    assert r.headers["x-request-id"] == rid
    assert fr[0][1]["model"] == "claude-sonnet-5" and fr[0][1]["served_by"] == "bedrock"
    assert fr[1][1] == {"seq": 1, "request_id": rid, "delta": "think", "redacted": False}
    assert "".join(d["delta"] for e, d in fr if e == "text") == "Hello"
    assert fr[-2][1]["tokens_in"] == 27 and fr[-2][1]["tokens_out"] == 3
    assert "cost_estimate" in fr[-2][1]
    assert fr[-1][1]["finish_reason"] == "stop"
    # The upstream got the public model mapped to the Bedrock id.
    assert fake.seen["modelId"] == "us.anthropic.claude-sonnet-5"
    # One usage event, interface doc 5.3.
    (ev,) = pub.events
    assert ev["sub"] == "user_1" and ev["allowance_ref"] == "allow_user_1_2026-10"
    assert ev["request_id"] == rid and ev["client_request_id"] == b["client_request_id"]
    assert (ev["backend"], ev["served_by"], ev["status"], ev["streamed"]) == ("bedrock", "bedrock", "ok", True)
    assert (ev["tokens_in"], ev["tokens_out"], ev["cache_read_tokens"], ev["cache_write_tokens"]) == (7, 3, 20, 0)
    assert "usage_estimated" not in ev
    # The in-flight id is released when the stream ends.
    assert rt.claim("user_1", b["client_request_id"])


def test_bedrock_tool_call_ends_with_tool_calls(cap, monkeypatch):
    events = [
        {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "tu_1", "name": "get_time"}},
                               "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"tz":'}}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '"UTC"}'}}, "contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 9, "totalTokens": 14}}},
    ]
    monkeypatch.setattr(bedrock, "_client", lambda region: FakeBedrock(events))
    tools = [{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}]
    r = client.post("/v1/chat/stream", json=body(tools=tools), headers=auth())
    fr = frames(r.text)
    calls = [d for e, d in fr if e == "tool_call"]
    assert all(c["id"] == "tu_1" and c["name"] == "get_time" for c in calls)
    assert [c["partial"] for c in calls] == [True, True, True, False]
    assert "".join(c["arguments"] for c in calls if c["partial"]) == '{"tz":"UTC"}'
    assert calls[-1]["arguments"] == '{"tz":"UTC"}'
    assert fr[-1] == ("done", {"seq": fr[-1][1]["seq"], "request_id": fr[0][1]["request_id"],
                               "finish_reason": "tool_calls"})
    assert not any(e == "tool_result" for e, _ in fr)


def test_flat_tool_form_is_accepted(cap, monkeypatch):
    fake = FakeBedrock(BEDROCK_EVENTS)
    monkeypatch.setattr(bedrock, "_client", lambda region: fake)
    r = client.post("/v1/chat/stream", json=body(tools=[{"name": "t1", "parameters": {"type": "object"}}]),
                    headers=auth())
    assert r.status_code == 200
    assert fake.seen["toolConfig"]["tools"][0]["toolSpec"]["name"] == "t1"


def test_bedrock_error_is_one_terminal_error_frame(cap, monkeypatch):
    rt, pub = cap
    monkeypatch.setattr(bedrock, "_client",
                        lambda region: FakeBedrock(exc=RuntimeError("ThrottlingException: slow down")))
    r = client.post("/v1/chat/stream", json=body(), headers=auth())
    fr = frames(r.text)
    assert [e for e, _ in fr] == ["start", "error"]
    assert fr[-1][1]["code"] == "rate_limited" and fr[-1][1]["retryable"] is True
    assert "ThrottlingException" not in fr[-1][1]["message"]  # no raw upstream text
    (ev,) = pub.events
    assert ev["status"] == "error"
    assert ev["usage_estimated"] is True and ev["tokens_out"] == 0 and ev["tokens_in"] == 0


def test_mantle_cannot_stream_emits_one_text_frame(cap, monkeypatch):
    async def fake_complete(model_id, body, region, public_id, defaults=None):
        return {"id": "c", "object": "chat.completion", "created": 0, "model": public_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "whole answer"},
                             "finish_reason": "stop", "upstream_stop_reason": "completed"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14,
                          "prompt_tokens_details": {"cached_tokens": 4}}}

    monkeypatch.setattr(mantle, "complete", fake_complete)
    rt, pub = cap
    r = client.post("/v1/chat/stream", json=body("gpt-5.6-sol"), headers=auth())
    fr = frames(r.text)
    assert [e for e, _ in fr] == ["start", "text", "usage", "done"]
    assert fr[1][1]["delta"] == "whole answer"
    assert fr[0][1]["served_by"] == "bedrock"
    ev = pub.events[0]
    assert ev["backend"] == "bedrock_mantle"
    assert (ev["tokens_in"], ev["cache_read_tokens"], ev["tokens_out"]) == (8, 4, 2)


def _sse(*objs):
    return b"".join(f"data: {json.dumps(o)}\n\n".encode() for o in objs) + b"data: [DONE]\n\n"


def test_runner_openai_compatible_stream(cap, monkeypatch):
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, content=_sse(
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
            {"choices": [{"delta": {"content": "a"}}]},
            {"choices": [{"delta": {"content": "b"}, "finish_reason": "length"}]},
            {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}},
        ), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(bs, "_upstream_transport", httpx.MockTransport(handler))
    rt, pub = cap
    r = client.post("/v1/chat/stream", json=body("gemma-4-12b-coder"), headers=auth())
    fr = frames(r.text)
    assert [e for e, _ in fr] == ["start", "thinking", "text", "text", "usage", "done"]
    assert fr[0][1]["served_by"] == "own"
    assert fr[-1][1]["finish_reason"] == "length"
    assert seen["url"].endswith("/chat/completions")
    assert seen["body"]["stream"] is True and seen["body"]["model"] != "gemma-4-12b-coder"
    ev = pub.events[0]
    assert (ev["backend"], ev["served_by"], ev["tokens_in"], ev["tokens_out"]) == ("runner", "own", 9, 2)


def test_runner_upstream_500_is_error_frame(cap, monkeypatch):
    monkeypatch.setattr(bs, "_upstream_transport",
                        httpx.MockTransport(lambda req: httpx.Response(500, content=b"boom")))
    r = client.post("/v1/chat/stream", json=body("gemma-4-12b-coder"), headers=auth())
    fr = frames(r.text)
    assert [e for e, _ in fr] == ["start", "error"]
    assert fr[-1][1]["code"] == "upstream_error" and fr[-1][1]["retryable"] is True


def test_openai_compat_backend_is_translated(cap, monkeypatch):
    entry = next(m for m in catalog.CATALOG if m.public_id == "gpt-5")
    monkeypatch.setattr(entry, "enabled", True)
    monkeypatch.setenv(entry.api_key_env, "test-not-a-real-key")
    seen = {}

    def handler(req):
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, content=_sse({"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}))

    monkeypatch.setattr(bs, "_upstream_transport", httpx.MockTransport(handler))
    rt, pub = cap
    r = client.post("/v1/chat/stream", json=body("gpt-5"), headers=auth())
    assert [e for e, _ in frames(r.text)] == ["start", "text", "usage", "done"]
    assert seen["auth"] == "Bearer test-not-a-real-key"
    assert pub.events[0]["backend"] == "openai_compat"


# ---- pre-stream errors --------------------------------------------------------------

def _err(r):
    return r.status_code, r.json()["error"]["code"]


def test_missing_and_bad_tokens_401(cap):
    assert _err(client.post("/v1/chat/stream", json=body())) == (401, "invalid_token")
    assert _err(client.post("/v1/chat/stream", json=body(), headers=auth("not-a-token"))) == (401, "invalid_token")
    assert _err(client.post("/v1/chat/stream", json=body(), headers=auth(token(OTHER_KEY)))) == (401, "invalid_token")
    old = int(time.time()) - 2000
    assert _err(client.post("/v1/chat/stream", json=body(),
                            headers=auth(token(iat=old, exp=old + 600)))) == (401, "token_expired")
    assert _err(client.post("/v1/chat/stream", json=body(),
                            headers=auth(token(iss="https://evil.example")))) == (401, "invalid_token")


def test_free_tier_402_with_tier_and_billing_state(cap):
    r = client.post("/v1/chat/stream", json=body(),
                    headers=auth(token(tier="free", billing_state="none")))
    assert r.status_code == 402
    err = r.json()["error"]
    assert err["code"] == "payment_required" and err["tier"] == "free" and err["billing_state"] == "none"


def test_suspended_403(cap):
    assert _err(client.post("/v1/chat/stream", json=body(),
                            headers=auth(token(billing_state="suspended")))) == (403, "account_suspended")


def test_missing_scope_403(cap):
    assert _err(client.post("/v1/chat/stream", json=body(),
                            headers=auth(token(scopes=["gen:models"])))) == (403, "missing_scope")


def test_blocked_sub_429_with_retry_after(cap):
    rt, _ = cap
    rt.state.apply_billing({"event_id": "e", "ts": "2026-10-25T00:00:00Z", "sub": "user_1",
                            "state": "blocked", "reason": "spend_cap_reached"})
    r = client.post("/v1/chat/stream", json=body(), headers=auth())
    assert _err(r) == (429, "spend_cap_reached")
    assert int(r.headers["retry-after"]) > 0


def test_revoked_jti_and_sub_401(cap):
    rt, _ = cap
    tok = token(jti="jti-revoked")
    rt.state.apply_revocation({"event_id": "e", "ts": "t", "jti": "jti-revoked", "reason": "logout"})
    assert _err(client.post("/v1/chat/stream", json=body(), headers=auth(tok))) == (401, "invalid_token")
    rt.state.apply_revocation({"event_id": "e", "ts": "t", "sub": "user_2",
                               "not_before": int(time.time()) + 5, "reason": "pw"})
    assert _err(client.post("/v1/chat/stream", json=body(),
                            headers=auth(token(sub="user_2")))) == (401, "invalid_token")


def test_cross_origin_403_and_same_origin_ok(cap, monkeypatch):
    monkeypatch.setattr(bedrock, "_client", lambda region: FakeBedrock(BEDROCK_EVENTS))
    h = auth()
    assert _err(client.post("/v1/chat/stream", json=body(),
                            headers={**h, "Origin": "https://evil.example"})) == (403, "cross_origin")
    assert _err(client.post("/v1/chat/stream", json=body(),
                            headers={**h, "Origin": f"http://{HOST}"})) == (403, "cross_origin")
    assert _err(client.post("/v1/chat/stream", json=body(),
                            headers={**h, "Origin": "null"})) == (403, "cross_origin")
    ok = client.post("/v1/chat/stream", json=body(), headers={**h, "Origin": f"https://{HOST}"})
    assert ok.status_code == 200


def test_extra_allowed_origin(cap, monkeypatch):
    monkeypatch.setattr(bedrock, "_client", lambda region: FakeBedrock(BEDROCK_EVENTS))
    monkeypatch.setenv("OASIS_GENERATION_STREAM_ALLOWED_ORIGINS", "http://localhost:5173")
    get_settings.cache_clear()
    r = client.post("/v1/chat/stream", json=body(), headers={**auth(), "Origin": "http://localhost:5173"})
    assert r.status_code == 200


def test_metering_unavailable_503(cap):
    rt, pub = cap
    pub.ok = False
    rt.meter.spool_max = 0
    rt.meter.emit({"event_id": "x"})
    r = client.post("/v1/chat/stream", json=body(), headers=auth())
    assert _err(r) == (503, "metering_unavailable")


def test_failed_publish_spools_then_flushes(cap, monkeypatch):
    rt, pub = cap
    monkeypatch.setattr(bedrock, "_client", lambda region: FakeBedrock(BEDROCK_EVENTS))
    pub.ok = False
    assert client.post("/v1/chat/stream", json=body(), headers=auth()).status_code == 200
    assert rt.meter.spool_count() == 1 and pub.events == []
    pub.ok = True
    assert rt.meter.flush() == 1
    assert pub.events[0]["status"] == "ok"


def test_duplicate_running_client_request_id_409(cap):
    rt, _ = cap
    b = body()
    assert rt.claim("user_1", b["client_request_id"])
    assert _err(client.post("/v1/chat/stream", json=b, headers=auth())) == (409, "request_in_flight")
    # Another user may use the same id.
    assert rt.claim("user_other", b["client_request_id"])


def test_unknown_model_and_bad_body(cap):
    assert _err(client.post("/v1/chat/stream", json=body("nope"), headers=auth())) == (404, "unknown_model")
    assert _err(client.post("/v1/chat/stream", json={"model": "claude-sonnet-5"}, headers=auth()))[0] == 400
    b = body()
    del b["client_request_id"]
    assert _err(client.post("/v1/chat/stream", json=b, headers=auth())) == (400, "bad_request")
    r = client.post("/v1/chat/stream", content=b"{nope", headers={**auth(), "content-type": "application/json"})
    assert _err(r) == (400, "bad_request")


def test_body_cap_413_content_length_and_chunked(cap, monkeypatch):
    monkeypatch.setenv("MAX_CHAT_BODY_BYTES", "200")
    get_settings.cache_clear()
    big = body(messages=[{"role": "user", "content": "x" * 500}])
    assert _err(client.post("/v1/chat/stream", json=big, headers=auth())) == (413, "request_too_large")

    sent = []

    def chunks():
        for _ in range(100):
            sent.append(1)
            yield b"x" * 100

    r = client.post("/v1/chat/stream", content=chunks(), headers=auth())
    assert _err(r) == (413, "request_too_large")


def test_default_body_cap_is_1_mib(cap):
    assert get_settings().max_chat_body_bytes == 1_048_576


def test_service_mode_has_no_stream_route(monkeypatch):
    monkeypatch.delenv("OASIS_GENERATION_AUTH_MODE", raising=False)
    get_settings.cache_clear()
    assert client.post("/v1/chat/stream", json=body(), headers=auth()).status_code == 404


# ---- stop on client disconnect, keepalive --------------------------------------------

def test_client_disconnect_cancels_upstream_and_bills_stopped(cap, monkeypatch):
    rt, pub = cap
    gate = threading.Event()
    fake = FakeBedrock(BEDROCK_EVENTS, gate=gate)
    monkeypatch.setattr(bedrock, "_client", lambda region: fake)
    settings = get_settings()
    req = bs.parse_body(body())
    assert rt.claim("user_1", req.client_request_id)
    claims = {"sub": "user_1", "jti": "j", "allowance_ref": "allow_user_1_2026-10"}

    async def scenario():
        gen = bs.stream_session(rt=rt, settings=settings, entry=catalog.resolve("claude-sonnet-5"),
                                req=req, claims=claims, request_id="req_t")
        first = await gen.__anext__()          # start
        second = await gen.__anext__()         # thinking (event 0); worker now waits at the gate
        await gen.aclose()                     # the browser went away
        gate.set()
        await asyncio.sleep(0.2)
        return first, second

    first, second = asyncio.run(scenario())
    assert b"event: start" in first and b"event: thinking" in second
    (ev,) = pub.events
    assert ev["status"] == "stopped"
    assert ev["usage_estimated"] is True and ev["tokens_out"] >= 1 and ev["tokens_in"] >= 1
    assert fake.stream.closed  # the Converse stream was closed, not read to the end
    assert rt.claim("user_1", req.client_request_id)  # released


def test_keepalive_comment_when_idle():
    async def slow():
        yield b"event: start\n\n"
        await asyncio.sleep(0.25)
        yield b"event: done\n\n"

    async def collect():
        return [c async for c in bs.with_keepalive(slow(), interval=0.1)]

    out = asyncio.run(collect())
    assert out[0] == b"event: start\n\n" and out[-1] == b"event: done\n\n"
    assert bs.KEEPALIVE_FRAME in out
    assert bs.KEEPALIVE_FRAME == b": ping\n\n"
    assert bs.KEEPALIVE_S == 15.0


def test_keepalive_close_reaches_inner_generator():
    closed = []

    async def inner():
        try:
            yield b"a"
            await asyncio.sleep(10)
            yield b"b"
        finally:
            closed.append(True)

    async def scenario():
        g = bs.with_keepalive(inner(), interval=0.05)
        assert await g.__anext__() == b"a"
        assert await g.__anext__() == bs.KEEPALIVE_FRAME
        await g.aclose()

    asyncio.run(scenario())
    assert closed == [True]


# ---- GET /v1/models in capability mode ----------------------------------------------

def test_models_capability_free_tier_allowed(cap):
    r = client.get("/v1/models", headers=auth(token(tier="free", billing_state="none", scopes=["gen:models"])))
    assert r.status_code == 200
    assert "claude-sonnet-5" in [m["id"] for m in r.json()["data"]]


def test_models_capability_needs_scope_and_token(cap):
    assert _err(client.get("/v1/models")) == (401, "invalid_token")
    assert _err(client.get("/v1/models", headers=auth(token(scopes=["gen:chat"])))) == (403, "missing_scope")
    assert _err(client.get("/v1/models", headers={**auth(), "Origin": "https://evil.example"})) == (403, "cross_origin")


# ---- auth modes: no fallback to "no auth" ---------------------------------------------

def test_capability_mode_never_falls_back_to_no_auth(cap, monkeypatch):
    # No service tokens configured (the dev-gateway task sets none).
    monkeypatch.setattr(bedrock, "_client", lambda region: FakeBedrock(BEDROCK_EVENTS))
    r = client.post("/v1/chat/completions",
                    json={"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 401
    assert client.post("/v1/chat/completions", headers=auth(),
                       json={"model": "claude-sonnet-5", "messages": []}).status_code == 401
    assert client.get("/v1/routing/self").status_code == 401
    assert client.post("/v1/x/decide", json={"model": "m", "labelset_id": "gate.v1"}).status_code == 401
    assert client.get("/v1/models").status_code == 401


def test_both_mode_service_tokens_and_capability(cap, monkeypatch):
    rt, pub = cap
    monkeypatch.setenv("OASIS_GENERATION_AUTH_MODE", "both")
    monkeypatch.setenv("OASIS_GENERATION_SERVICE_TOKENS", "fleet:svc-test-token")
    get_settings.cache_clear()
    assert client.get("/v1/models", headers=auth("svc-test-token")).status_code == 200
    assert client.get("/v1/models", headers=auth()).status_code == 200
    assert client.get("/v1/models", headers=auth("wrong")).status_code == 401
    # A capability token is not a service token on the fleet route.
    assert client.post("/v1/chat/completions", headers=auth(),
                       json={"model": "claude-sonnet-5", "messages": []}).status_code == 401
    # Service-token callers are not metered by the usage-event path.
    monkeypatch.setattr(bedrock, "_client", lambda region: type("F", (), {
        "converse": lambda self, **kw: {"output": {"message": {"content": [{"text": "ok"}]}},
                                        "stopReason": "end_turn",
                                        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}})())
    r = client.post("/v1/chat/completions", headers=auth("svc-test-token"),
                    json={"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200
    assert pub.events == []


def test_both_mode_without_service_tokens_is_not_open(cap, monkeypatch):
    monkeypatch.setenv("OASIS_GENERATION_AUTH_MODE", "both")
    get_settings.cache_clear()
    assert client.post("/v1/chat/completions",
                       json={"model": "claude-sonnet-5", "messages": []}).status_code == 401


@pytest.mark.parametrize("mode", ["capability", "both"])
@pytest.mark.parametrize("missing", ["CAPABILITY_ISSUER", "CAPABILITY_JWKS_URL"])
def test_startup_refuses_without_issuer_or_jwks(monkeypatch, mode, missing):
    monkeypatch.setenv("OASIS_GENERATION_AUTH_MODE", mode)
    monkeypatch.setenv("CAPABILITY_ISSUER", ISS)
    monkeypatch.setenv("CAPABILITY_JWKS_URL", "https://auth.test/.well-known/jwks.json")
    monkeypatch.delenv(missing)
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="refusing to start"):
        main._refuse_open_deployment()
    get_settings.cache_clear()


def test_startup_refuses_plain_http_jwks(monkeypatch):
    monkeypatch.setenv("OASIS_GENERATION_AUTH_MODE", "capability")
    monkeypatch.setenv("CAPABILITY_ISSUER", ISS)
    monkeypatch.setenv("CAPABILITY_JWKS_URL", "http://auth.test/.well-known/jwks.json")
    get_settings.cache_clear()
    with pytest.raises(ValueError):
        main._refuse_open_deployment()
    get_settings.cache_clear()


def test_startup_ok_in_capability_mode_without_service_tokens(monkeypatch):
    monkeypatch.setenv("OASIS_GENERATION_AUTH_MODE", "capability")
    monkeypatch.setenv("CAPABILITY_ISSUER", ISS)
    monkeypatch.setenv("CAPABILITY_JWKS_URL", "https://auth.test/.well-known/jwks.json")
    monkeypatch.delenv("OASIS_GENERATION_SERVICE_TOKENS", raising=False)
    get_settings.cache_clear()
    main._refuse_open_deployment()
    get_settings.cache_clear()


def test_invalid_auth_mode_is_refused(monkeypatch):
    monkeypatch.setenv("OASIS_GENERATION_AUTH_MODE", "open")
    get_settings.cache_clear()
    with pytest.raises(Exception):
        get_settings()
    get_settings.cache_clear()
