"""BYOK registry tests (GEN-008). Network is a httpx.MockTransport; DNS is stubbed."""

import asyncio
import json
import socket

import httpx
import pytest

from oasis_generation import ssrf
from oasis_generation.byok import ByokError, ByokRegistry

KEY = "super-secret-key-value"


def _dns(monkeypatch, mapping):
    def fake(host, port, type=0, **_):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in mapping[host]]
    monkeypatch.setattr(ssrf.socket, "getaddrinfo", fake)


def run(c):
    return asyncio.run(c)


def oai_handler(*, tool=True, usage=True, stream_done=True, status=200):
    def h(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        if status != 200:
            return httpx.Response(status, json={"error": "x"})
        if body.get("stream"):
            sse = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            if stream_done:
                sse += "data: [DONE]\n\n"
            return httpx.Response(200, text=sse)
        msg = {"content": "ok"}
        if body.get("tools"):
            msg = {"content": None, "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]} if tool else {"content": "no tool"}
        out = {"choices": [{"message": msg}]}
        if usage:
            out["usage"] = {"prompt_tokens": 5, "completion_tokens": 2}
        return httpx.Response(200, json=out)
    return h


def reg(handler, kind="openai_compat", region="eu-west-1"):
    r = ByokRegistry(key_resolver=lambda ref: KEY if ref == "ACME_KEY" else None,
                     transport=httpx.MockTransport(handler))
    r.register("acme", {"kind": kind, "base_url": "https://api.acme.example/v1",
                        "key_ref": "ACME_KEY", "model": "m1", "region": region})
    return r


def test_conformance_pass_activates(monkeypatch):
    _dns(monkeypatch, {"api.acme.example": ["93.184.216.34"]})
    r = reg(oai_handler())
    out = run(r.run_conformance("acme"))
    assert out["status"] == "active", out
    assert r.active_names() == {"acme"} and r.active_regions() == {"acme": "eu-west-1"}


@pytest.mark.parametrize("kw,failing", [
    ({"tool": False}, "tool"),
    ({"usage": False}, "usage"),
    ({"stream_done": False}, "stream"),
    ({"status": 500}, "plain"),
])
def test_conformance_failures_block_activation(monkeypatch, kw, failing):
    _dns(monkeypatch, {"api.acme.example": ["93.184.216.34"]})
    r = reg(oai_handler(**kw))
    out = run(r.run_conformance("acme"))
    assert out["status"] == "failed" and out["checks"][failing] != "ok"
    assert r.active_names() == set()


def test_connection_is_pinned_and_host_header_kept(monkeypatch):
    _dns(monkeypatch, {"api.acme.example": ["93.184.216.34"]})
    seen = []
    inner = oai_handler()

    def h(req):
        seen.append((req.url.host, req.headers.get("host"), req.extensions.get("sni_hostname")))
        return inner(req)
    run(reg(h).run_conformance("acme"))
    assert seen and all(s == ("93.184.216.34", "api.acme.example", "api.acme.example") for s in seen)


def test_dns_rebind_to_private_refused_at_call_time(monkeypatch):
    # public at register time, private at call time
    _dns(monkeypatch, {"api.acme.example": ["93.184.216.34"]})
    r = reg(oai_handler())
    _dns(monkeypatch, {"api.acme.example": ["169.254.169.254"]})
    out = run(r.run_conformance("acme"))
    assert out["status"] == "failed"
    # every probe was blocked before any request left the gateway
    assert out["checks"]["plain"].startswith("refused")
    assert out["checks"]["stream"].startswith("refused")
    assert out["checks"]["tool"].startswith("refused")
    assert all(v != "ok" for v in out["checks"].values())


def test_register_rejects_bad_urls(monkeypatch):
    _dns(monkeypatch, {"internal.example": ["10.0.0.9"]})
    r = ByokRegistry()
    for url in ("http://api.example.com/v1", "https://internal.example/v1", "https://169.254.169.254/v1",
                "https://[::1]/v1", "https://user:pw@api.example.com/v1"):
        with pytest.raises(ByokError):
            r.register("x", {"kind": "openai_compat", "base_url": url, "key_ref": "K", "model": "m"})


def test_redirect_is_not_followed(monkeypatch):
    _dns(monkeypatch, {"api.acme.example": ["93.184.216.34"]})
    calls = []

    def h(req):
        calls.append(str(req.url))
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})
    out = run(reg(h).run_conformance("acme"))
    assert out["status"] == "failed"
    assert not any("169.254" in c for c in calls)


def test_key_never_leaks_into_report(monkeypatch):
    _dns(monkeypatch, {"api.acme.example": ["93.184.216.34"]})
    r = reg(oai_handler(status=401))
    out = run(r.run_conformance("acme"))
    assert KEY not in json.dumps(out)


def test_unresolvable_key_ref_fails_without_value(monkeypatch):
    _dns(monkeypatch, {"api.acme.example": ["93.184.216.34"]})
    r = ByokRegistry(key_resolver=lambda ref: None, transport=httpx.MockTransport(oai_handler()))
    r.register("acme", {"kind": "openai_compat", "base_url": "https://api.acme.example/v1",
                        "key_ref": "MISSING", "model": "m"})
    out = run(r.run_conformance("acme"))
    assert out["status"] == "failed"


def test_key_value_pasted_as_key_ref_rejected():
    r = ByokRegistry()
    for bad in ("sk-abc123", "AKIAABCDEFG", "x" * 80):
        with pytest.raises(Exception):
            r.register("a", {"kind": "openai_compat", "base_url": "https://a.example/v1",
                             "key_ref": bad, "model": "m"})


def test_anthropic_conformance(monkeypatch):
    _dns(monkeypatch, {"api.acme.example": ["93.184.216.34"]})

    def h(req):
        assert req.headers["x-api-key"] == KEY and req.headers["anthropic-version"]
        b = json.loads(req.content)
        if b.get("stream"):
            ev = ('data: {"type":"content_block_delta","delta":{"text":"ok"}}\n\n'
                  'data: {"type":"message_stop"}\n\n')
            return httpx.Response(200, text=ev)
        if b.get("tools"):
            c = [{"type": "tool_use", "name": "get_weather", "input": {"city": "Paris"}}]
        else:
            c = [{"type": "text", "text": "ok"}]
        return httpx.Response(200, json={"content": c, "usage": {"input_tokens": 3, "output_tokens": 1}})
    out = run(reg(h, kind="anthropic").run_conformance("acme"))
    assert out["status"] == "active", out


def test_bedrock_role_needs_role_arn_and_runner():
    r = ByokRegistry()
    with pytest.raises(ByokError):
        r.register("b", {"kind": "bedrock_role"})
    r.register("b", {"kind": "bedrock_role", "role_arn": "arn:aws:iam::1:role/x", "region": "eu-central-1"})
    assert run(r.run_conformance("b"))["status"] == "failed"   # no runner -> never active

    async def ok(_e):
        return {"plain": True, "stream": True, "tool": True, "usage": True}
    r2 = ByokRegistry(bedrock_runner=ok)
    r2.register("b", {"kind": "bedrock_role", "role_arn": "arn:aws:iam::1:role/x"})
    assert run(r2.run_conformance("b"))["status"] == "active"
