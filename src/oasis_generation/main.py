"""oasis-generation gateway.

OpenAI-compatible facade in front of one or more model backends. Routes each
request by the resolved catalog entry's `backend` (GEN-003 provider fanout):

  - runner        -> httpx passthrough to the self-hosted engine
                     (Docker Model Runner locally, vLLM in the cloud).
  - openai_compat -> httpx passthrough to another OpenAI-compatible provider
                     (per-entry base_url + bearer key).
  - bedrock       -> Amazon Bedrock via the Converse API (see bedrock.py).

Before any of that, a caller switched to "remote" (remote.py, GEN-008) is
forwarded whole to the other gateway, and this gateway only records the call.

Provider keys live only here (the gateway), so bot egress collapses to
gateway + Telegram. Wake-on-request lands behind the same routes later (GEN-004).
"""

import os
import re

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import bedrock, mantle
from .catalog import CatalogEntry, enabled_models, resolve
from .config import Settings, get_settings
from .remote import TARGETS, RouteStore, read_secret_file, token_matches

app = FastAPI(title="oasis-generation", version="0.0.1")


@app.on_event("startup")
def _refuse_open_deployment() -> None:
    """An internet-facing gateway with no tokens would serve anyone (GEN-008)."""
    s = get_settings()
    if s.require_auth and not s.tokens:
        raise RuntimeError(
            "OASIS_GENERATION_REQUIRE_AUTH is set but OASIS_GENERATION_SERVICE_TOKENS is empty; refusing to start"
        )


def check_auth(request: Request, settings: Settings = Depends(get_settings)) -> None:
    if not settings.tokens:
        if settings.require_auth:
            raise HTTPException(status_code=503, detail="gateway has no service tokens configured")
        request.state.client = "auth-disabled"
        return
    auth = request.headers.get("authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    clients = settings.token_clients
    if token not in clients:
        raise HTTPException(status_code=401, detail="invalid or missing service token")
    # Attribution rides on the request from here (ADM-050). The token is NOT
    # stored anywhere — only the name it maps to.
    request.state.client = clients[token]
    # A gateway that forwards a bot's call (remote.py) names the bot here, so
    # this gateway's ledger can say which bot behind that gateway called.
    # Informational only: it never changes auth or routing.
    origin = request.headers.get("x-oasis-origin-client", "")
    if _CLIENT_NAME.fullmatch(origin):
        request.state.origin_client = origin


_CLIENT_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}")


@app.middleware("http")
async def _served_by_header(request: Request, call_next):
    """Every model response says which gateway served it (GEN-008 §11.6)."""
    response = await call_next(request)
    served = getattr(request.state, "served_by", None)
    if served:
        response.headers["X-Oasis-Served-By"] = served
    return response


# Per-request usage log (ADM-050). One JSONL line per call: who, which model,
# which backend, how many tokens. Deliberately records NO prompt or completion
# content — attribution needs neither, and storing prompts turns a cost ledger
# into a data-retention problem.
USAGE_LOG = os.environ.get("OASIS_GENERATION_USAGE_LOG", "/usage/requests.jsonl")


def _record_usage(
    client: str, entry: CatalogEntry, usage: dict | None, streamed: bool, status: int,
    origin_client: str | None = None,
    stop_reason: str | None = None, finish_reason: str | None = None,
) -> None:
    _record_usage_raw(
        client, entry.public_id, entry.backend, usage, streamed, status, origin_client,
        stop_reason, finish_reason,
    )


def _record_usage_raw(
    client: str, model: str, backend: str, usage: dict | None, streamed: bool, status: int,
    origin_client: str | None = None,
    stop_reason: str | None = None, finish_reason: str | None = None,
) -> None:
    """Append one usage record. Never raises: a cost-ledger write must not be
    able to fail a model call the caller is waiting on."""
    try:
        import datetime as _dt
        import json as _json

        u = usage or {}

        def first(*keys):
            # Not `a or b`: a real 0 (an empty, filtered turn) must stay 0.
            return next((u[k] for k in keys if u.get(k) is not None), None)

        rec = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "client": client,
            "model": model,
            "backend": backend,
            "status": status,
            "streamed": streamed,
            # Absent on streamed calls whose backend omits a usage block; left
            # null rather than zero so "unknown" is distinguishable from "free".
            "prompt_tokens": first("prompt_tokens", "input_tokens"),
            "completion_tokens": first("completion_tokens", "output_tokens"),
            "cached_tokens": (
                (u.get("prompt_tokens_details") or {}).get("cached_tokens")
                or u.get("cache_read_input_tokens")
            ),
        }
        # Raw upstream stop reason (Converse stopReason, or the Responses
        # status) and the OpenAI finish_reason it became. Only the Bedrock
        # backends report them; null elsewhere. Added 2026-10-18: a
        # content_filtered turn with 0 output tokens is otherwise invisible.
        if stop_reason is not None or finish_reason is not None:
            rec["stop_reason"] = stop_reason
            rec["finish_reason"] = finish_reason
        if origin_client:
            rec["origin_client"] = origin_client
        path = USAGE_LOG
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001 — ledger is best-effort, never fatal
        pass


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "oasis-generation"}


@app.get("/v1/models", dependencies=[Depends(check_auth)])
async def list_models(request: Request, settings: Settings = Depends(get_settings)):
    if _route_of(request, settings) == "remote":
        return await _forward_remote(request, settings, "GET", "/models", None)
    request.state.served_by = "local"
    return _local_models()


def _local_models() -> dict:
    # `capabilities` is an oasis-generation extension to the OpenAI model object
    # (standard clients ignore unknown fields); it drives the Telegram /genconfig
    # surface so each model shows only the controls it actually supports (§7.7).
    return {
        "object": "list",
        "data": [
            {
                "id": m.public_id,
                "object": "model",
                "owned_by": "oasis-generation",
                "tier": m.tier,
                "capabilities": m.capabilities(),
            }
            for m in enabled_models()
        ],
    }


async def _proxy_openai_compatible(
    body: dict, entry: CatalogEntry, base_url: str, settings: Settings,
    headers: dict | None, request: Request | None = None,
):
    """Forward an OpenAI-shaped request to an OpenAI-compatible upstream and
    relay the response (streaming or not). Shared by the runner and
    openai_compat backends — they differ only in base_url + auth header."""
    body = {**body, "model": entry.upstream_id}

    # OpenAI's GPT-5 generation rejects `max_tokens` outright — "Unsupported
    # parameter: 'max_tokens' is not supported with this model. Use
    # 'max_completion_tokens' instead" (HTTP 400, observed live 2026-08-28).
    # openclaw sends the older field, so translate it here rather than asking
    # every caller to know which upstream wants which spelling.
    #
    # Scoped to api.openai.com on purpose: Google's OpenAI-compatible endpoint
    # still accepts `max_tokens`, and renaming it there would break Gemini to
    # fix OpenAI. Both were verified against this exact code path.
    if "api.openai.com" in base_url and "max_tokens" in body:
        body["max_completion_tokens"] = body.pop("max_tokens")

    url = f"{base_url.rstrip('/')}/chat/completions"
    client = httpx.AsyncClient(timeout=settings.upstream_timeout_s)

    if body.get("stream"):
        upstream = client.stream("POST", url, json=body, headers=headers)

        async def relay():
            try:
                async with upstream as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            finally:
                await client.aclose()

        if request is not None:
            return _logged_stream(request, entry, relay())
        return StreamingResponse(relay(), media_type="text/event-stream")

    try:
        resp = await client.post(url, json=body, headers=headers)
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"upstream engine unreachable: {exc}") from exc
    finally:
        await client.aclose()

    # resp.json() USED TO SIT INSIDE THE try ABOVE, whose except catches only
    # httpx.HTTPError. A non-JSON body therefore raised JSONDecodeError, escaped
    # this function, and surfaced as an opaque "500 Internal Server Error" with
    # no detail — which is exactly how gemma-4-12b-coder/-agentic failed for
    # weeks after their local weights were removed on 2026-07-13 (the upstream
    # returns an empty body, not JSON). Parse separately and report the real
    # condition as the 502 this code always intended.
    try:
        payload = resp.json()
    except ValueError:
        snippet = (resp.text or "")[:200]
        raise HTTPException(
            status_code=502,
            detail=(
                f"{entry.public_id}: upstream returned a non-JSON body "
                f"(HTTP {resp.status_code}). Usually means the engine for this "
                f"model is not running. Body: {snippet!r}"
            ),
        ) from None

    # Restore the public id so callers never see upstream-internal names.
    if isinstance(payload, dict):
        payload["model"] = entry.public_id
    if request is not None and resp.status_code == 200:
        _record_usage(
            _client_of(request), entry,
            payload.get("usage") if isinstance(payload, dict) else None,
            streamed=False, status=resp.status_code, origin_client=_origin_of(request),
        )
    return JSONResponse(payload, status_code=resp.status_code)


def _client_of(request: Request) -> str:
    return getattr(request.state, "client", "unknown")


def _origin_of(request: Request) -> str | None:
    return getattr(request.state, "origin_client", None)


# ── Per-client route switch (GEN-008, remote.py) ────────────────────────────

# Tests swap this for an httpx.MockTransport; production leaves it None.
_remote_transport: httpx.AsyncBaseTransport | None = None


def _store(settings: Settings) -> RouteStore:
    return RouteStore(settings.route_state_file)


def _route_of(request: Request, settings: Settings) -> str:
    # A call that already came through another gateway is never forwarded
    # again: two gateways pointed at each other would otherwise loop.
    if _origin_of(request):
        return "local"
    return _store(settings).get(_client_of(request))


async def _forward_remote(
    request: Request, settings: Settings, method: str, path: str, body: dict | None
):
    """Forward one call, unchanged, to the remote gateway. No fallback."""
    request.state.served_by = "remote"
    client_name = _client_of(request)
    model = (body or {}).get("model", "") if body else ""
    token = read_secret_file(settings.remote_token_file)
    if not settings.remote_base_url.startswith("https://") or not token:
        raise HTTPException(
            status_code=502,
            detail=f"client {client_name!r} is routed to the remote gateway, but no remote "
            "gateway is configured on this gateway. Switch it back to local.",
        )
    url = f"{settings.remote_base_url.rstrip('/')}{path}"
    headers = {"Authorization": f"Bearer {token}", "X-Oasis-Origin-Client": client_name}
    client = httpx.AsyncClient(timeout=settings.upstream_timeout_s, transport=_remote_transport)

    if body is not None and body.get("stream"):
        upstream = client.stream(method, url, json=body, headers=headers)

        async def relay():
            try:
                async with upstream as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            finally:
                await client.aclose()

        _record_usage_raw(client_name, model, "remote", None, True, 200, _origin_of(request))
        return StreamingResponse(relay(), media_type="text/event-stream")

    try:
        resp = await client.request(method, url, json=body, headers=headers)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"remote gateway unreachable: {type(exc).__name__}"
        ) from exc
    finally:
        await client.aclose()
    try:
        payload = resp.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail=f"remote gateway returned a non-JSON body (HTTP {resp.status_code})",
        ) from None
    if body is not None:
        usage = payload.get("usage") if isinstance(payload, dict) else None
        _record_usage_raw(client_name, model, "remote", usage, False, resp.status_code, _origin_of(request))
    return JSONResponse(payload, status_code=resp.status_code)


def _known_clients(settings: Settings) -> list[str]:
    return sorted(set(settings.token_clients.values()) - {"unattributed"})


def _routing_view(settings: Settings) -> dict:
    snap = _store(settings).snapshot()
    return {
        "targets": list(TARGETS),
        "remote_configured": settings.remote_configured,
        "remote_base_url": settings.remote_base_url or None,
        "clients": {c: snap.get(c, "local") for c in _known_clients(settings)},
    }


def check_admin(request: Request, settings: Settings = Depends(get_settings)) -> None:
    expected = read_secret_file(settings.admin_token_file)
    if not expected:
        raise HTTPException(status_code=503, detail="admin API disabled (no admin token file)")
    presented = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not token_matches(presented, expected):
        raise HTTPException(status_code=401, detail="invalid admin token")


def _parse_target(body: object, settings: Settings) -> str:
    target = body.get("target") if isinstance(body, dict) else None
    if target not in TARGETS:
        raise HTTPException(status_code=400, detail=f"target must be one of {list(TARGETS)}")
    if target == "remote" and not settings.remote_configured:
        raise HTTPException(status_code=409, detail="no remote gateway is configured on this gateway")
    return target


@app.get("/admin/routing", dependencies=[Depends(check_admin)])
def admin_routing(settings: Settings = Depends(get_settings)) -> dict:
    return _routing_view(settings)


@app.put("/admin/routing/{client_name}", dependencies=[Depends(check_admin)])
async def admin_set_routing(
    client_name: str, request: Request, settings: Settings = Depends(get_settings)
) -> dict:
    if client_name not in _known_clients(settings):
        raise HTTPException(status_code=404, detail=f"unknown client {client_name!r}")
    target = _parse_target(await request.json(), settings)
    prev, new = _store(settings).set(client_name, target, actor="admin")
    return {"client": client_name, "from": prev, "to": new}


@app.get("/v1/routing/self", dependencies=[Depends(check_auth)])
def self_routing(request: Request, settings: Settings = Depends(get_settings)) -> dict:
    me = _client_of(request)
    return {
        "client": me,
        "target": _store(settings).get(me),
        "targets": list(TARGETS),
        "remote_configured": settings.remote_configured,
        "may_switch": settings.may_self_switch(me),
    }


@app.put("/v1/routing/self", dependencies=[Depends(check_auth)])
async def self_set_routing(request: Request, settings: Settings = Depends(get_settings)) -> dict:
    me = _client_of(request)
    if not settings.may_self_switch(me):
        raise HTTPException(status_code=403, detail=f"client {me!r} may not switch its own route")
    target = _parse_target(await request.json(), settings)
    prev, new = _store(settings).set(me, target, actor=f"self:{me}")
    return {"client": me, "from": prev, "to": new}


def _logged_json(request: Request, entry: CatalogEntry, payload: dict) -> JSONResponse:
    """Return the payload, recording its usage block on the way out."""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    choice = ((payload.get("choices") or [{}])[0] if isinstance(payload, dict) else {}) or {}
    _record_usage(_client_of(request), entry, usage, streamed=False, status=200,
                  origin_client=_origin_of(request),
                  stop_reason=choice.get("upstream_stop_reason"),
                  finish_reason=choice.get("finish_reason"))
    return JSONResponse(payload)


def _logged_stream(
    request: Request, entry: CatalogEntry, gen, outcome: dict | None = None,
) -> StreamingResponse:
    """Streamed calls are recorded when the stream ENDS, not at dispatch.

    The row is written as the generator finishes (or is closed by a client
    disconnect), so it can carry what the backend put in `outcome`: usage,
    stop_reason and finish_reason (the Bedrock backends fill it; others leave
    the token counts null). Nothing is buffered: chunks pass straight through.

    Before 2026-10-18 the row was written at dispatch with null tokens, so a
    streamed turn that Bedrock cut off with content_filtered and 0 output
    tokens looked identical to a normal one in the ledger.
    """
    client_name = _client_of(request)
    origin = _origin_of(request)

    async def recorded():
        try:
            async for chunk in gen:
                yield chunk
        finally:
            o = outcome or {}
            _record_usage(client_name, entry, o.get("usage"), streamed=True, status=200,
                          origin_client=origin,
                          stop_reason=o.get("stop_reason"),
                          finish_reason=o.get("finish_reason"))

    return StreamingResponse(recorded(), media_type="text/event-stream")


@app.post("/v1/chat/completions", dependencies=[Depends(check_auth)])
async def chat_completions(request: Request, settings: Settings = Depends(get_settings)):
    body = await request.json()
    if _route_of(request, settings) == "remote":
        return await _forward_remote(request, settings, "POST", "/chat/completions", body)
    request.state.served_by = "local"
    entry = resolve(body.get("model", ""))
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown or disabled model: {body.get('model')!r}")

    if entry.backend == "bedrock":
        try:
            if body.get("stream"):
                outcome: dict = {}
                return _logged_stream(request, entry, bedrock.stream(
                    entry.upstream_id, body, settings.bedrock_region,
                    entry.public_id, entry.inference, outcome=outcome,
                ), outcome)
            payload = await bedrock.complete(
                entry.upstream_id, body, settings.bedrock_region,
                entry.public_id, entry.inference,
            )
            return _logged_json(request, entry, payload)
        except HTTPException:
            raise
        except Exception as exc:  # boto/credential/model errors -> 502
            raise HTTPException(status_code=502, detail=f"bedrock error: {exc}") from exc

    if entry.backend == "bedrock_mantle":
        # GPT-5.6-sol — OpenAI Responses API on the bedrock-mantle endpoint (SigV4).
        try:
            if body.get("stream"):
                outcome = {}
                return _logged_stream(request, entry, mantle.stream(
                    entry.upstream_id, body, settings.bedrock_region,
                    entry.public_id, entry.inference, outcome=outcome,
                ), outcome)
            payload = await mantle.complete(
                entry.upstream_id, body, settings.bedrock_region,
                entry.public_id, entry.inference,
            )
            return _logged_json(request, entry, payload)
        except HTTPException:
            raise
        except Exception as exc:  # signing / network / model errors -> 502
            raise HTTPException(status_code=502, detail=f"bedrock-mantle error: {exc}") from exc

    if entry.backend == "openai_compat":
        if not entry.base_url:
            raise HTTPException(status_code=500, detail=f"{entry.public_id}: openai_compat entry missing base_url")
        api_key = os.environ.get(entry.api_key_env or "", "").strip()
        if entry.api_key_env and not api_key:
            raise HTTPException(
                status_code=502,
                detail=f"{entry.public_id}: {entry.api_key_env} not set on the gateway",
            )
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        return await _proxy_openai_compatible(
            body, entry, entry.base_url, settings, headers, request
        )

    # runner (default): passthrough to the self-hosted engine, no auth header.
    return await _proxy_openai_compatible(
        body, entry, settings.upstream_base_url, settings, None, request
    )
