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

import asyncio
import logging
import os
import re

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from pydantic import BaseModel

from . import bedrock, browser_stream, mantle, wiring
from .browser_stream import ApiError
from .capability import check_inference_allowed, check_jwks_url
from .byok import ByokError
from .catalog import CatalogEntry, enabled_models, resolve
from .config import Settings, get_settings
from .routes import NoRoute, Target
from .remote import TARGETS, RouteStore, read_secret_file, token_matches

app = FastAPI(title="oasis-generation", version="0.0.1")


logger = logging.getLogger(__name__)


def check_startup_config(s: Settings) -> None:
    """Refuse to start a deployment that would serve without auth.

    GEN-008: an internet-facing gateway (require_auth) with no service tokens.
    ORG-171: capability or both mode without a JWKS URL or an issuer (then no
    token could be verified, and nothing must fall back to "no auth")."""
    if s.auth_mode == "service" and s.require_auth and not s.tokens:
        raise RuntimeError(
            "OASIS_GENERATION_REQUIRE_AUTH is set but OASIS_GENERATION_SERVICE_TOKENS is empty; refusing to start"
        )
    if s.auth_mode == "both" and s.require_auth and not s.tokens:
        raise RuntimeError(
            "OASIS_GENERATION_REQUIRE_AUTH is set but OASIS_GENERATION_SERVICE_TOKENS is empty; refusing to start"
        )
    if s.capability_enabled:
        if not s.capability_issuer or not s.capability_jwks_url:
            raise RuntimeError(
                f"OASIS_GENERATION_AUTH_MODE={s.auth_mode} needs CAPABILITY_JWKS_URL and "
                "CAPABILITY_ISSUER; refusing to start"
            )
        check_jwks_url(s.capability_jwks_url)


def _refuse_open_deployment() -> None:
    check_startup_config(get_settings())


@app.on_event("startup")
async def _startup() -> None:
    s = get_settings()
    check_startup_config(s)
    if s.capability_enabled:
        rt = browser_stream.runtime(s)
        rt.meter.check_writable()
        rt.start_bridge(s)
        app.state.spool_flusher = asyncio.create_task(_flush_spool_forever(rt))


@app.on_event("shutdown")
async def _shutdown() -> None:
    task = getattr(app.state, "spool_flusher", None)
    if task is not None:
        task.cancel()
    rt = browser_stream._runtime
    if rt is not None and rt.bridge is not None:
        rt.bridge.stop()


async def _flush_spool_forever(rt, interval_s: float = 5.0) -> None:
    """Resend spooled usage events in the background (interface doc 5.3)."""
    while True:
        await asyncio.sleep(interval_s)
        try:
            sent = await asyncio.to_thread(rt.meter.flush)
            if sent:
                logger.info("usage spool: resent %d event(s)", sent)
        except Exception:  # noqa: BLE001 - never let the flusher die
            logger.exception("usage spool flush failed")


@app.exception_handler(ApiError)
async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(exc.body(), status_code=exc.status, headers=exc.headers)


def check_auth(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """Service-token auth (the fleet). Never "auth disabled" once capability auth is on."""
    if settings.auth_mode == "capability":
        # Capability mode serves only the capability routes; a service-token
        # route must not become an open, unmetered door when no tokens are set.
        raise HTTPException(status_code=401, detail="service tokens are not accepted in capability auth mode")
    if not settings.tokens:
        if settings.auth_mode == "both":
            raise HTTPException(status_code=401, detail="invalid or missing service token")
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


@app.get("/v1/models")
async def list_models(request: Request, settings: Settings = Depends(get_settings)):
    token = browser_stream.bearer(request)
    use_capability = settings.auth_mode == "capability" or (
        settings.auth_mode == "both" and token not in settings.token_clients)
    if use_capability:
        # ORG-171: any valid token with gen:models; the free tier may list models.
        browser_stream.check_same_origin(request, settings)
        await browser_stream.authenticate(token, browser_stream.runtime(settings), "gen:models")
        request.state.served_by = "local"
        return _local_models()
    check_auth(request, settings)
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


async def _dispatch_target(tg: Target, body: dict, entry: CatalogEntry | None,
                           table, settings: Settings, request: Request):
    """Send one request to one route-table target. Raises on failure so the
    caller can try the next target; returns a Response on success."""
    model = body.get("model", "")
    # Calls through byok/remote need no catalog entry; give logging a stand-in.
    ent = entry or CatalogEntry(public_id=model, upstream_id=model, tier="routed", enabled=True)
    if tg.kind == "own":
        ep = table.endpoint(tg)
        key = os.environ.get(ep.api_key_env or "", "").strip() if ep.api_key_env else ""
        if ep.api_key_env and not key:
            raise HTTPException(502, f"{tg}: {ep.api_key_env} not set on the gateway")
        return await _proxy_openai_compatible(
            body, ent, ep.base_url, settings, {"Authorization": f"Bearer {key}"} if key else None, request)
    if tg.kind == "remote":
        ep = table.endpoint(tg)
        key = os.environ.get(ep.api_key_env or "", "").strip() if ep.api_key_env else ""
        # The remote gateway resolves the public id itself: keep it, do not map to upstream_id.
        remote_ent = ent.model_copy(update={"upstream_id": model})
        return await _proxy_openai_compatible(
            body, remote_ent, ep.base_url, settings, {"Authorization": f"Bearer {key}"} if key else None, request)
    if tg.kind == "bedrock":
        if entry is None or entry.backend != "bedrock":
            raise HTTPException(502, f"{tg}: model {model!r} has no bedrock catalog entry")
        region = table.data.bedrock[tg.name].region
        if body.get("stream"):
            return _logged_stream(request, entry, bedrock.stream(
                entry.upstream_id, body, region, entry.public_id, entry.inference))
        payload = await bedrock.complete(entry.upstream_id, body, region, entry.public_id, entry.inference)
        return _logged_json(request, entry, payload)
    if tg.kind == "byok":
        reg = wiring.byok(settings)
        try:
            reg.check_owner(tg.name, _client_of(request))   # S1: a key is used only for its owner
            if body.get("stream"):
                reg.forward_ready(tg.name)  # fail before returning a stream
                return _logged_stream(request, ent, reg.forward_stream(tg.name, body))
            code, payload = await reg.forward(tg.name, body)
        except ByokError as exc:
            raise HTTPException(502, f"{tg}: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(502, f"{tg}: non-JSON upstream body (HTTP {code})")
        payload["model"] = model
        if code == 200:
            return _logged_json(request, ent, payload)
        return JSONResponse(payload, status_code=code)
    raise HTTPException(500, f"unhandled target kind {tg.kind}")


async def _routed_chat(request: Request, body: dict, settings: Settings, table_holder):
    table = table_holder.current()
    # Tenant = the caller name the service token maps to (ADM-050 attribution).
    tenant = _client_of(request)
    model = body.get("model", "")
    try:
        resolved = table.resolve(tenant, model)
    except NoRoute as exc:
        # 422, not 404/502: the request is understood but the tenant's limits
        # (residency/fallback) leave nowhere legal to send it.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    entry = resolve(model)
    last: Exception | None = None
    for i, tg in enumerate(resolved.targets):
        try:
            resp = await _dispatch_target(tg, body, entry, table, settings, request)
            if getattr(resp, "status_code", 200) >= 500 and i + 1 < len(resolved.targets):
                last = HTTPException(resp.status_code, f"{tg} returned {resp.status_code}")
                continue
            return resp
        except HTTPException as exc:
            last = exc
        except Exception as exc:  # noqa: BLE001 - boto/network errors -> try the next target
            last = exc
        # Streams are not retried once they started; failures above happen before that.
    detail = getattr(last, "detail", None) or str(last)
    raise HTTPException(status_code=502, detail=f"all targets failed for {model!r}: {detail}")


@app.post("/v1/chat/completions", dependencies=[Depends(check_auth)])
async def chat_completions(request: Request, settings: Settings = Depends(get_settings)):
    if settings.auth_mode == "capability":
        # Unreachable today (check_auth refuses every caller in this mode); kept
        # so the body cap holds if this route ever accepts capability tokens.
        body = await browser_stream.read_json_capped(request, settings.max_chat_body_bytes)
    else:
        body = await request.json()
    if _route_of(request, settings) == "remote":
        return await _forward_remote(request, settings, "POST", "/chat/completions", body)
    request.state.served_by = "local"
    table_holder = wiring.routes(settings)
    if table_holder is not None:
        return await _routed_chat(request, body, settings, table_holder)
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


# ---- Browser stream (ORG-171) ------------------------------------------------
@app.post("/v1/chat/stream")
async def chat_stream(request: Request, settings: Settings = Depends(get_settings)):
    """BROWSER_STREAM_CONTRACT.md: capability token, SSE events, metered."""
    if not settings.capability_enabled:
        raise ApiError(404, "not_found", "capability auth is not enabled on this gateway")
    rt = browser_stream.runtime(settings)
    browser_stream.check_same_origin(request, settings)
    claims = await browser_stream.authenticate(browser_stream.bearer(request), rt, "gen:chat")
    decision = check_inference_allowed(claims)
    if decision == 403:
        raise ApiError(403, "account_suspended", "the account is suspended")
    if decision == 402:
        raise ApiError(402, "payment_required", "add a card to use AI",
                       extra={"tier": claims.get("tier"), "billing_state": claims.get("billing_state")})
    if rt.state.is_blocked(claims["sub"]):
        raise ApiError(429, "spend_cap_reached", "the monthly spend cap is reached",
                       headers={"Retry-After": str(browser_stream.seconds_to_next_month(rt.clock()))})
    raw = await browser_stream.read_json_capped(request, settings.max_chat_body_bytes)
    req = browser_stream.parse_body(raw)
    entry = resolve(req.model)
    if entry is None or entry.backend not in browser_stream.STREAM_BACKENDS:
        raise ApiError(404, "unknown_model", f"unknown or disabled model: {req.model!r}")
    if not rt.meter.available():
        raise ApiError(503, "metering_unavailable", "usage metering is unavailable; retry later",
                       headers={"Retry-After": "30"})
    if not rt.claim(claims["sub"], req.client_request_id):
        raise ApiError(409, "request_in_flight", "a request with this client_request_id is still running")
    request_id = browser_stream.new_request_id()
    request.state.served_by = "local"
    gen = browser_stream.stream_session(
        rt=rt, settings=settings, entry=entry, req=req, claims=claims, request_id=request_id)
    return StreamingResponse(
        browser_stream.with_keepalive(gen), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Request-Id": request_id})


# ---- X endpoints (GEN-008) -------------------------------------------------
class DecideIn(BaseModel):
    model: str
    labelset_id: str
    ctx: dict | list | str | None = None


class GateIn(BaseModel):
    model: str
    action: dict
    ctx: dict | list | str | None = None
    baseline: str | None = None   # tighten-only floor; absent = "escalate"


@app.post("/v1/x/decide", dependencies=[Depends(check_auth)])
async def x_decide(body: DecideIn, settings: Settings = Depends(get_settings)):
    """Stateless X call. Returns the x_decision record; failures abstain, never 5xx."""
    try:
        return await wiring.xservice(settings).decide(
            model=body.model, labelset_id=body.labelset_id, ctx=body.ctx)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\"")) from exc


@app.post("/v1/gate", dependencies=[Depends(check_auth)])
async def x_gate(body: GateIn, settings: Settings = Depends(get_settings)):
    """Verdict on a proposed side effect. Restrictive-only; fails closed to escalate."""
    try:
        return await wiring.xservice(settings).gate(
            model=body.model, action=body.action, ctx=body.ctx, baseline=body.baseline)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/byok/{name}/conformance", dependencies=[Depends(check_admin)])
async def byok_conformance(name: str, settings: Settings = Depends(get_settings)):
    """Run the four-check conformance run; the provider becomes routable only on a pass.

    ADMIN token only (review S1): the run spends the provider owner's key (4 billed
    calls) and flips the provider to routable, so a service token must not reach it."""
    reg = wiring.byok(settings)
    try:
        return await reg.run_conformance(name)
    except ByokError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
