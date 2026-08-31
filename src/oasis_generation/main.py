"""oasis-generation gateway.

OpenAI-compatible facade in front of one or more model backends. Routes each
request by the resolved catalog entry's `backend` (GEN-003 provider fanout):

  - runner        -> httpx passthrough to the self-hosted engine
                     (Docker Model Runner locally, vLLM in the cloud).
  - openai_compat -> httpx passthrough to another OpenAI-compatible provider
                     (per-entry base_url + bearer key).
  - bedrock       -> Amazon Bedrock via the Converse API (see bedrock.py).

Provider keys live only here (the gateway), so bot egress collapses to
gateway + Telegram. Wake-on-request lands behind the same routes later (GEN-004).
"""

import os

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import bedrock, mantle
from .catalog import CatalogEntry, enabled_models, resolve
from .config import Settings, get_settings

app = FastAPI(title="oasis-generation", version="0.0.1")


def check_auth(request: Request, settings: Settings = Depends(get_settings)) -> None:
    if not settings.tokens:
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


# Per-request usage log (ADM-050). One JSONL line per call: who, which model,
# which backend, how many tokens. Deliberately records NO prompt or completion
# content — attribution needs neither, and storing prompts turns a cost ledger
# into a data-retention problem.
USAGE_LOG = os.environ.get("OASIS_GENERATION_USAGE_LOG", "/usage/requests.jsonl")


def _record_usage(
    client: str, entry: CatalogEntry, usage: dict | None, streamed: bool, status: int
) -> None:
    """Append one usage record. Never raises: a cost-ledger write must not be
    able to fail a model call the caller is waiting on."""
    try:
        import datetime as _dt
        import json as _json

        u = usage or {}
        rec = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "client": client,
            "model": entry.public_id,
            "backend": entry.backend,
            "status": status,
            "streamed": streamed,
            # Absent on streamed calls whose backend omits a usage block; left
            # null rather than zero so "unknown" is distinguishable from "free".
            "prompt_tokens": u.get("prompt_tokens") or u.get("input_tokens"),
            "completion_tokens": u.get("completion_tokens") or u.get("output_tokens"),
            "cached_tokens": (
                (u.get("prompt_tokens_details") or {}).get("cached_tokens")
                or u.get("cache_read_input_tokens")
            ),
        }
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
def list_models() -> dict:
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
            streamed=False, status=resp.status_code,
        )
    return JSONResponse(payload, status_code=resp.status_code)


def _client_of(request: Request) -> str:
    return getattr(request.state, "client", "unknown")


def _logged_json(request: Request, entry: CatalogEntry, payload: dict) -> JSONResponse:
    """Return the payload, recording its usage block on the way out."""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    _record_usage(_client_of(request), entry, usage, streamed=False, status=200)
    return JSONResponse(payload)


def _logged_stream(request: Request, entry: CatalogEntry, gen) -> StreamingResponse:
    """Streamed calls are recorded at dispatch, with token counts left null.

    The generator is consumed by the client, not here, so waiting for a usage
    block would mean buffering the whole stream — which would defeat streaming.
    A row with null tokens still gives call counts per bot per model, and the
    authoritative cost stays the CUR either way.
    """
    _record_usage(_client_of(request), entry, None, streamed=True, status=200)
    return StreamingResponse(gen, media_type="text/event-stream")


@app.post("/v1/chat/completions", dependencies=[Depends(check_auth)])
async def chat_completions(request: Request, settings: Settings = Depends(get_settings)):
    body = await request.json()
    entry = resolve(body.get("model", ""))
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown or disabled model: {body.get('model')!r}")

    if entry.backend == "bedrock":
        try:
            if body.get("stream"):
                return _logged_stream(request, entry, bedrock.stream(
                    entry.upstream_id, body, settings.bedrock_region,
                    entry.public_id, entry.inference,
                ))
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
                return _logged_stream(request, entry, mantle.stream(
                    entry.upstream_id, body, settings.bedrock_region,
                    entry.public_id, entry.inference,
                ))
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
