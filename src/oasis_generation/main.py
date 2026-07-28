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
        return
    auth = request.headers.get("authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    if token not in settings.tokens:
        raise HTTPException(status_code=401, detail="invalid or missing service token")


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
    body: dict, entry: CatalogEntry, base_url: str, settings: Settings, headers: dict | None
):
    """Forward an OpenAI-shaped request to an OpenAI-compatible upstream and
    relay the response (streaming or not). Shared by the runner and
    openai_compat backends — they differ only in base_url + auth header."""
    body = {**body, "model": entry.upstream_id}
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

        return StreamingResponse(relay(), media_type="text/event-stream")

    try:
        resp = await client.post(url, json=body, headers=headers)
        payload = resp.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"upstream engine unreachable: {exc}") from exc
    finally:
        await client.aclose()
    # Restore the public id so callers never see upstream-internal names.
    if isinstance(payload, dict):
        payload["model"] = entry.public_id
    return JSONResponse(payload, status_code=resp.status_code)


@app.post("/v1/chat/completions", dependencies=[Depends(check_auth)])
async def chat_completions(request: Request, settings: Settings = Depends(get_settings)):
    body = await request.json()
    entry = resolve(body.get("model", ""))
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown or disabled model: {body.get('model')!r}")

    if entry.backend == "bedrock":
        try:
            if body.get("stream"):
                return StreamingResponse(
                    bedrock.stream(
                        entry.upstream_id, body, settings.bedrock_region,
                        entry.public_id, entry.inference,
                    ),
                    media_type="text/event-stream",
                )
            payload = await bedrock.complete(
                entry.upstream_id, body, settings.bedrock_region,
                entry.public_id, entry.inference,
            )
            return JSONResponse(payload)
        except HTTPException:
            raise
        except Exception as exc:  # boto/credential/model errors -> 502
            raise HTTPException(status_code=502, detail=f"bedrock error: {exc}") from exc

    if entry.backend == "bedrock_mantle":
        # GPT-5.6-sol — OpenAI Responses API on the bedrock-mantle endpoint (SigV4).
        try:
            if body.get("stream"):
                return StreamingResponse(
                    mantle.stream(
                        entry.upstream_id, body, settings.bedrock_region,
                        entry.public_id, entry.inference,
                    ),
                    media_type="text/event-stream",
                )
            payload = await mantle.complete(
                entry.upstream_id, body, settings.bedrock_region,
                entry.public_id, entry.inference,
            )
            return JSONResponse(payload)
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
        return await _proxy_openai_compatible(body, entry, entry.base_url, settings, headers)

    # runner (default): passthrough to the self-hosted engine, no auth header.
    return await _proxy_openai_compatible(body, entry, settings.upstream_base_url, settings, None)
