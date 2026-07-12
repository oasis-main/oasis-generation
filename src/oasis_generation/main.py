"""oasis-generation gateway.

OpenAI-compatible facade in front of one or more model runners. v0 proxies
chat completions to a single upstream engine (Docker Model Runner locally,
vLLM in the cloud); provider fanout and wake-on-request land behind the same
routes later (GEN-003, GEN-004).
"""

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .catalog import enabled_models, resolve
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
    return {
        "object": "list",
        "data": [
            {"id": m.public_id, "object": "model", "owned_by": "oasis-generation", "tier": m.tier}
            for m in enabled_models()
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(check_auth)])
async def chat_completions(request: Request, settings: Settings = Depends(get_settings)):
    body = await request.json()
    entry = resolve(body.get("model", ""))
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown or disabled model: {body.get('model')!r}")
    body["model"] = entry.upstream_id

    url = f"{settings.upstream_base_url}/chat/completions"
    client = httpx.AsyncClient(timeout=settings.upstream_timeout_s)

    if body.get("stream"):
        upstream = client.stream("POST", url, json=body)

        async def relay():
            try:
                async with upstream as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            finally:
                await client.aclose()

        return StreamingResponse(relay(), media_type="text/event-stream")

    try:
        resp = await client.post(url, json=body)
        payload = resp.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"upstream engine unreachable: {exc}") from exc
    finally:
        await client.aclose()
    # Restore the public id so callers never see upstream-internal names.
    if isinstance(payload, dict):
        payload["model"] = entry.public_id
    return JSONResponse(payload, status_code=resp.status_code)
