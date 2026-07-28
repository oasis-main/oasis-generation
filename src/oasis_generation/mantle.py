"""Amazon Bedrock **mantle** backend for the gateway (GPT-5.6-sol).

Some models — currently OpenAI GPT-5.6-sol — are served ONLY on the Bedrock
*mantle* endpoint via the OpenAI **Responses API**, not the Converse API. This
module translates the gateway's OpenAI chat-completions surface <-> Responses so
GPT-5.6-sol sits behind the same OpenAI-compatible facade as everything else.

Auth is SigV4 (service "bedrock") with the gateway's AWS credential chain —
verified live 2026-07-24; no separate Bedrock API key is needed. Endpoint:
  https://bedrock-mantle.{region}.api.aws/openai/v1/responses

v0 is text-first: system/user/assistant turns translate; tool-calling and true
token streaming are follow-ups (see notes). `store` is forced False so prompts
and responses are not persisted by the platform.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from .bedrock import _content_to_text, _resolve_inference

if TYPE_CHECKING:
    from .catalog import InferenceDefaults

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 300.0

# GPT-5.6-sol reasoning-effort ladder (verified live 2026-07-24):
# none | low | medium | high | xhigh. Map the shared effort vocabulary onto it.
_RESPONSES_EFFORTS = {"none", "low", "medium", "high", "xhigh"}
_VERBOSITY_BY_PROFILE = {"fast": "low", "balanced": "medium", "deep": "high"}


def _responses_effort(effort: str | None) -> str | None:
    if not effort:
        return None
    e = str(effort).lower()
    if e == "minimal":
        return "low"
    if e == "max":
        return "xhigh"
    return e if e in _RESPONSES_EFFORTS else "medium"


@lru_cache(maxsize=1)
def _session() -> "boto3.Session":
    return boto3.Session()


def _endpoint(region: str) -> str:
    return f"https://bedrock-mantle.{region}.api.aws/openai/v1/responses"


def _sign(url: str, data: str, region: str) -> dict[str, str]:
    """SigV4-sign a POST to the mantle endpoint (service 'bedrock')."""
    creds = _session().get_credentials().get_frozen_credentials()
    req = AWSRequest(method="POST", url=url, data=data, headers={"Content-Type": "application/json"})
    SigV4Auth(creds, "bedrock", region).add_auth(req)
    return dict(req.headers)


def openai_to_responses(
    model_id: str, body: dict, defaults: "InferenceDefaults | None" = None
) -> dict[str, Any]:
    """OpenAI chat-completions body -> Bedrock-mantle Responses request.

    System/developer messages fold into top-level `instructions`; user/assistant
    turns become `input` message items (input_text / output_text parts). Inference
    params come from the model's InferenceDefaults/profile (effort -> reasoning.effort,
    profile -> text.verbosity, max_tokens -> max_output_tokens). `store` is False.
    """
    instructions: list[str] = []
    items: list[dict] = []
    for msg in body.get("messages", []):
        role = msg.get("role")
        text = _content_to_text(msg.get("content"))
        if role in ("system", "developer"):
            if text:
                instructions.append(text)
            continue
        if role == "tool":
            # Tool results are not translated in v0 (Responses uses a distinct
            # function_call_output item shape) — see module notes.
            continue
        if not text:
            continue
        conv_role = "assistant" if role == "assistant" else "user"
        part_type = "output_text" if conv_role == "assistant" else "input_text"
        items.append({"type": "message", "role": conv_role, "content": [{"type": part_type, "text": text}]})

    _thinking, effort, max_tokens, allow_temp, _allow_samp = _resolve_inference(body, defaults)

    payload: dict[str, Any] = {
        "model": model_id,
        "input": items if items else "",
        "max_output_tokens": max_tokens,
        "store": False,  # privacy: do not persist prompts/responses on the platform
    }
    if instructions:
        payload["instructions"] = "\n\n".join(instructions)
    reff = _responses_effort(effort)
    if reff:
        payload["reasoning"] = {"effort": reff}
    profile_name = body.get("profile") or (defaults.default_profile if defaults else None)
    verbosity = _VERBOSITY_BY_PROFILE.get(profile_name) if profile_name else None
    if verbosity:
        payload["text"] = {"verbosity": verbosity}
    # GPT-5.6-sol rejects temperature; only forwarded if the model's policy allows it.
    if body.get("temperature") is not None and allow_temp:
        payload["temperature"] = float(body["temperature"])
    return payload


def _finish_reason(resp: dict) -> str:
    if resp.get("status") == "incomplete":
        reason = (resp.get("incomplete_details") or {}).get("reason")
        return "length" if reason == "max_output_tokens" else "stop"
    return "stop"


def responses_to_openai(resp: dict, public_id: str, completion_id: str, created: int) -> dict:
    """Bedrock-mantle Responses object -> OpenAI chat.completion."""
    text_parts: list[str] = []
    for item in resp.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                text_parts.append(part.get("text") or "")
    usage = resp.get("usage") or {}
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": public_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(text_parts)},
                "finish_reason": _finish_reason(resp),
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


async def complete(
    model_id: str, body: dict, region: str, public_id: str,
    defaults: "InferenceDefaults | None" = None,
) -> dict:
    payload = openai_to_responses(model_id, body, defaults)
    data = json.dumps(payload)
    url = _endpoint(region)
    headers = _sign(url, data, region)
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        resp = await client.post(url, content=data, headers=headers)
    if resp.status_code >= 400:
        logger.error("mantle complete() %s for %s: %s", resp.status_code, public_id, resp.text[:300])
        raise RuntimeError(f"bedrock-mantle {resp.status_code}: {resp.text[:300]}")
    return responses_to_openai(resp.json(), public_id, f"chatcmpl-{uuid.uuid4().hex}", int(time.time()))


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


async def stream(
    model_id: str, body: dict, region: str, public_id: str,
    defaults: "InferenceDefaults | None" = None,
) -> AsyncIterator[bytes]:
    """v0 streaming: the Responses call is non-streaming, then the completed
    answer is emitted as a single OpenAI SSE content chunk. True token streaming
    (Responses SSE events -> OpenAI deltas) is a follow-up."""
    result = await complete(model_id, {**body, "stream": False}, region, public_id, defaults)
    cid = result["id"]
    created = result["created"]
    choice = result["choices"][0]
    content = choice["message"]["content"]

    def chunk(delta: dict, finish: str | None = None) -> dict:
        return {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": public_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    yield _sse(chunk({"role": "assistant"}))
    if content:
        yield _sse(chunk({"content": content}))
    yield _sse(chunk({}, choice["finish_reason"]))
    yield b"data: [DONE]\n\n"
