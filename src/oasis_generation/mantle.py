"""Amazon Bedrock **mantle** backend for the gateway (GPT-5.6-sol).

Some models — currently OpenAI GPT-5.6-sol — are served ONLY on the Bedrock
*mantle* endpoint via the OpenAI **Responses API**, not the Converse API. This
module translates the gateway's OpenAI chat-completions surface <-> Responses so
GPT-5.6-sol sits behind the same OpenAI-compatible facade as everything else.

Auth is SigV4 (service "bedrock") with the gateway's AWS credential chain —
verified live 2026-07-24; no separate Bedrock API key is needed. Endpoint:
  https://bedrock-mantle.{region}.api.aws/openai/v1/responses

Full modality (all verified live 2026-07-24):
  - text: system -> instructions; user/assistant -> input message items.
  - images: OpenAI image_url content parts -> Responses `input_image`.
  - tools: OpenAI `tools`/`tool_choice` -> Responses FLAT function tools
    ({type,name,parameters}); assistant tool_calls -> `function_call` items;
    role "tool" results -> `function_call_output` items; response `function_call`
    output -> OpenAI tool_calls.
`store` is forced False so prompts/responses are not persisted by the platform.
Streaming is a single-chunk fake-stream (true token streaming is a follow-up).
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


def _to_input_content(content: Any, role: str) -> list[dict]:
    """OpenAI message content (str | list of parts) -> Responses content parts.

    Text -> input_text (user) / output_text (assistant); OpenAI image_url parts
    (either {"url": ...} or a bare string) -> Responses `input_image`.
    """
    text_type = "output_text" if role == "assistant" else "input_text"
    parts: list[dict] = []
    if isinstance(content, str):
        if content:
            parts.append({"type": text_type, "text": content})
        return parts
    if isinstance(content, list):
        for p in content:
            if isinstance(p, str):
                if p:
                    parts.append({"type": text_type, "text": p})
                continue
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype in (None, "text", "input_text", "output_text") and p.get("text"):
                parts.append({"type": text_type, "text": str(p["text"])})
            elif ptype == "image_url":
                iu = p.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else iu
                if url:
                    parts.append({"type": "input_image", "image_url": url})
            elif ptype == "input_image" and p.get("image_url"):
                parts.append({"type": "input_image", "image_url": p["image_url"]})
    return parts


def _tools_to_responses(body: dict) -> list[dict] | None:
    """OpenAI `tools` -> Responses FLAT function tools ({type,name,parameters}).
    tool_choice=="none" drops tools entirely (the cross-model-safe way to disable)."""
    tools = body.get("tools")
    if not tools or body.get("tool_choice") == "none":
        return None
    specs: list[dict] = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        spec: dict[str, Any] = {
            "type": "function",
            "name": name,
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        }
        if fn.get("description"):
            spec["description"] = fn["description"]
        specs.append(spec)
    return specs or None


def _tool_choice_to_responses(choice: Any) -> Any | None:
    """OpenAI tool_choice -> Responses tool_choice ("auto"/"required" or
    {type:function,name}). "none"/"auto"/None are handled by caller/default."""
    if choice in ("auto", "required"):
        return choice
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = (choice.get("function") or {}).get("name")
        if name:
            return {"type": "function", "name": name}
    return None


def _tool_call_arguments(raw: Any) -> str:
    """Responses `arguments` is a JSON string; tolerate an already-parsed dict."""
    if isinstance(raw, str):
        return raw
    return json.dumps(raw or {})


def openai_to_responses(
    model_id: str, body: dict, defaults: "InferenceDefaults | None" = None
) -> dict[str, Any]:
    """OpenAI chat-completions body -> Bedrock-mantle Responses request.

    System/developer -> `instructions`; user/assistant -> `input` message items
    (text + images); assistant `tool_calls` -> `function_call` items; role "tool"
    -> `function_call_output`; OpenAI `tools`/`tool_choice` -> Responses tools.
    Inference params come from the model's InferenceDefaults/profile. `store` False.
    """
    instructions: list[str] = []
    items: list[dict] = []
    for msg in body.get("messages", []):
        role = msg.get("role")
        if role in ("system", "developer"):
            text = _content_to_text(msg.get("content"))
            if text:
                instructions.append(text)
            continue
        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id") or "",
                # Converse rejects empty; keep a placeholder if the tool returned nothing.
                "output": _content_to_text(msg.get("content")) or "(no content)",
            })
            continue
        if role == "assistant":
            content_parts = _to_input_content(msg.get("content"), "assistant")
            if content_parts:
                items.append({"type": "message", "role": "assistant", "content": content_parts})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                items.append({
                    "type": "function_call",
                    "call_id": tc.get("id") or "",
                    "name": fn.get("name") or "",
                    "arguments": _tool_call_arguments(fn.get("arguments")),
                })
            continue
        # user (and any other role) -> a user message item.
        content_parts = _to_input_content(msg.get("content"), "user")
        if content_parts:
            items.append({"type": "message", "role": "user", "content": content_parts})

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
    tools = _tools_to_responses(body)
    if tools:
        payload["tools"] = tools
        choice = _tool_choice_to_responses(body.get("tool_choice"))
        if choice is not None:
            payload["tool_choice"] = choice
    return payload


def _finish_reason(resp: dict) -> str:
    if resp.get("status") == "incomplete":
        reason = (resp.get("incomplete_details") or {}).get("reason")
        return "length" if reason == "max_output_tokens" else "stop"
    return "stop"


def responses_to_openai(resp: dict, public_id: str, completion_id: str, created: int) -> dict:
    """Bedrock-mantle Responses object -> OpenAI chat.completion.

    `message` output items -> assistant text; `function_call` items -> OpenAI
    tool_calls (arguments already a JSON string). A tool-call turn sets content
    null and finish_reason "tool_calls", per OpenAI convention.
    """
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for item in resp.get("output") or []:
        itype = item.get("type")
        if itype == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    text_parts.append(part.get("text") or "")
        elif itype == "function_call":
            tool_calls.append({
                "id": item.get("call_id") or item.get("id") or "",
                "type": "function",
                "function": {
                    "name": item.get("name") or "",
                    "arguments": item.get("arguments") or "{}",
                },
            })
    text = "".join(text_parts)
    message: dict[str, Any] = {"role": "assistant", "content": text if text else (None if tool_calls else "")}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = resp.get("usage") or {}
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": public_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else _finish_reason(resp),
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
    answer (text and/or tool_calls) is emitted as OpenAI SSE chunks. True token
    streaming (Responses SSE events -> OpenAI deltas) is a follow-up."""
    result = await complete(model_id, {**body, "stream": False}, region, public_id, defaults)
    cid = result["id"]
    created = result["created"]
    choice = result["choices"][0]
    message = choice["message"]

    def chunk(delta: dict, finish: str | None = None) -> dict:
        return {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": public_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    yield _sse(chunk({"role": "assistant"}))
    if message.get("content"):
        yield _sse(chunk({"content": message["content"]}))
    for i, tc in enumerate(message.get("tool_calls") or []):
        yield _sse(chunk({"tool_calls": [{
            "index": i,
            "id": tc["id"],
            "type": "function",
            "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
        }]}))
    yield _sse(chunk({}, choice["finish_reason"]))
    yield b"data: [DONE]\n\n"
