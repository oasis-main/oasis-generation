"""Amazon Bedrock backend for the gateway (GEN-003).

Translates OpenAI chat-completions <-> Bedrock Converse so the gateway can front
Bedrock-hosted models (Claude, GLM, DeepSeek, Llama, Nova, GPT-OSS, …) behind
the same OpenAI-compatible surface as the self-hosted runners. Direct boto3
(not a proxy library) so brand-new 2026 model ids work regardless of any third
party's model map — the Converse API takes an arbitrary model/inference-profile
id and streams generically.

Auth is the AWS SDK default credential chain (env keys / profile / role): the
gateway process holds the key, the bots never do — which is the GEN-003 security
interlock (bot egress collapses to gateway + Telegram).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import boto3
from botocore.config import Config as BotoConfig

if TYPE_CHECKING:  # avoid a runtime import cycle; only used for typing.
    from .catalog import InferenceDefaults

logger = logging.getLogger(__name__)

# Bedrock Converse requires a maxTokens; use a generous default when the caller
# omits it (some clients only send it on demand).
_DEFAULT_MAX_TOKENS = 4096

# OpenAI-wire `reasoning_effort` -> Anthropic/Bedrock effort. openclaw emits
# minimal/low/medium/high; xhigh/max pass through if a caller uses them.
_EFFORT_ALIASES = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}


def _resolve_inference(
    body: dict, defaults: "InferenceDefaults | None"
) -> tuple[str, str | None, int, bool, bool]:
    """Resolve the effective inference knobs for one request.

    Precedence (design doc §7.5): explicit request field > selected profile >
    per-model catalog default. Returns (thinking_mode, effort, max_tokens,
    allow_temperature, allow_sampling). With no catalog `inference` block this
    degrades to legacy behaviour (native thinking, sampling allowed, default
    max_tokens) so untyped models keep working unchanged.
    """
    thinking = defaults.thinking if defaults else "native"
    allow_temp = defaults.allow_temperature if defaults else True
    allow_samp = defaults.allow_sampling if defaults else True

    effort: str | None = None
    profile_max: int | None = None
    if defaults and defaults.profiles:
        name = body.get("profile") or defaults.default_profile
        prof = defaults.profiles.get(name) or defaults.profiles.get(defaults.default_profile)
        if prof:
            effort = prof.effort
            profile_max = prof.max_tokens

    # An explicit reasoning_effort on the request overrides the profile's effort.
    requested_effort = body.get("reasoning_effort")
    if requested_effort:
        effort = _EFFORT_ALIASES.get(str(requested_effort).lower(), effort)

    # max_tokens: explicit request wins, else the profile, else the flat default.
    max_tokens = int(body.get("max_tokens") or profile_max or _DEFAULT_MAX_TOKENS)
    return thinking, effort, max_tokens, allow_temp, allow_samp

# Converse stopReason -> OpenAI finish_reason.
_FINISH_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
}


@lru_cache(maxsize=8)
def _client(region: str):
    # Reuse one client per region across requests (thread-safe for calls).
    # "standard" mode retries transient errors (throttling, 5xx, timeouts) with
    # exponential backoff + full jitter *before* we ever see them — applies to
    # the initial request that opens converse_stream(), not to a stream already
    # in progress, so it can't duplicate partial output. max_attempts=4 (1
    # original + 3 retries) rides out a brief throttle without masking a real
    # outage for too long. The gateway wraps this in its own timeout.
    cfg = BotoConfig(retries={"max_attempts": 4, "mode": "standard"}, read_timeout=300)
    return boto3.client("bedrock-runtime", region_name=region, config=cfg)


def _content_to_text(content: Any) -> str:
    """Flatten OpenAI message content (str | list of parts) to plain text.

    Text-only projection (image parts dropped). Used for system messages and
    tool results; user turns go through `_content_to_converse_blocks`, which
    keeps images. Reasoning is handled on the response side, not here.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in (None, "text") and "text" in part:
                out.append(str(part["text"]))
            elif isinstance(part, str):
                out.append(part)
        return "".join(out)
    return "" if content is None else str(content)


# data:image/<fmt>;base64,<data>  — the self-contained image form (no fetch).
_DATA_URI_RE = re.compile(r"^data:image/(png|jpe?g|gif|webp);base64,(.+)$", re.IGNORECASE | re.DOTALL)


def _image_block(image_url: Any) -> dict | None:
    """OpenAI image_url -> Converse image content block, or None if unusable.

    Converse needs raw bytes, so only self-contained **data: URIs** are supported
    (decoded here, no network). Remote http(s) URLs are NOT fetched — that would
    add an SSRF surface on the gateway; supporting them (with egress guards) is a
    follow-up. mantle/Responses models handle remote URLs upstream instead.
    """
    if not isinstance(image_url, str):
        return None
    m = _DATA_URI_RE.match(image_url.strip())
    if not m:
        return None
    fmt = m.group(1).lower()
    if fmt == "jpg":
        fmt = "jpeg"
    try:
        raw = base64.b64decode(m.group(2), validate=True)
    except (ValueError, Exception):  # noqa: BLE001 — malformed base64 must not crash a request
        return None
    if not raw:
        return None
    return {"image": {"format": fmt, "source": {"bytes": raw}}}


def _content_to_converse_blocks(content: Any) -> list[dict]:
    """OpenAI message content (str | list) -> Converse content blocks (text +
    images). Used for user turns so vision-capable models (Claude) receive images.
    Consecutive text parts are merged into one block (preserving order around images)."""
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if not isinstance(content, list):
        return [] if content is None else [{"text": str(content)}]

    blocks: list[dict] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            joined = "".join(buf)
            buf.clear()
            if joined:
                blocks.append({"text": joined})

    for part in content:
        if isinstance(part, str):
            if part:
                buf.append(part)
        elif isinstance(part, dict):
            ptype = part.get("type")
            if ptype in (None, "text") and part.get("text"):
                buf.append(str(part["text"]))
            elif ptype == "image_url":
                iu = part.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else iu
                block = _image_block(url)
                if block:
                    flush()  # keep any preceding text before the image, in order
                    blocks.append(block)
    flush()
    return blocks


def _parse_tool_args(raw: Any) -> dict:
    """OpenAI carries tool-call arguments as a JSON *string*; Converse toolUse
    wants the parsed object. Tolerate dicts (already parsed), empty, and
    malformed JSON (model mid-stream fragments should never crash a request)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _tool_choice_to_converse(choice: Any) -> dict | None:
    """OpenAI tool_choice -> Converse toolChoice.

    "auto"/None -> None: Converse defaults to auto when tools are present, and
    omitting the field is the most cross-model-compatible option (some Bedrock
    models reject an explicit "any"/"tool" choice). "none" is handled upstream
    in _tools_to_converse (drop the tools entirely).
    """
    if choice == "required":
        return {"any": {}}
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = (choice.get("function") or {}).get("name")
        if name:
            return {"tool": {"name": name}}
    return None


def _tools_to_converse(body: dict) -> dict | None:
    """OpenAI `tools` + `tool_choice` -> Converse `toolConfig`, or None when the
    caller sent no usable tools (or tool_choice="none")."""
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
            "name": name,
            # Converse nests the JSON Schema one level down under "json".
            "inputSchema": {"json": fn.get("parameters") or {"type": "object", "properties": {}}},
        }
        if fn.get("description"):
            spec["description"] = fn["description"]
        specs.append({"toolSpec": spec})
    if not specs:
        return None
    cfg: dict[str, Any] = {"tools": specs}
    choice = _tool_choice_to_converse(body.get("tool_choice"))
    if choice is not None:
        cfg["toolChoice"] = choice
    return cfg


def _message_to_blocks(msg: dict) -> tuple[str, list[dict]]:
    """Return (converse_role, content_blocks) for one non-system OpenAI message.

    Preserves tool structure so multi-turn tool loops survive the round-trip:
      assistant.tool_calls -> {"toolUse": ...} blocks
      role "tool"          -> a user-turn {"toolResult": ...} block
    """
    role = msg.get("role")
    if role == "tool":
        # A tool result is delivered to Converse inside a *user* turn.
        result_text = _content_to_text(msg.get("content"))
        return "user", [
            {
                "toolResult": {
                    "toolUseId": msg.get("tool_call_id") or "",
                    # Converse rejects an empty content list; keep a placeholder.
                    "content": [{"text": result_text or "(no content)"}],
                }
            }
        ]
    if role == "assistant":
        blocks: list[dict] = []
        text = _content_to_text(msg.get("content"))
        if text:
            blocks.append({"text": text})
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            blocks.append(
                {
                    "toolUse": {
                        "toolUseId": tc.get("id") or "",
                        "name": fn.get("name") or "",
                        "input": _parse_tool_args(fn.get("arguments")),
                    }
                }
            )
        return "assistant", blocks
    # user (and any other role) -> user text + image blocks (vision-capable models).
    return "user", _content_to_converse_blocks(msg.get("content"))


# Prompt caching (2026-10-01). Converse caches ONLY up to an explicit
# `cachePoint` block, and an OpenAI chat body carries no such marker, so before
# this the gateway sent every fleet turn uncached: September dev Bedrock was
# ~$1,001 of input against $0.02 of cache reads (Yes Man's Cost Explorer pull).
# Claude on Bedrock allows up to 4 cache points; the gateway places 3, matching
# Anthropic's prefix order tools -> system -> messages:
#   1. end of the tool list    (stable for a bot's whole life)
#   2. end of the system prompt (stable per session; the entrypoint sorts paths
#      so the bytes do not move)
#   3. end of the final user turn — the rolling point. The next agent-loop turn
#      re-sends this exact prefix plus a few new blocks, and Anthropic's 20-block
#      lookback finds the previous point, so each turn reads the history as a
#      cache hit and writes only the new tail.
# A prefix shorter than the model minimum is simply not cached (no error).
# Non-Anthropic Bedrock models (Nova, Llama, GLM, GPT) reject or ignore the
# block, so they never get one. OASIS_GENERATION_PROMPT_CACHE=0 turns it off.
#
# CACHE ONLY WHAT RECURS (same day, after the first deploy). A cache write
# costs 1.25x input and pays off only if the same prefix is READ within the
# ~5-minute TTL. Unconditional points made one-shot traffic dearer: the
# oasis-reviewer judge puts a per-call random nonce in its SYSTEM prompt, so
# every judge call wrote ~13k tokens at 1.25x and none were ever read. So:
#   - tools / system points go on only when the same (model, prefix) hash was
#     seen within the TTL — a never-repeating prefix stays at 1x;
#   - the rolling message point goes on only for a continuing conversation
#     (an assistant turn exists) whose system prefix recurred, because history
#     after a changing system prompt can never be read back either.
# In-process memory of hashes only (no prompt text); a restart just costs one
# uncached call per prefix.
_CACHE_POINT = {"cachePoint": {"type": "default"}}
_CACHE_TTL_S = 300.0
_PREFIX_SEEN_MAX = 4096
_prefix_seen: dict[str, float] = {}
_history_seen: dict[str, tuple[int, str]] = {}
# A child of uvicorn's INFO-level logger: the app sets no logging config, so a
# plain module logger's INFO lines would be dropped.
_cache_log = logging.getLogger("uvicorn.error.prompt_cache")


def supports_prompt_cache(model_id: str) -> bool:
    if os.environ.get("OASIS_GENERATION_PROMPT_CACHE", "1") == "0":
        return False
    return "anthropic.claude" in (model_id or "")


def _recurred(key: str, now: float) -> bool:
    """True if `key` was seen within the cache TTL; records this sighting."""
    last = _prefix_seen.get(key)
    _prefix_seen[key] = now
    if len(_prefix_seen) > _PREFIX_SEEN_MAX:
        for old in sorted(_prefix_seen, key=_prefix_seen.get)[: _PREFIX_SEEN_MAX // 4]:
            del _prefix_seen[old]
    return last is not None and now - last <= _CACHE_TTL_S


def _digest(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _add_cache_points(kwargs: dict[str, Any], cache_key: str, now: float | None = None) -> dict:
    now = time.monotonic() if now is None else now
    tools = (kwargs.get("toolConfig") or {}).get("tools") or []
    system = kwargs.get("system") or []
    messages = kwargs.get("messages") or []
    tools_h = _digest(cache_key, tools)
    system_h = _digest(cache_key, tools, system)
    placed = []
    if tools and _recurred("t:" + tools_h, now):
        tools.append(dict(_CACHE_POINT))
        placed.append("tools")
    system_recurred = bool(system) and _recurred("s:" + system_h, now)
    if system_recurred:
        system.append(dict(_CACHE_POINT))
        placed.append("system")
    # Diagnostic: is the history up to the previous call (same system prefix)
    # byte-identical? "no" means the client rewrote earlier turns (pruning,
    # compaction), which no cache point can survive. Hashes only.
    history = "-"
    if system:
        prev = _history_seen.get(system_h)
        if prev and len(messages) >= prev[0]:
            history = "yes" if _digest(messages[: prev[0]]) == prev[1] else "no"
        _history_seen[system_h] = (len(messages), _digest(messages))
        if len(_history_seen) > _PREFIX_SEEN_MAX:
            for old in list(_history_seen)[: _PREFIX_SEEN_MAX // 4]:
                del _history_seen[old]
    # Rolling point: only after a user turn (a trailing assistant turn is a
    # prefill) in a conversation that already has an assistant turn.
    continuing = any(m["role"] == "assistant" for m in messages[:-1])
    if (system_recurred and continuing and messages
            and messages[-1]["role"] == "user" and messages[-1]["content"]):
        # 4th point (Claude's maximum): the end of the user turn before the
        # latest assistant turn — exactly where the PREVIOUS call put its
        # rolling point. Anthropic only looks back 20 blocks from a cache
        # point, and one assistant turn with many parallel tool calls adds
        # more than that, so without this the previous cache entry is missed.
        last_asst = max(i for i, m in enumerate(messages[:-1]) if m["role"] == "assistant")
        if last_asst > 0 and messages[last_asst - 1]["role"] == "user":
            messages[last_asst - 1]["content"].append(dict(_CACHE_POINT))
            placed.append("previous")
        messages[-1]["content"].append(dict(_CACHE_POINT))
        placed.append("messages")
    return {"tools": tools_h if tools else "-", "system": system_h if system else "-",
            "n_messages": len(messages), "placed": placed, "history": history}


def openai_to_converse(
    body: dict, defaults: "InferenceDefaults | None" = None, prompt_cache: bool = False,
    cache_key: str = "",
) -> dict[str, Any]:
    """Build converse()/converse_stream() kwargs (minus modelId) from an OpenAI
    chat-completions body, applying the model's inference policy (`defaults`).

    The policy controls thinking/effort (via `additionalModelRequestFields`),
    the per-profile max_tokens, and whether temperature/top_p are allowed —
    keeping every request valid for the target model (e.g. no `temperature` on
    the adaptive-Claude removed-set, which 400s). `defaults=None` = legacy
    passthrough."""
    system_parts: list[dict] = []
    messages: list[dict] = []
    for msg in body.get("messages", []):
        if msg.get("role") == "system":
            text = _content_to_text(msg.get("content"))
            if text:
                system_parts.append({"text": text})
            continue
        conv_role, blocks = _message_to_blocks(msg)
        if not blocks:
            continue
        # Converse requires strictly alternating roles; merge consecutive same-role.
        if messages and messages[-1]["role"] == conv_role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": conv_role, "content": blocks})

    thinking_mode, effort, max_tokens, allow_temp, allow_samp = _resolve_inference(body, defaults)
    thinking_on = thinking_mode in ("adaptive", "always_on")

    inference: dict[str, Any] = {"maxTokens": max_tokens}
    # Temperature/top_p: honoured only if the model allows them AND thinking is
    # off (Anthropic rejects sampling params while extended thinking is active).
    if body.get("temperature") is not None and allow_temp and not thinking_on:
        inference["temperature"] = float(body["temperature"])
    if body.get("top_p") is not None and allow_samp and not thinking_on:
        inference["topP"] = float(body["top_p"])
    stop = body.get("stop")
    if stop:
        inference["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)

    kwargs: dict[str, Any] = {"messages": messages, "inferenceConfig": inference}
    if system_parts:
        kwargs["system"] = system_parts
    tool_cfg = _tools_to_converse(body)
    if tool_cfg:
        kwargs["toolConfig"] = tool_cfg

    # Extended thinking + effort ride in additionalModelRequestFields (verified
    # accepted on Converse for opus-4-8 / sonnet-5, 2026-07-24). "off"/"native"
    # inject nothing so the model's own default behaviour stands.
    amrf: dict[str, Any] = {}
    if thinking_mode == "adaptive":
        amrf["thinking"] = {"type": "adaptive"}
    if thinking_on and effort:
        amrf["output_config"] = {"effort": effort}
    if amrf:
        kwargs["additionalModelRequestFields"] = amrf
    if prompt_cache:
        info = _add_cache_points(kwargs, cache_key)
        # Hashes only, never prompt text: shows whether a client's tools/system
        # prefix is stable across turns, which decides whether caching can pay.
        _cache_log.info("prompt-cache model=%s tools=%s system=%s messages=%d history=%s placed=%s",
                    cache_key, info["tools"], info["system"], info["n_messages"],
                    info["history"], ",".join(info["placed"]) or "none")
    return kwargs


def _extract_content(content_blocks: list[dict]) -> tuple[str, str, list[dict]]:
    """Return (text, reasoning_text, tool_calls) from a Converse content list.

    tool_calls are already in OpenAI shape: Converse `input` (an object) is
    re-serialized to the JSON *string* OpenAI clients expect in `arguments`.
    """
    text_parts, reasoning_parts, tool_calls = [], [], []
    for block in content_blocks or []:
        if "text" in block:
            text_parts.append(block["text"])
        elif "reasoningContent" in block:
            rc = block["reasoningContent"]
            rt = (rc.get("reasoningText") or {}).get("text") if isinstance(rc, dict) else None
            if rt:
                reasoning_parts.append(rt)
        elif "toolUse" in block:
            tu = block["toolUse"] or {}
            tool_calls.append(
                {
                    "id": tu.get("toolUseId") or "",
                    "type": "function",
                    "function": {
                        "name": tu.get("name") or "",
                        "arguments": json.dumps(tu.get("input") or {}),
                    },
                }
            )
    return "".join(text_parts), "".join(reasoning_parts), tool_calls


def _usage(u: dict | None) -> dict:
    u = u or {}
    return {
        "prompt_tokens": u.get("inputTokens", 0),
        "completion_tokens": u.get("outputTokens", 0),
        "total_tokens": u.get("totalTokens", 0),
        # Passed through so callers (e.g. the sleep-cycle context-nap) can see
        # the true prompt size including cached tokens.
        "cache_read_input_tokens": u.get("cacheReadInputTokens", 0),
        "cache_write_input_tokens": u.get("cacheWriteInputTokens", 0),
    }


def converse_to_openai(resp: dict, public_id: str, completion_id: str, created: int) -> dict:
    """Bedrock converse() response -> OpenAI chat.completion object."""
    message = (resp.get("output") or {}).get("message") or {}
    text, reasoning, tool_calls = _extract_content(message.get("content") or [])
    msg: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if tool_calls:
        msg["tool_calls"] = tool_calls
        # OpenAI convention: content is null on a pure tool-call turn.
        if not text:
            msg["content"] = None
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": public_id,
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": _FINISH_REASON.get(resp.get("stopReason", ""), "stop"),
            }
        ],
        "usage": _usage(resp.get("usage")),
    }


async def complete(
    model_id: str, body: dict, region: str, public_id: str,
    defaults: "InferenceDefaults | None" = None,
) -> dict:
    """Non-streaming completion. boto3 is sync, so run it off the event loop."""
    kwargs = openai_to_converse(body, defaults, prompt_cache=supports_prompt_cache(model_id), cache_key=model_id)
    try:
        resp = await asyncio.to_thread(_client(region).converse, modelId=model_id, **kwargs)
    except Exception:
        logger.exception(
            "bedrock complete() failed model_id=%s public_id=%s region=%s",
            model_id, public_id, region,
        )
        raise
    return converse_to_openai(
        resp, public_id, f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
    )


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


async def stream(
    model_id: str, body: dict, region: str, public_id: str,
    defaults: "InferenceDefaults | None" = None,
) -> AsyncIterator[bytes]:
    """Streaming completion as OpenAI SSE chunks.

    boto3's converse_stream returns a synchronous EventStream; iterate it in a
    worker thread and hand chunks to the async generator via a queue so the
    FastAPI event loop is never blocked.
    """
    kwargs = openai_to_converse(body, defaults, prompt_cache=supports_prompt_cache(model_id), cache_key=model_id)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    def base(delta: dict, finish: str | None = None) -> dict:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": public_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def worker() -> None:
        try:
            resp = _client(region).converse_stream(modelId=model_id, **kwargs)
            for event in resp["stream"]:
                loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
        except Exception as exc:  # surface upstream errors into the stream
            # Previously this was swallowed with no server-side trace at all —
            # a throttled/erroring call would show as a clean "200 OK" in the
            # access log with an empty reply on the client end. Log it here so
            # a repeat is diagnosable from `docker logs` instead of guesswork.
            logger.exception(
                "bedrock stream() failed model_id=%s public_id=%s region=%s",
                model_id, public_id, region,
            )
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    loop.run_in_executor(None, worker)

    # Open with the assistant role so clients see a well-formed first chunk.
    yield _sse(base({"role": "assistant"}))
    finish_reason = "stop"
    usage: dict | None = None
    # Converse indexes content blocks (text and tool calls share one sequence);
    # OpenAI indexes tool_calls on their own. Map block index -> tool_call index.
    tool_index_by_block: dict[int, int] = {}
    next_tool_index = 0
    while True:
        kind, payload = await queue.get()
        if kind == "done":
            break
        if kind == "error":
            # Emit a terminal error frame, then close cleanly.
            yield _sse(base({}, "stop") | {"error": str(payload)})
            break
        event = payload
        if "contentBlockStart" in event:
            # A tool call opens here: id + name arrive now, arguments stream as
            # toolUse.input deltas on the matching contentBlockIndex.
            start = event["contentBlockStart"].get("start") or {}
            tu = start.get("toolUse")
            if tu:
                block_idx = event["contentBlockStart"].get("contentBlockIndex")
                tool_index_by_block[block_idx] = next_tool_index
                yield _sse(base({"tool_calls": [{
                    "index": next_tool_index,
                    "id": tu.get("toolUseId") or "",
                    "type": "function",
                    "function": {"name": tu.get("name") or "", "arguments": ""},
                }]}))
                next_tool_index += 1
        elif "contentBlockDelta" in event:
            cbd = event["contentBlockDelta"]
            delta = cbd.get("delta", {})
            if "text" in delta:
                yield _sse(base({"content": delta["text"]}))
            elif "toolUse" in delta:
                # Partial JSON string fragment for the open tool call.
                tool_idx = tool_index_by_block.get(cbd.get("contentBlockIndex"), 0)
                yield _sse(base({"tool_calls": [{
                    "index": tool_idx,
                    "function": {"arguments": (delta["toolUse"] or {}).get("input", "")},
                }]}))
            elif "reasoningContent" in delta:
                rt = (delta["reasoningContent"] or {}).get("text")
                if rt:
                    yield _sse(base({"reasoning_content": rt}))
        elif "messageStop" in event:
            finish_reason = _FINISH_REASON.get(
                event["messageStop"].get("stopReason", ""), "stop"
            )
        elif "metadata" in event and include_usage:
            usage = _usage(event["metadata"].get("usage"))

    final = base({}, finish_reason)
    if include_usage and usage is not None:
        final["usage"] = usage
    yield _sse(final)
    yield b"data: [DONE]\n\n"
