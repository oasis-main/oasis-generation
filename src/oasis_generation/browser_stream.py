"""POST /v1/chat/stream: the browser stream contract (ORG-171).

docs/capability-token/BROWSER_STREAM_CONTRACT.md (861af9a) is the wire format.
This module holds everything behind the route except the HTTP wiring in main.py:

  - CapabilityRuntime   per-process state for capability callers: JWKS cache,
                        revocation / blocked-sub state, usage meter, MQ bridge,
                        and the in-flight client_request_id set.
  - authenticate()      bearer -> verified claims, or an ApiError with the
                        contract's status and code.
  - check_same_origin() the 403 cross_origin rule.
  - upstream sources    every backend as an async iterator of OpenAI
                        chat.completion.chunk dicts: Bedrock Converse
                        (bedrock.stream), bedrock-mantle Responses (mantle.stream,
                        one chunk: the backend cannot stream yet), and the
                        OpenAI-compatible upstreams (runner, openai_compat).
  - Translator          OpenAI chunks -> contract events (text, thinking,
                        tool_call, usage, done / error).
  - stream_session()    the event stream itself: seq, request_id, exactly one
                        terminal frame, upstream cancel on client disconnect, and
                        one usage event per upstream call (also stopped or failed).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import anyio
import httpx

from . import bedrock, mantle
from .capability import (AUD_GEN, AuthState, JwksCache, http_fetch_jwks, peek_kid, verify)
from .catalog import CatalogEntry
from .config import Settings
from .metering import MqttBridge, UsageMeter, build_usage_event, unique_client_id

logger = logging.getLogger(__name__)

# Tests swap this for an httpx.MockTransport; production leaves it None.
_upstream_transport: httpx.AsyncBaseTransport | None = None

STREAM_BACKENDS = ("bedrock", "bedrock_mantle", "runner", "openai_compat")
# public catalog backend -> (usage-event backend, served_by)
_BACKEND_LABELS = {
    "bedrock": ("bedrock", "bedrock"),
    "bedrock_mantle": ("bedrock_mantle", "bedrock"),
    "runner": ("runner", "own"),
    "openai_compat": ("openai_compat", "own"),
}
MAX_CLIENT_REQUEST_ID = 128
TOOL_OUTPUT_CAP = 16_000


class ApiError(Exception):
    """A pre-stream error: plain JSON {"error": {"code","message"}} (+ extra)."""

    def __init__(self, status: int, code: str, message: str, extra: dict | None = None,
                 headers: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.extra = extra or {}
        self.headers = headers or {}

    def body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, **self.extra}}


# ---- per-process runtime ---------------------------------------------------------

class CapabilityRuntime:
    def __init__(self, settings: Settings, *, fetch=None, publisher=None,
                 clock: Callable[[], float] = time.time):
        self.issuer = settings.capability_issuer
        self.clock = clock
        self.jwks = JwksCache(settings.capability_jwks_url, fetch=fetch or http_fetch_jwks)
        self.state = AuthState(clock=clock)
        self.meter = UsageMeter(publisher, settings.usage_spool_dir, settings.usage_spool_max,
                                clock=clock)
        self.bridge: MqttBridge | None = None
        self.stale_after_s = settings.upstream_timeout_s + 60
        self._inflight: dict[tuple[str, str], float] = {}

    def claim(self, sub: str, crid: str) -> bool:
        """Record (sub, client_request_id) as running; False if it already runs.

        An entry older than the upstream timeout is treated as gone: a stream
        whose generator never started cannot then block that id for ever."""
        key, now = (sub, crid), time.monotonic()
        started = self._inflight.get(key)
        if started is not None and now - started < self.stale_after_s:
            return False
        self._inflight[key] = now
        return True

    def release(self, sub: str, crid: str) -> None:
        self._inflight.pop((sub, crid), None)

    def start_bridge(self, settings: Settings) -> None:
        if not settings.mqtt_host:
            logger.warning("capability auth is on but MQTT_HOST is empty: usage events are "
                           "spooled only, and metered requests get 503 after 10 minutes")
            return
        self.bridge = MqttBridge(
            host=settings.mqtt_host, port=settings.mqtt_port, username=settings.mqtt_username,
            password=settings.mqtt_service_secret.get_secret_value(),
            client_id=unique_client_id(settings.mqtt_client_id), state=self.state)
        self.meter.publisher = self.bridge
        self.bridge.start()


_runtime: CapabilityRuntime | None = None


def runtime(settings: Settings) -> CapabilityRuntime:
    global _runtime
    if _runtime is None:
        _runtime = CapabilityRuntime(settings)
    return _runtime


def set_runtime(rt: CapabilityRuntime | None) -> None:
    global _runtime
    _runtime = rt


# ---- auth, origin, decisions -------------------------------------------------------

def bearer(request) -> str:
    auth = request.headers.get("authorization", "")
    return auth[7:].strip() if auth[:7].lower() == "bearer " else ""


async def authenticate(token: str, rt: CapabilityRuntime, scope: str | None) -> dict:
    """Verify a capability token. `scope=None` checks everything except the
    scope, so a metered route can answer 402/403 for billing before it checks
    the scope (a free token carries only gen:models; ORG-171)."""
    if not token:
        raise ApiError(401, "invalid_token", "missing bearer token")
    jwks = await rt.jwks.get(peek_kid(token))
    r = verify(token, jwks, int(rt.clock()), issuer=rt.issuer, audience=AUD_GEN,
               required_scope=scope, revoked_jti=rt.state.revoked_jti(),
               not_before_by_sub=rt.state.not_before_by_sub())
    if r.ok:
        return r.claims
    if r.reason == "expired":
        raise ApiError(401, "token_expired", "capability token expired")
    if r.reason == "missing_scope":
        raise ApiError(403, "missing_scope", f"capability token lacks scope {scope}")
    raise ApiError(401, "invalid_token", f"capability token refused: {r.reason}")


def require_scope(claims: dict, scope: str) -> None:
    """403 missing_scope when a verified token lacks `scope`."""
    if scope not in claims.get("scopes", []):
        raise ApiError(403, "missing_scope", f"capability token lacks scope {scope}")


_LOOPBACK = ("localhost", "127.0.0.1", "::1")


def check_same_origin(request, settings: Settings) -> None:
    """Refuse a request whose Origin is present and is not this host (403 cross_origin).

    Allowed: no Origin header; an Origin equal to https://<Host header> (http
    only for a loopback host); an exact origin listed in
    OASIS_GENERATION_STREAM_ALLOWED_ORIGINS."""
    origin = request.headers.get("origin")
    if origin is None:
        return
    allowed = {o.strip().rstrip("/") for o in settings.stream_allowed_origins.split(",") if o.strip()}
    if origin.rstrip("/") in allowed:
        return
    parts = urlsplit(origin)
    host = (request.headers.get("host") or "").lower()
    if parts.netloc and parts.netloc.lower() == host and parts.path in ("", "/"):
        if parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in _LOOPBACK):
            return
    raise ApiError(403, "cross_origin", "requests from another origin are refused")


def seconds_to_next_month(now: float) -> int:
    d = _dt.datetime.fromtimestamp(now, _dt.timezone.utc)
    nxt = (d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
           + _dt.timedelta(days=32)).replace(day=1)
    return max(1, int(nxt.timestamp() - now))


# ---- request body ------------------------------------------------------------------

_ROLES = ("system", "user", "assistant", "tool")


def _normalize_tools(tools) -> list[dict]:
    """Accept OpenAI function tools and the flat {name, description, parameters} form."""
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ApiError(400, "bad_request", "tools must be a list")
    out = []
    for t in tools:
        if not isinstance(t, dict):
            raise ApiError(400, "bad_request", "each tool must be an object")
        fn = t.get("function") if t.get("type") == "function" else t
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise ApiError(400, "bad_request", "each tool needs a name")
        spec = {"name": fn["name"],
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}}}
        if isinstance(fn.get("description"), str):
            spec["description"] = fn["description"]
        out.append({"type": "function", "function": spec})
    return out


async def read_json_capped(request, max_bytes: int):
    """Read a JSON body of at most `max_bytes`, else 413 request_too_large.

    The Content-Length is checked first; a chunked body is read only up to the
    cap, never fully. The ALB WAF lets /v1/chat/* bodies through unbounded."""
    too_large = ApiError(413, "request_too_large", f"request body is larger than {max_bytes} bytes")
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                raise too_large
        except ValueError:
            raise ApiError(400, "bad_request", "bad Content-Length") from None
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > max_bytes:
            raise too_large
    try:
        return json.loads(bytes(buf))
    except ValueError:
        raise ApiError(400, "bad_request", "body must be JSON") from None


@dataclass
class StreamRequest:
    model: str
    messages: list[dict]
    tools: list[dict]
    client_request_id: str


def parse_body(body) -> StreamRequest:
    if not isinstance(body, dict):
        raise ApiError(400, "bad_request", "body must be a JSON object")
    model, messages, crid = body.get("model"), body.get("messages"), body.get("client_request_id")
    if not isinstance(model, str) or not model:
        raise ApiError(400, "bad_request", "model is required")
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "bad_request", "messages must be a non-empty list")
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") not in _ROLES:
            raise ApiError(400, "bad_request", f"each message needs a role in {list(_ROLES)}")
    if not isinstance(crid, str) or not crid or len(crid) > MAX_CLIENT_REQUEST_ID:
        raise ApiError(400, "bad_request", "client_request_id is required (a uuid)")
    return StreamRequest(model, messages, _normalize_tools(body.get("tools")), crid)


# ---- upstream sources: OpenAI chunk dicts ----------------------------------------------

class UpstreamError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class SSEParser:
    """Bytes -> the `data:` payload of each complete SSE event."""

    def __init__(self):
        self._buf = ""

    def feed(self, chunk: bytes) -> list[str]:
        self._buf += chunk.decode("utf-8", errors="replace").replace("\r\n", "\n")
        out = []
        while "\n\n" in self._buf:
            raw, self._buf = self._buf.split("\n\n", 1)
            data = [ln[5:].lstrip(" ") for ln in raw.split("\n") if ln.startswith("data:")]
            if data:
                out.append("\n".join(data))
        return out


async def _chunks_from_sse(byte_gen: AsyncIterator[bytes]) -> AsyncIterator[dict]:
    parser = SSEParser()
    try:
        async for b in byte_gen:
            for data in parser.feed(b):
                if data.strip() == "[DONE]":
                    return
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    finally:
        await byte_gen.aclose()


async def _openai_compatible_bytes(base_url: str, body: dict, headers: dict | None,
                                   timeout: float) -> AsyncIterator[bytes]:
    if "api.openai.com" in base_url and "max_tokens" in body:
        body = {**body, "max_completion_tokens": body["max_tokens"]}
        body.pop("max_tokens")
    url = f"{base_url.rstrip('/')}/chat/completions"
    client = httpx.AsyncClient(timeout=timeout, transport=_upstream_transport)
    try:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                text = (await resp.aread())[:300]
                logger.warning("browser stream upstream HTTP %s: %r", resp.status_code, text)
                raise UpstreamError(f"upstream HTTP {resp.status_code}", status=resp.status_code)
            async for b in resp.aiter_bytes():
                yield b
    except httpx.HTTPError as exc:
        raise UpstreamError(f"upstream unreachable: {type(exc).__name__}") from exc
    finally:
        await client.aclose()


def upstream_chunks(entry: CatalogEntry, body: dict, settings: Settings) -> AsyncIterator[dict]:
    """Open the upstream call for `entry` and return its OpenAI chunk dicts."""
    if entry.backend == "bedrock":
        return _chunks_from_sse(bedrock.stream(
            entry.upstream_id, body, settings.bedrock_region, entry.public_id, entry.inference))
    if entry.backend == "bedrock_mantle":
        return _chunks_from_sse(mantle.stream(
            entry.upstream_id, body, settings.bedrock_region, entry.public_id, entry.inference))
    up = {**body, "model": entry.upstream_id}
    if entry.backend == "openai_compat":
        import os

        key = os.environ.get(entry.api_key_env or "", "").strip() if entry.api_key_env else ""
        if not entry.base_url or (entry.api_key_env and not key):
            return _fail(UpstreamError(f"{entry.public_id}: provider not configured on the gateway"))
        return _chunks_from_sse(_openai_compatible_bytes(
            entry.base_url, up, {"Authorization": f"Bearer {key}"} if key else None,
            settings.upstream_timeout_s))
    return _chunks_from_sse(_openai_compatible_bytes(
        settings.upstream_base_url, up, None, settings.upstream_timeout_s))


async def _fail(exc: Exception) -> AsyncIterator[dict]:
    raise exc
    yield {}  # pragma: no cover - makes this an async generator


# ---- translation -------------------------------------------------------------------------

_FINISH = {"stop": "stop", "length": "length", "tool_calls": "tool_calls",
           # The contract has no content-filter reason; the turn ends normally.
           "content_filter": "stop"}


def split_usage(u: dict | None) -> dict | None:
    """OpenAI-shaped usage -> tokens_in (uncached input), tokens_out, cache read/write."""
    if not isinstance(u, dict):
        return None

    def num(v):
        return v if isinstance(v, int) and v >= 0 else 0

    prompt = num(u.get("prompt_tokens", u.get("input_tokens")))
    out = num(u.get("completion_tokens", u.get("output_tokens")))
    read = u.get("cache_read_input_tokens")
    if read is None:
        read = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    read = num(read)
    write = num(u.get("cache_write_input_tokens"))
    return {"prompt_total": prompt, "tokens_in": max(0, prompt - read - write), "tokens_out": out,
            "cache_read_tokens": read, "cache_write_tokens": write}


@dataclass
class _Tool:
    id: str
    name: str
    args: str = ""


@dataclass
class Translator:
    """OpenAI chat.completion.chunk dicts -> (event, data) pairs of the contract."""

    finish: str | None = None
    usage: dict | None = None
    error: str | None = None
    chars_out: int = 0
    tools: dict[int, _Tool] = field(default_factory=dict)
    open_tool: int | None = None

    def _close_tool(self, idx: int | None) -> list[tuple[str, dict]]:
        if idx is None or idx not in self.tools:
            return []
        t = self.tools[idx]
        self.open_tool = None if self.open_tool == idx else self.open_tool
        return [("tool_call", {"id": t.id, "name": t.name, "arguments": t.args, "partial": False})]

    def on_chunk(self, chunk: dict) -> list[tuple[str, dict]]:
        out: list[tuple[str, dict]] = []
        if chunk.get("error"):
            self.error = str(chunk["error"])
            return out
        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if isinstance(text, str) and text:
                self.chars_out += len(text)
                out.append(("text", {"delta": text}))
            think = delta.get("reasoning_content")
            if think is None:
                think = delta.get("reasoning")
            if isinstance(think, str) and think:
                self.chars_out += len(think)
                out.append(("thinking", {"delta": think, "redacted": False}))
            for tc in delta.get("tool_calls") or []:
                out.extend(self._tool_fragment(tc))
            if choice.get("finish_reason"):
                self.finish = choice["finish_reason"]
        return out

    def _tool_fragment(self, tc: dict) -> list[tuple[str, dict]]:
        out: list[tuple[str, dict]] = []
        idx = tc.get("index", 0)
        fn = tc.get("function") or {}
        frag = fn.get("arguments") or ""
        if not isinstance(frag, str):
            frag = json.dumps(frag)
        if idx not in self.tools:
            if self.open_tool is not None:
                out.extend(self._close_tool(self.open_tool))
            self.tools[idx] = _Tool(id=tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                                    name=fn.get("name") or "")
            self.open_tool = idx
        t = self.tools[idx]
        if not t.name and fn.get("name"):
            t.name = fn["name"]
        t.args += frag
        self.chars_out += len(frag)
        out.append(("tool_call", {"id": t.id, "name": t.name, "arguments": frag, "partial": True}))
        return out

    def finish_events(self) -> list[tuple[str, dict]]:
        """Close open tool calls; return them. The caller then sends usage + done."""
        out = []
        for idx in list(self.tools):
            if idx == self.open_tool:
                out.extend(self._close_tool(idx))
        return out

    def done_reason(self) -> str:
        reason = _FINISH.get(self.finish or "stop", "stop")
        # Every tool in a browser request is a client tool: a turn that called
        # one ends with tool_calls so the UI runs it.
        if self.tools and reason != "length":
            return "tool_calls"
        return reason


def classify_error(message: str, status: int | None = None) -> tuple[str, bool]:
    """(contract error code, retryable) for an upstream failure."""
    text = message.lower()
    if status == 429 or "throttl" in text or "too many requests" in text or " 429" in text:
        return "rate_limited", True
    if status is not None and 400 <= status < 500:
        return "upstream_error", False
    if "validationexception" in text or "accessdenied" in text:
        return "upstream_error", False
    return "upstream_error", True


async def _guarded(src: AsyncIterator[dict]) -> AsyncIterator[dict]:
    """Every failure inside the upstream source becomes an UpstreamError."""
    try:
        async for chunk in src:
            yield chunk
    except UpstreamError:
        raise
    except Exception as exc:  # noqa: BLE001 - boto / signing / network errors
        logger.warning("browser stream upstream failed: %s: %s", type(exc).__name__, exc)
        raise UpstreamError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        with anyio.CancelScope(shield=True):
            await src.aclose()


def _frame(event: str, seq: int, request_id: str, data: dict) -> bytes:
    payload = {"seq": seq, "request_id": request_id, **data}
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _estimate_tokens(chars: int) -> int:
    return (chars + 3) // 4


async def stream_session(
    *, rt: CapabilityRuntime, settings: Settings, entry: CatalogEntry, req: StreamRequest,
    claims: dict, request_id: str,
    source_factory: Callable[[CatalogEntry, dict, Settings], AsyncIterator[dict]] = upstream_chunks,
) -> AsyncIterator[bytes]:
    """The SSE body. Exactly one of done / error ends it; usage is metered in `finally`."""
    backend, served_by = _BACKEND_LABELS[entry.backend]
    body: dict[str, Any] = {"model": entry.public_id, "messages": req.messages, "stream": True,
                            "stream_options": {"include_usage": True}}
    if req.tools:
        body["tools"] = req.tools
    tr = Translator()
    seq = 0
    status = "error"
    called = False

    def frame(event: str, **data) -> bytes:
        nonlocal seq
        b = _frame(event, seq, request_id, data)
        seq += 1
        return b

    try:
        yield frame("start", model=entry.public_id, served_by=served_by)
        called = True
        src = _guarded(source_factory(entry, body, settings))
        try:
            async for chunk in src:
                for event, data in tr.on_chunk(chunk):
                    yield frame(event, **data)
                if tr.error:
                    break
        finally:
            with anyio.CancelScope(shield=True):
                await src.aclose()
        if tr.error:
            raise UpstreamError(tr.error)
        for event, data in tr.finish_events():
            yield frame(event, **data)
        u = split_usage(tr.usage)
        yield frame("usage", tokens_in=u["prompt_total"] if u else None,
                    tokens_out=u["tokens_out"] if u else None, cost_estimate=None)
        status = "ok"
        yield frame("done", finish_reason=tr.done_reason())
    except UpstreamError as exc:
        status = "error"
        code, retryable = classify_error(str(exc), exc.status)
        msg = ("the model provider is rate limiting; retry later" if code == "rate_limited"
               else "the model provider returned an error")
        yield frame("error", code=code, message=msg, retryable=retryable)
    except (GeneratorExit, asyncio.CancelledError):
        status = "stopped"
        raise
    except Exception:  # noqa: BLE001 - a bug here must still end the stream cleanly
        logger.exception("browser stream internal error request_id=%s", request_id)
        status = "error"
        yield frame("error", code="internal", message="internal gateway error", retryable=True)
    finally:
        rt.release(claims["sub"], req.client_request_id)
        if called:
            _meter(rt, entry, req, claims, request_id, backend, served_by, tr, status)


def _meter(rt: CapabilityRuntime, entry: CatalogEntry, req: StreamRequest, claims: dict,
           request_id: str, backend: str, served_by: str, tr: Translator, status: str) -> None:
    """One usage event per upstream call. Never raises (the meter spools on failure)."""
    try:
        u = split_usage(tr.usage)
        extra = {}
        if u is None:
            # The upstream never sent its usage block (stopped or failed call).
            # Bill an estimate (4 characters per token) rather than nothing, and
            # say so. Input is counted only when the upstream got the prompt
            # and answered or was stopped.
            used = tr.chars_out > 0 or status == "stopped"
            prompt_chars = len(json.dumps(req.messages)) if used else 0
            u = {"tokens_in": _estimate_tokens(prompt_chars),
                 "tokens_out": _estimate_tokens(tr.chars_out),
                 "cache_read_tokens": 0, "cache_write_tokens": 0}
            extra["usage_estimated"] = True
        rt.meter.emit(build_usage_event(
            sub=claims["sub"], jti=claims["jti"], allowance_ref=claims.get("allowance_ref") or "",
            request_id=request_id, client_request_id=req.client_request_id,
            model=entry.public_id, backend=backend, served_by=served_by,
            tokens_in=u["tokens_in"], tokens_out=u["tokens_out"],
            cache_read_tokens=u["cache_read_tokens"], cache_write_tokens=u["cache_write_tokens"],
            status=status, streamed=True, **extra))
    except Exception:  # noqa: BLE001
        logger.exception("usage event build failed request_id=%s", request_id)


KEEPALIVE_S = 15.0
KEEPALIVE_FRAME = b": ping\n\n"


async def with_keepalive(gen: AsyncIterator[bytes], interval: float = KEEPALIVE_S) -> AsyncIterator[bytes]:
    """Relay `gen`; send an SSE comment line when nothing was written for `interval` s.

    The dev ALB closes a connection that is idle for 60 s, and a long think can
    send nothing for longer. A comment line has no event and no seq; the
    contract says clients ignore it. Closing this generator (client disconnect)
    cancels the pending read, so `gen` sees the cancel and stops its upstream.
    """
    pending: asyncio.Future | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(gen.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                yield KEEPALIVE_FRAME
                continue
            task, pending = pending, None
            try:
                chunk = task.result()
            except StopAsyncIteration:
                return
            yield chunk
    finally:
        with anyio.CancelScope(shield=True):
            if pending is not None and not pending.done():
                pending.cancel()
                try:
                    await pending
                except (asyncio.CancelledError, StopAsyncIteration, Exception):  # noqa: BLE001
                    pass
            await gen.aclose()


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex}"


def cap_tool_output(text: str) -> str:
    """For a future server-side tool loop (tool_result frames): cap the output."""
    return text if len(text) <= TOOL_OUTPUT_CAP else text[:TOOL_OUTPUT_CAP] + "...[truncated]"
