"""BYOK registry (GEN-008): tenant-supplied model providers.

Entry:  {kind: openai_compat|anthropic|bedrock_role, base_url, key_ref, region?}

  key_ref  is the NAME of an environment variable (or a secret-store handle
           resolved by `key_resolver`). The key value never appears in the
           registry, in logs, or in an exception message.

Lifecycle:  register() -> status "pending"; run_conformance() -> "active" only if
every check passes, else "failed". Routes may only use ACTIVE providers
(`active_names()` feeds RouteTable.byok_known / byok_regions).

Conformance (all four must pass, against the real endpoint):
  1. plain completion returns text
  2. streamed completion yields at least one delta and terminates
  3. one forced tool call comes back as a well-formed tool call
  4. a usage block with token counts is present (plain call)

SSRF: EVERY outbound call goes through ssrf.pin(): https only, all resolved
addresses checked, connection pinned to the checked IP, redirects off. That
includes the conformance run. base_url is also checked at register time so a bad
entry is rejected immediately, but the per-call check is the one that counts
(DNS can change between the two).

bedrock_role has no base_url: it assumes a customer IAM role via STS. The network
part is delegated to an injected `bedrock_runner` (the gateway's bedrock.py) so
this module stays free of boto3.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from . import ssrf

Kind = Literal["openai_compat", "anthropic", "bedrock_role"]
Status = Literal["pending", "active", "failed"]

log = logging.getLogger("oasis_generation.byok")

ANTHROPIC_VERSION = "2023-06-01"
PROBE_TIMEOUT_S = 30.0          # httpx per-operation timeout (connect/read gap)
# S2: per-operation timeouts do not bound a slow-drip endpoint (1 byte / 29 s holds
# a connection forever). These are TOTAL budgets and hard size caps per call.
CALL_DEADLINE_S = 60.0          # whole plain call: connect + headers + full body
STREAM_DEADLINE_S = 300.0       # whole streamed call, wall clock
MAX_BODY_BYTES = 1 * 1024 * 1024
MAX_STREAM_BYTES = 8 * 1024 * 1024


class ByokError(RuntimeError):
    pass


class ByokEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Kind
    base_url: str | None = None
    key_ref: str | None = None
    region: str | None = None          # declared by the tenant; "eu" or "eu-*" counts as EU
    model: str | None = None           # model id used for the conformance probes
    role_arn: str | None = None        # bedrock_role only
    tenant: str | None = None          # owner (S1); set by the operator/admin, never trusted from a caller

    @field_validator("key_ref")
    @classmethod
    def _no_secret_in_ref(cls, v: str | None) -> str | None:
        # A key pasted into key_ref instead of a reference is the classic mistake.
        if v and (len(v) > 64 or v.startswith(("sk-", "sk_", "AKIA", "xoxb-", "ghp_"))):
            raise ValueError("key_ref must be a reference name, not a key value")
        return v


class _Body:
    """A fully-read, size-capped upstream body with the bits of httpx.Response we use."""

    def __init__(self, status_code: int, content: bytes, encoding: str = "utf-8") -> None:
        self.status_code, self.content, self._enc = status_code, content, encoding

    @property
    def text(self) -> str:
        return self.content.decode(self._enc, "replace")

    def json(self):
        return json.loads(self.text)


@dataclass
class Provider:
    name: str
    entry: ByokEntry
    status: Status = "pending"
    report: dict = field(default_factory=dict)

    @property
    def owner(self) -> str | None:
        return self.entry.tenant


def _env_resolver(ref: str) -> str | None:
    return os.environ.get(ref)


class ByokRegistry:
    def __init__(
        self,
        key_resolver: Callable[[str], str | None] = _env_resolver,
        transport: httpx.AsyncBaseTransport | None = None,   # tests inject a MockTransport
        bedrock_runner: Callable[[ByokEntry], Awaitable[dict]] | None = None,
        allowed_ports: frozenset[int] | set[int] = ssrf.DEFAULT_ALLOWED_PORTS,  # S3: operator-set
    ) -> None:
        self._ports = frozenset(allowed_ports)
        self._p: dict[str, Provider] = {}
        self._resolve_key = key_resolver
        self._transport = transport
        self._bedrock_runner = bedrock_runner

    # ---- registration -----------------------------------------------------
    def register(self, name: str, raw: dict) -> Provider:
        if not name or not name.replace("-", "").replace("_", "").isalnum():
            raise ByokError(f"bad provider name {name!r}")
        entry = ByokEntry.model_validate(raw)
        if entry.kind == "bedrock_role":
            if not entry.role_arn:
                raise ByokError("bedrock_role requires role_arn")
        else:
            if not entry.base_url:
                raise ByokError(f"{entry.kind} requires base_url")
            if not entry.key_ref:
                raise ByokError(f"{entry.kind} requires key_ref")
            if not entry.model:
                raise ByokError(f"{entry.kind} requires model (used for conformance probes)")
            try:
                ssrf.check_url(entry.base_url, self._ports)
            except ssrf.SSRFError as exc:
                raise ByokError(f"base_url refused: {exc}") from exc
        prov = Provider(name, entry)
        self._p[name] = prov
        return prov

    def get(self, name: str) -> Provider:
        try:
            return self._p[name]
        except KeyError:
            raise ByokError(f"unknown byok provider {name!r}") from None

    def check_owner(self, name: str, caller: str) -> None:
        """S1: a provider holds its owner's key; only that owner's traffic may use it.
        An entry with no tenant is operator-shared and open to any caller."""
        owner = self.get(name).owner
        if owner is not None and owner != "*" and owner != caller:
            log.warning("byok %s: caller %r is not owner %r", name, caller, owner)
            raise ByokError(f"byok provider {name!r} is not available to this caller")

    def active_names(self) -> set[str]:
        return {n for n, p in self._p.items() if p.status == "active"}

    def active_regions(self) -> dict[str, str | None]:
        return {n: p.entry.region for n, p in self._p.items() if p.status == "active"}

    # ---- the guarded call -------------------------------------------------
    def _headers(self, entry: ByokEntry) -> dict[str, str]:
        key = self._resolve_key(entry.key_ref or "")
        if not key:
            raise ByokError(f"key_ref {entry.key_ref!r} did not resolve")  # name only, never value
        if entry.kind == "anthropic":
            return {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
                    "content-type": "application/json"}
        return {"Authorization": f"Bearer {key}", "content-type": "application/json"}

    def _path(self, entry: ByokEntry) -> str:
        return "/messages" if entry.kind == "anthropic" else "/chat/completions"

    def _client(self) -> httpx.AsyncClient:
        # trust_env off: no proxy env var may redirect a BYOK call somewhere unchecked.
        return httpx.AsyncClient(timeout=PROBE_TIMEOUT_S, transport=self._transport,
                                 trust_env=False, follow_redirects=False)

    def prepare(self, entry: ByokEntry) -> tuple[dict, str]:
        """SSRF-check + pin, return (httpx request kwargs, original url). Called
        immediately before EVERY outbound request, never cached."""
        base = (entry.base_url or "").rstrip("/") + self._path(entry)
        try:
            pinned = ssrf.pin(base, self._ports)
        except ssrf.SSRFError as exc:
            # S5: the resolved address / range must not reach a caller (it maps the
            # internal network). Detail goes to the log only.
            log.warning("byok call refused for %s: %s", entry.base_url, exc)
            raise ByokError("call refused: destination not allowed") from exc
        args = pinned.request_args()
        args["headers"] = {**args["headers"], **self._headers(entry)}
        return args, base

    async def _post(self, entry: ByokEntry, body: dict, stream: bool = False):
        """One guarded POST under a TOTAL deadline and a byte cap (S2).
        Returns (status, _Body) or, for stream=True, (status, list[str] lines)."""
        try:
            return await asyncio.wait_for(self._post_inner(entry, body, stream), CALL_DEADLINE_S)
        except asyncio.TimeoutError:
            raise ByokError("upstream did not finish within the call deadline") from None

    async def _post_inner(self, entry: ByokEntry, body: dict, stream: bool):
        args, _ = self.prepare(entry)
        url = args.pop("url")
        args.pop("follow_redirects", None)
        client = self._client()
        try:
            req = client.build_request("POST", url, json=body, **args)
            resp = await client.send(req, stream=True)
            try:
                if 300 <= resp.status_code < 400:
                    raise ByokError(f"upstream redirected ({resp.status_code}); redirects are not followed")
                cap = MAX_STREAM_BYTES if stream else MAX_BODY_BYTES
                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf += chunk
                    if len(buf) > cap:
                        raise ByokError(f"upstream body exceeds {cap} bytes")
                    if stream and buf.count(b"\n") > 2000:
                        break
                if stream:
                    return resp.status_code, buf.decode("utf-8", "replace").splitlines()
                return resp.status_code, _Body(resp.status_code, bytes(buf), resp.encoding or "utf-8")
            finally:
                await resp.aclose()
        finally:
            await client.aclose()

    # ---- forwarding (real traffic) ---------------------------------------
    def forward_ready(self, name: str) -> ByokEntry:
        prov = self.get(name)
        if prov.status != "active":
            raise ByokError(f"byok provider {name!r} is {prov.status}, not active")
        if prov.entry.kind != "openai_compat":
            # anthropic / bedrock_role need request translation that is not built yet.
            raise ByokError(f"forwarding for kind {prov.entry.kind!r} is not implemented")
        return prov.entry

    async def forward(self, name: str, body: dict) -> tuple[int, dict | str]:
        """Non-streaming chat completion through an ACTIVE openai_compat provider.
        SSRF-checked and IP-pinned on every call."""
        entry = self.forward_ready(name)
        body = {**body, "model": entry.model or body.get("model")}
        code, resp = await self._post(entry, body)
        try:
            return code, resp.json()
        except ValueError:
            return code, resp.text[:200]

    async def forward_stream(self, name: str, body: dict):
        """Async generator of raw SSE bytes. The SSRF check happens before the
        first byte is requested. Wall-clock deadline and byte cap apply (S2)."""
        entry = self.forward_ready(name)
        body = {**body, "model": entry.model or body.get("model"), "stream": True}
        args, _ = self.prepare(entry)
        url = args.pop("url")
        args.pop("follow_redirects", None)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + STREAM_DEADLINE_S

        def left() -> float:
            r = deadline - loop.time()
            if r <= 0:
                raise ByokError("upstream stream exceeded its deadline")
            return r

        client = self._client()
        try:
            req = client.build_request("POST", url, json=body, **args)
            try:
                resp = await asyncio.wait_for(client.send(req, stream=True), left())
            except asyncio.TimeoutError:
                raise ByokError("upstream stream exceeded its deadline") from None
            try:
                if 300 <= resp.status_code < 400:
                    raise ByokError(f"upstream redirected ({resp.status_code}); not followed")
                it = resp.aiter_bytes().__aiter__()
                total = 0
                while True:
                    try:
                        chunk = await asyncio.wait_for(it.__anext__(), left())
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        raise ByokError("upstream stream exceeded its deadline") from None
                    total += len(chunk)
                    if total > MAX_STREAM_BYTES:
                        raise ByokError("upstream stream exceeded its size cap")
                    yield chunk
            finally:
                await resp.aclose()
        finally:
            await client.aclose()

    # ---- conformance ------------------------------------------------------
    def _body(self, entry: ByokEntry, *, stream=False, tool=False) -> dict:
        prompt = "Reply with the single word: ok"
        if tool:
            prompt = "What is the weather in Paris? Use the tool."
        if entry.kind == "anthropic":
            b: dict = {"model": entry.model, "max_tokens": 64, "stream": stream,
                       "messages": [{"role": "user", "content": prompt}]}
            if tool:
                b["tools"] = [{"name": "get_weather", "description": "weather by city",
                               "input_schema": {"type": "object",
                                                "properties": {"city": {"type": "string"}},
                                                "required": ["city"]}}]
                b["tool_choice"] = {"type": "tool", "name": "get_weather"}
            return b
        b = {"model": entry.model, "max_tokens": 64, "stream": stream,
             "messages": [{"role": "user", "content": prompt}]}
        if stream:
            b["stream_options"] = {"include_usage": True}
        if tool:
            fn = {"name": "get_weather", "description": "weather by city",
                  "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                 "required": ["city"]}}
            b["tools"] = [{"type": "function", "function": fn}]
            b["tool_choice"] = {"type": "function", "function": {"name": "get_weather"}}
        return b

    @staticmethod
    def _text(kind: str, j: dict) -> str:
        if kind == "anthropic":
            return "".join(c.get("text", "") for c in j.get("content", []) if c.get("type") == "text")
        try:
            return j["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            return ""

    @staticmethod
    def _tool_ok(kind: str, j: dict) -> bool:
        if kind == "anthropic":
            for c in j.get("content", []):
                if c.get("type") == "tool_use" and isinstance(c.get("input"), dict) and "city" in c["input"]:
                    return True
            return False
        try:
            calls = j["choices"][0]["message"].get("tool_calls") or []
            if not calls:
                return False
            args = json.loads(calls[0]["function"]["arguments"])
            return calls[0]["function"]["name"] == "get_weather" and isinstance(args, dict) and "city" in args
        except (KeyError, IndexError, TypeError, ValueError):
            return False

    @staticmethod
    def _usage_ok(kind: str, j: dict) -> bool:
        u = j.get("usage")
        if not isinstance(u, dict):
            return False
        a, b = (("input_tokens", "output_tokens") if kind == "anthropic"
                else ("prompt_tokens", "completion_tokens"))
        return isinstance(u.get(a), int) and isinstance(u.get(b), int) and u[a] >= 0 and u[b] >= 0

    @staticmethod
    def _stream_ok(kind: str, lines: list[str]) -> bool:
        data = [l[5:].strip() for l in lines if l.startswith("data:")]
        if not data:
            return False
        if kind == "anthropic":
            types = []
            for d in data:
                try:
                    types.append(json.loads(d).get("type"))
                except ValueError:
                    pass
            return "content_block_delta" in types and "message_stop" in types
        deltas = 0
        for d in data:
            if d == "[DONE]":
                return deltas > 0
            try:
                ch = json.loads(d).get("choices") or []
            except ValueError:
                continue
            if ch and (ch[0].get("delta") or {}).get("content"):
                deltas += 1
        return False  # no [DONE] terminator

    async def run_conformance(self, name: str) -> dict:
        prov = self.get(name)
        e = prov.entry
        checks: dict[str, str] = {}   # check -> "ok" | reason (never contains key material)

        if e.kind == "bedrock_role":
            if self._bedrock_runner is None:
                checks["bedrock"] = "no bedrock runner configured"
            else:
                try:
                    res = await self._bedrock_runner(e)
                    for k in ("plain", "stream", "tool", "usage"):
                        checks[k] = "ok" if res.get(k) is True else f"failed: {res.get(k)!r}"
                except Exception as exc:  # noqa: BLE001
                    checks["bedrock"] = f"error: {type(exc).__name__}"
        else:
            async def guarded(label, coro):
                try:
                    return await coro
                except ByokError as exc:
                    checks[label] = f"refused: {exc}"
                except httpx.HTTPError as exc:
                    checks[label] = f"network: {type(exc).__name__}"
                return None

            r = await guarded("plain", self._post(e, self._body(e)))
            if r:
                code, resp = r
                try:
                    j = resp.json()
                except ValueError:
                    j = {}
                checks["plain"] = "ok" if code == 200 and self._text(e.kind, j).strip() else f"HTTP {code} / no text"
                checks["usage"] = "ok" if code == 200 and self._usage_ok(e.kind, j) else "no usage block"
            else:
                checks.setdefault("plain", "no response")
                checks.setdefault("usage", "no response")

            r = await guarded("stream", self._post(e, self._body(e, stream=True), stream=True))
            if r:
                code, lines = r
                checks["stream"] = "ok" if code == 200 and self._stream_ok(e.kind, lines) else f"HTTP {code} / bad stream"
            else:
                checks.setdefault("stream", "no response")

            r = await guarded("tool", self._post(e, self._body(e, tool=True)))
            if r:
                code, resp = r
                try:
                    j = resp.json()
                except ValueError:
                    j = {}
                checks["tool"] = "ok" if code == 200 and self._tool_ok(e.kind, j) else f"HTTP {code} / no valid tool call"
            else:
                checks.setdefault("tool", "no response")

        passed = bool(checks) and all(v == "ok" for v in checks.values()) and (
            e.kind == "bedrock_role" or set(checks) >= {"plain", "stream", "tool", "usage"})
        prov.status = "active" if passed else "failed"
        prov.report = checks
        return {"provider": name, "status": prov.status, "checks": checks}
