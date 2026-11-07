"""Probe which catalog models can run the client-tool loop of the browser stream.

A model passes when both rounds of the loop work through the same code path as
POST /v1/chat/stream (browser_stream.upstream_chunks + Translator):

  round 1  the request has one tool; the stream gives a complete tool call with
           the right name and valid JSON arguments, and ends with tool_calls.
  round 2  the request adds the assistant tool call and a role "tool" result;
           the stream gives text and does not repeat the same call (a repeat is
           a loop: the interface would call the tool again and again).

The result decides `client_tools=True` in catalog.py (supports_client_tools on
GET /v1/models). This script only reads and reports; it never edits catalog.py.

Every round is a real, billed upstream call (two short calls per model). Auth is
what the gateway uses: the AWS default credential chain for the bedrock and
bedrock_mantle backends, OASIS_GENERATION_UPSTREAM_BASE_URL for runner models, and
the api_key_env of each openai_compat entry. Run it with the credentials of the
gateway (dev-bedrock-invoker), not with an admin profile.

    python scripts/probe_client_tools.py                     # all enabled models
    python scripts/probe_client_tools.py --models claude-haiku-5-5,nova-micro
    python scripts/probe_client_tools.py --json probe.json   # also write the results
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Callable

from oasis_generation import browser_stream
from oasis_generation.catalog import CatalogEntry, enabled_models
from oasis_generation.config import Settings, get_settings

TOOL_NAME = "record_choice"
TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Record the colour that the user chose.",
        "parameters": {
            "type": "object",
            "properties": {"colour": {"type": "string", "enum": ["red", "blue"]}},
            "required": ["colour"],
        },
    },
}
PROMPT = (
    "This is an automated test of tool calling. Call the tool record_choice with the "
    "colour blue. Do not write any text before the call. After the tool result, reply "
    "with one short sentence that confirms the colour."
)
TOOL_RESULT = '{"recorded": true}'
MAX_TOKENS = 2048     # adaptive thinking shares this ceiling; the "fast" profile keeps it short
TIMEOUT_S = 90.0

SourceFactory = Callable[[CatalogEntry, dict, Settings], AsyncIterator[dict]]


@dataclass
class Round:
    ok: bool = False
    finish: str | None = None
    text_chars: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None
    seconds: float = 0.0


@dataclass
class Result:
    model: str
    backend: str
    passed: bool
    reason: str
    round1: Round
    round2: Round | None


def _body(entry: CatalogEntry, messages: list[dict]) -> dict:
    return {
        "model": entry.public_id,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [TOOL],
        "max_tokens": MAX_TOKENS,
        "profile": "fast",
    }


async def _run(entry: CatalogEntry, body: dict, settings: Settings, source: SourceFactory) -> Round:
    """One upstream call through the browser-stream translation; never raises."""
    r = Round()
    tr = browser_stream.Translator()
    t0 = time.monotonic()

    async def consume() -> None:
        src = browser_stream._guarded(source(entry, body, settings))
        try:
            async for chunk in src:
                for event, data in tr.on_chunk(chunk):
                    if event == "text":
                        r.text_chars += len(data["delta"])
                if tr.error:
                    break
        finally:
            await src.aclose()

    try:
        await asyncio.wait_for(consume(), TIMEOUT_S)
        if tr.error:
            raise browser_stream.UpstreamError(tr.error)
        r.finish = tr.done_reason()
        r.tool_calls = [{"id": t.id, "name": t.name, "arguments": t.args} for t in tr.tools.values()]
        u = browser_stream.split_usage(tr.usage)
        if u:
            r.tokens_in, r.tokens_out = u["prompt_total"], u["tokens_out"]
        r.ok = True
    except asyncio.TimeoutError:
        r.error = f"no complete answer in {TIMEOUT_S:.0f} s"
    except Exception as exc:  # noqa: BLE001 - every failure is a probe result, not a crash
        r.error = f"{type(exc).__name__}: {exc}"[:300]
    r.seconds = round(time.monotonic() - t0, 2)
    return r


def _valid_call(r: Round) -> str | None:
    """None when round 1 is a valid call of TOOL_NAME, else the reason it is not."""
    if r.finish != "tool_calls":
        return f"round 1 ended with {r.finish!r}, not 'tool_calls'"
    calls = [c for c in r.tool_calls if c["name"] == TOOL_NAME]
    if not calls:
        return f"round 1 called {[c['name'] for c in r.tool_calls]!r}, not {TOOL_NAME!r}"
    try:
        args = json.loads(calls[0]["arguments"] or "{}")
    except json.JSONDecodeError:
        return f"round 1 arguments are not JSON: {calls[0]['arguments'][:80]!r}"
    if not isinstance(args, dict) or args.get("colour") not in ("red", "blue"):
        return f"round 1 arguments do not match the schema: {args!r}"[:200]
    return None


def _same_args(a: str, b: str) -> bool:
    try:
        return json.loads(a or "{}") == json.loads(b or "{}")
    except json.JSONDecodeError:
        return a == b


async def probe(entry: CatalogEntry, settings: Settings,
                source: SourceFactory = browser_stream.upstream_chunks) -> Result:
    user = {"role": "user", "content": PROMPT}
    r1 = await _run(entry, _body(entry, [user]), settings, source)
    if not r1.ok:
        return Result(entry.public_id, entry.backend, False, f"round 1 failed: {r1.error}", r1, None)
    bad = _valid_call(r1)
    if bad:
        return Result(entry.public_id, entry.backend, False, bad, r1, None)
    call = next(c for c in r1.tool_calls if c["name"] == TOOL_NAME)
    messages = [
        user,
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call["id"], "type": "function",
             "function": {"name": TOOL_NAME, "arguments": call["arguments"] or "{}"}}]},
        {"role": "tool", "tool_call_id": call["id"], "content": TOOL_RESULT},
    ]
    r2 = await _run(entry, _body(entry, messages), settings, source)
    if not r2.ok:
        return Result(entry.public_id, entry.backend, False, f"round 2 failed: {r2.error}", r1, r2)
    if r2.finish == "length":
        return Result(entry.public_id, entry.backend, False, "round 2 hit the token limit", r1, r2)
    if any(c["name"] == TOOL_NAME and _same_args(c["arguments"], call["arguments"]) for c in r2.tool_calls):
        return Result(entry.public_id, entry.backend, False,
                      "round 2 repeated the same call after the tool result (a loop)", r1, r2)
    if r2.text_chars == 0:
        return Result(entry.public_id, entry.backend, False, "round 2 gave no text after the tool result", r1, r2)
    return Result(entry.public_id, entry.backend, True, "ok", r1, r2)


async def probe_all(entries: list[CatalogEntry], settings: Settings,
                    source: SourceFactory = browser_stream.upstream_chunks) -> list[Result]:
    # One model at a time: simple to read, and no burst on the account limits.
    return [await probe(e, settings, source) for e in entries]


def _select(names: str | None) -> list[CatalogEntry]:
    entries = [e for e in enabled_models() if e.backend in browser_stream.STREAM_BACKENDS]
    if not names:
        return entries
    want = [n.strip() for n in names.split(",") if n.strip()]
    by_id = {e.public_id: e for e in entries}
    missing = [n for n in want if n not in by_id]
    if missing:
        raise SystemExit(f"not enabled or not a stream backend: {', '.join(missing)}")
    return [by_id[n] for n in want]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", help="comma-separated public ids (default: all enabled models)")
    ap.add_argument("--json", dest="json_path", help="also write the results to this file")
    args = ap.parse_args(argv)

    entries = _select(args.models)
    print(f"Probing {len(entries)} model(s), two billed calls each.", file=sys.stderr)
    results = asyncio.run(probe_all(entries, get_settings()))

    width = max(len(r.model) for r in results) if results else 10
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"{mark}  {r.model:<{width}}  {r.backend:<14} {r.reason}")
    passed = [r.model for r in results if r.passed]
    print(f"\n{len(passed)} of {len(results)} passed. Set client_tools=True for: "
          f"{', '.join(passed) or '(none)'}")
    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
