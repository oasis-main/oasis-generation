#!/usr/bin/env python3
"""oasis-decider: a typed-decision sidecar for the reviewer (CLAW-118, SHADOW).

The reviewer asks typed questions (choice / noul / score) about one tool call
and gets a probability for every option, from a small non-generative model
(Laya, ModernBERT-large, 421M). The model writes no text, so it cannot emit a
malformed verdict, and it answers in tens to hundreds of milliseconds on CPU.

SHADOW ONLY. Nothing in the fleet acts on these answers. The reviewer logs
them next to the Layer 2 judge's verdict so the two can be compared on real
traffic before any decision depends on this model. See
oasis-x/.swarm/GENERATIVE_PLAN.md §10.

API (stdlib HTTP, no framework, so the only third-party code in this
container is the pinned `laya` package and the model libraries):

  GET  /healthz       -> {"ok": true, "model": ..., "revision": ...}
  POST /v1/decide     {"state": <str | object>, "questions": {<id>: <question>}}
                      -> {"answers": {...}, "ms": <float>, "model": ..., "revision": ...}

A question is {"type": "choice"|"noul"|"score", "instructions": str,
"criteria": {label: description} (choice) | [level, ...] (score)}. This is
the request shape AnyJev and Laya use, so a later backend swap does not
change the reviewer.

Posture: no internet (HF_HUB_OFFLINE=1, weights baked into the image at a
pinned revision and verified by SHA-256 at build), read-only root, no
capabilities, uid 10001. One model, one inference at a time.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY_BYTES = 64 * 1024
MAX_QUESTIONS = 8
MAX_CHOICE_OPTIONS = 20  # Laya: options share a 256-token head budget
MAX_STATE_CHARS = 6000
# Calls in progress + waiting for the model lock. Above this the server says
# "busy" at once (503). Without it, a burst from several bots queued behind the
# single lock until the reviewer's timeout fired (2026-10-07 20:14-20:26 UTC:
# 39 of 78 shadow calls timed out at 8 s). A fast 503 is cheaper and is
# reported as "busy", not confused with a slow model.
DEFAULT_MAX_PENDING = 8
QUESTION_TYPES = ("choice", "noul", "score")


class BadRequest(ValueError):
    pass


def validate(body: object) -> tuple[object, dict]:
    """Return (state, questions) or raise BadRequest. Pure: unit-tested."""
    if not isinstance(body, dict):
        raise BadRequest("body must be a JSON object")
    state, questions = body.get("state"), body.get("questions")
    if isinstance(state, str):
        if not state.strip():
            raise BadRequest("state is empty")
        if len(state) > MAX_STATE_CHARS:
            raise BadRequest(f"state longer than {MAX_STATE_CHARS} chars")
    elif isinstance(state, dict):
        if len(json.dumps(state)) > MAX_STATE_CHARS:
            raise BadRequest(f"state longer than {MAX_STATE_CHARS} chars as JSON")
    else:
        raise BadRequest("state must be a string or an object")
    if not isinstance(questions, dict) or not questions:
        raise BadRequest("questions must be a non-empty object")
    if len(questions) > MAX_QUESTIONS:
        raise BadRequest(f"at most {MAX_QUESTIONS} questions")
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise BadRequest(f"question {qid!r} must be an object")
        qtype = q.get("type")
        if qtype not in QUESTION_TYPES:
            raise BadRequest(f"question {qid!r}: type must be one of {QUESTION_TYPES}")
        if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
            raise BadRequest(f"question {qid!r}: instructions required")
        crit = q.get("criteria")
        if qtype == "choice":
            if not isinstance(crit, dict) or not 2 <= len(crit) <= MAX_CHOICE_OPTIONS:
                raise BadRequest(f"question {qid!r}: choice needs 2..{MAX_CHOICE_OPTIONS} criteria")
        elif qtype == "score":
            if not isinstance(crit, list) or len(crit) < 2:
                raise BadRequest(f"question {qid!r}: score needs a list of at least 2 levels")
    return state, questions


class Decider:
    """Holds the one model. `predict` is serialized: torch on CPU gains nothing
    from concurrent forward passes here, and a lock bounds peak memory."""

    def __init__(self, model_dir: str, name: str, revision: str):
        import laya  # imported here so the unit tests need no torch

        self.name, self.revision = name, revision
        self._agent = laya.load(model_dir, device="cpu")
        self._lock = threading.Lock()

    def predict(self, state, questions) -> tuple[dict, dict | None]:
        with self._lock:
            return unwrap(self._agent.predict(state, questions))


def unwrap(out) -> tuple[dict, dict | None]:
    """laya returns {"model", "answers", "usage"}; callers want the answers.
    Accept a bare answers dict too, so a backend swap does not break this."""
    if isinstance(out, dict) and isinstance(out.get("answers"), dict):
        return out["answers"], out.get("usage")
    return out, None


class Admission:
    """Counts calls in progress or waiting. Thread-safe."""

    def __init__(self, limit: int):
        self.limit, self.pending = limit, 0
        self._lock = threading.Lock()

    def enter(self) -> bool:
        with self._lock:
            if self.pending >= self.limit:
                return False
            self.pending += 1
            return True

    def leave(self) -> None:
        with self._lock:
            self.pending -= 1


def make_handler(decider, max_pending: int = DEFAULT_MAX_PENDING):
    admission = Admission(max_pending)

    class Handler(BaseHTTPRequestHandler):
        server_version = "oasis-decider/1"

        def _send(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):  # one line per request, no bodies
            sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

        def do_GET(self):
            if self.path == "/healthz":
                self._send(200, {"ok": True, "model": decider.name, "revision": decider.revision})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/decide":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send(413 if length > MAX_BODY_BYTES else 400, {"error": "bad Content-Length"})
                return
            try:
                state, questions = validate(json.loads(self.rfile.read(length)))
            except (BadRequest, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})
                return
            if not admission.enter():
                self._send(503, {"error": f"busy: {admission.pending} calls pending"})
                return
            started = time.perf_counter()
            try:
                answers, usage = decider.predict(state, questions)
            except Exception as e:  # noqa: BLE001 - report, never crash the server
                self._send(500, {"error": f"{type(e).__name__}: {e}"[:300]})
                return
            finally:
                admission.leave()
            self._send(200, {
                "answers": answers,
                "usage": usage,
                "ms": round((time.perf_counter() - started) * 1000, 1),
                "model": decider.name,
                "revision": decider.revision,
            })

    return Handler


def main() -> None:
    model_dir = os.environ.get("DECIDER_MODEL_DIR", "/opt/decider/model")
    name = os.environ.get("DECIDER_MODEL_NAME", "unknown")
    revision = os.environ.get("DECIDER_MODEL_REVISION", "unknown")
    port = int(os.environ.get("DECIDER_PORT", "8790"))
    max_pending = int(os.environ.get("DECIDER_MAX_PENDING", str(DEFAULT_MAX_PENDING)))
    started = time.perf_counter()
    decider = Decider(model_dir, name, revision)
    print(f"oasis-decider: loaded {name}@{revision[:12]} in {time.perf_counter() - started:.1f}s; "
          f"listening on :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), make_handler(decider, max_pending)).serve_forever()


if __name__ == "__main__":
    main()
