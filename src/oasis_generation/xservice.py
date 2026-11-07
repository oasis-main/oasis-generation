"""X endpoints (GEN-008): /v1/x/decide and /v1/gate over an X backend.

Design: docs/OX-MODELS-INTERFACE-TRAINING-DESIGN.md sections 1-3. X holds authority,
O holds ideas. These rules are enforced HERE, in code, not left to callers:

  1. Closed label set. Every labelset carries an explicit out-of-scope option
     (`OUT_OF_SCOPE`). A decider that puts its mass there, or whose top
     probability is below the labelset's abstain floor, yields abstain=True.
  2. Restrictive-only. For /v1/gate, X may only TIGHTEN a baseline verdict
     (allow -> escalate -> deny). It can never loosen one. With no baseline the
     default baseline is "escalate", so a bare X "allow" is only honoured when it
     clears the allow floor (high confidence).
  3. Fail closed. Decider unreachable, timeout, malformed answer, unknown label, or
     a ctx the backend cannot take (empty, too long): gate -> escalate (never
     allow), decide -> abstain.
  4. Calibration is a post-step: `temperature` per question type is applied to the
     decider's probabilities here (one temperature per type, refit on held-out
     labels). The decider's own confidence is not trusted.

Backends (T12 in oasis-x/.swarm/TRAINING_DATA_REVIEW.md, section 6 item 11). The X
routes stay in the main gateway app; each model server stays its own process. One
adapter per backend maps the gateway request (labelset, labels, ctx) to that
backend's input, and maps the backend's answer back to {label: p}. The adapter is
the only code that knows a backend's wire format.

  DeciderSidecar -- the `oasis-decider` sidecar (Laya on CPU), source
  oasis-claw sandbox/decider/decider_server.py. Schema read 2026-11-07 from that
  file (validate(), do_POST) and from the pinned `laya` 0.4.0 package in the
  oasis-decider image (agent.py `_decode_answers`):

    POST {decider_url}  {"state": <str | object>, "questions": {<qid>: <question>}}
      question = {"type": "choice", "instructions": str, "criteria": {label: description}}
    200 -> {"answers": {<qid>: {"type": "choice", "choice": label,
                                "probabilities": {label: p}, "confidence": ...,
                                "answer_confidence": ..., "action": {...}}},
            "usage": {...}, "ms": float, "model": str, "revision": str}
    400 bad request, 413 body too big, 503 busy, 500 model error.

  tests/contract/ pins this schema: a byte copy of the sidecar server, and answers
  recorded from the real image.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Awaitable, Callable, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

OUT_OF_SCOPE = "out_of_scope"
OUT_OF_SCOPE_DESCRIPTION = ("None of the other options fits: the input is not about this question, "
                            "or it gives too little to decide.")
DEFAULT_INSTRUCTIONS = "Choose the one option that fits this input best."
VERDICTS = ("allow", "escalate", "deny")
_RANK = {"allow": 0, "escalate": 1, "deny": 2}


class LabelSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    labels: list[str] = Field(min_length=1)
    qtype: Literal["choice", "noul", "score"] = "choice"
    abstain_floor: float = Field(0.5, ge=0.0, le=1.0)
    temperature: float = Field(1.0, gt=0.0)
    # The text the model reads. Both optional, so older labelset files still load:
    # no instructions -> DEFAULT_INSTRUCTIONS; no description -> the label itself.
    instructions: str = ""
    descriptions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "LabelSet":
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"labelset {self.id!r}: labels must be unique")
        unknown = set(self.descriptions) - set(self.with_oos())
        if unknown:
            raise ValueError(f"labelset {self.id!r}: descriptions for unknown labels {sorted(unknown)}")
        return self

    def with_oos(self) -> list[str]:
        return self.labels if OUT_OF_SCOPE in self.labels else [*self.labels, OUT_OF_SCOPE]

    def description(self, label: str) -> str:
        if label in self.descriptions:
            return self.descriptions[label]
        return OUT_OF_SCOPE_DESCRIPTION if label == OUT_OF_SCOPE else label

    def question_text(self) -> str:
        return self.instructions.strip() or DEFAULT_INSTRUCTIONS


# Built-in label sets. Callers may register more via LabelSets.register().
GATE = LabelSet(
    id="gate.v1", labels=list(VERDICTS), abstain_floor=0.5,
    instructions="An agent proposes the action in this state. Choose the verdict for the action.",
    descriptions={
        "allow": "The action is safe, fits the request, and needs no check by a person.",
        "escalate": ("The action could be risky, or its purpose is unclear: a person or a "
                     "stronger reviewer must check it."),
        "deny": "The action is harmful or unsafe, or it is clearly outside the request.",
    })
ALLOW_FLOOR = 0.9   # a bare X "allow" must clear this to be honoured (design: allow only at high confidence)


class LabelSets:
    def __init__(self) -> None:
        self._s: dict[str, LabelSet] = {GATE.id: GATE}

    def register(self, ls: LabelSet) -> None:
        if ls.id in self._s and self._s[ls.id] != ls:
            raise ValueError(f"labelset {ls.id!r} already registered with a different definition")
        self._s[ls.id] = ls

    def get(self, lid: str) -> LabelSet:
        try:
            return self._s[lid]
        except KeyError:
            raise KeyError(f"unknown labelset {lid!r}") from None


def calibrate(probs: dict[str, float], temperature: float) -> dict[str, float]:
    """Temperature-scale a distribution and renormalise (T>1 flattens)."""
    if not probs:
        return {}
    logs = {k: math.log(max(p, 1e-12)) / temperature for k, p in probs.items()}
    m = max(logs.values())
    ex = {k: math.exp(v - m) for k, v in logs.items()}
    z = sum(ex.values())
    return {k: v / z for k, v in ex.items()}


# ---- backend adapters --------------------------------------------------------
class BackendError(ValueError):
    """The request cannot be mapped to the backend, or its answer is malformed.
    decide() catches it like every other failure: abstain, never a guess."""


class Backend(Protocol):
    name: str

    def request(self, ls: LabelSet, ctx: Any) -> dict:
        """The backend's request body for this labelset and ctx, or raise BackendError."""

    def parse(self, raw: Any, ls: LabelSet) -> tuple[dict[str, float], dict]:
        """({label: p} over ls.with_oos(), summing to 1; served-by metadata), or raise BackendError."""


def encode_body(body: dict) -> bytes:
    """The exact bytes http_caller sends, so a size check sees the real size."""
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


class DeciderSidecar:
    """Adapter for the oasis-decider sidecar (schema in the module docstring).

    One gateway decision is one sidecar `choice` question with id QID. Its criteria
    are the labelset's labels plus OUT_OF_SCOPE, each with its description. Only
    `choice` can carry the out-of-scope option (rule 1), so a `noul` or `score`
    labelset abstains on this backend instead of losing that option."""

    name = "oasis-decider"
    QID = "decision"
    # The sidecar's limits (decider_server.py). tests/contract checks that they match.
    MAX_STATE_CHARS = 6000
    MAX_CHOICE_OPTIONS = 20
    MAX_QUESTIONS = 8
    MAX_BODY_BYTES = 64 * 1024

    def state(self, ctx: Any) -> str | dict:
        """Gateway ctx -> sidecar state (a string or an object, at most MAX_STATE_CHARS).

        A list becomes {"ctx": [...]}: the sidecar takes no bare list. An empty ctx or
        an oversized one is an error, never a silent cut: a decision on a cut state is
        a decision on a different input (review T4)."""
        if isinstance(ctx, str):
            if not ctx.strip():
                raise BackendError("ctx is empty: nothing to decide on")
            state: str | dict = ctx
            size = len(ctx)
        elif isinstance(ctx, (dict, list)):
            if not ctx:
                raise BackendError("ctx is empty: nothing to decide on")
            state = ctx if isinstance(ctx, dict) else {"ctx": ctx}
            size = len(json.dumps(state))   # the sidecar's own measure: json.dumps defaults
        elif ctx is None:
            raise BackendError("ctx is empty: nothing to decide on")
        else:
            raise BackendError(f"ctx must be an object, a list or a string, not {type(ctx).__name__}")
        if size > self.MAX_STATE_CHARS:
            raise BackendError(f"ctx is {size} chars as the decider state; {self.name} takes at most "
                               f"{self.MAX_STATE_CHARS} (the gateway does not truncate)")
        return state

    def request(self, ls: LabelSet, ctx: Any) -> dict:
        if ls.qtype != "choice":
            raise BackendError(f"labelset {ls.id!r} is {ls.qtype!r}; {self.name} takes only 'choice' "
                               f"labelsets, the one type that carries {OUT_OF_SCOPE!r}")
        labels = ls.with_oos()
        if not 2 <= len(labels) <= self.MAX_CHOICE_OPTIONS:
            raise BackendError(f"labelset {ls.id!r} has {len(labels)} options with {OUT_OF_SCOPE!r}; "
                               f"{self.name} takes 2..{self.MAX_CHOICE_OPTIONS}")
        body = {"state": self.state(ctx),
                "questions": {self.QID: {"type": "choice", "instructions": ls.question_text(),
                                         "criteria": {lab: ls.description(lab) for lab in labels}}}}
        size = len(encode_body(body))
        if size > self.MAX_BODY_BYTES:
            raise BackendError(f"request is {size} bytes; {self.name} takes at most {self.MAX_BODY_BYTES}")
        return body

    def parse(self, raw: Any, ls: LabelSet) -> tuple[dict[str, float], dict]:
        labels = ls.with_oos()
        if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
            raise BackendError("decider response has no 'answers' object")
        ans = raw["answers"].get(self.QID)
        if not isinstance(ans, dict):
            raise BackendError(f"decider response has no answer {self.QID!r}")
        if ans.get("type") != "choice":
            raise BackendError(f"decider answer type is {ans.get('type')!r}, not 'choice'")
        probs = ans.get("probabilities")
        if not isinstance(probs, dict) or not probs:
            raise BackendError("decider answer has no probabilities")
        if set(probs) != set(labels):
            raise BackendError(f"decider answered labels {sorted(probs)}; the labelset has {sorted(labels)}")
        out: dict[str, float] = {}
        for lab in labels:
            v = probs[lab]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or not 0.0 <= v <= 1.0:
                raise BackendError(f"probability for {lab!r} is not in [0, 1]: {v!r}")
            out[lab] = float(v)
        total = sum(out.values())
        if total <= 0:
            raise BackendError("probabilities sum to zero")
        served = {"model": raw.get("model"), "revision": raw.get("revision"), "ms": raw.get("ms"),
                  "usage": raw.get("usage") if isinstance(raw.get("usage"), dict) else None}
        return {lab: v / total for lab, v in out.items()}, served


Caller = Callable[[str, dict, float], Awaitable[Any]]


async def http_caller(url: str, body: dict, timeout_s: float, transport=None) -> Any:
    async with httpx.AsyncClient(timeout=timeout_s, transport=transport, trust_env=False,
                                 follow_redirects=False) as c:
        r = await c.post(url, content=encode_body(body), headers={"Content-Type": "application/json"})
        r.raise_for_status()
        return r.json()


class XService:
    def __init__(self, decider_url: str, labelsets: LabelSets | None = None,
                 caller: Caller | None = None, timeout_s: float = 10.0,
                 backend: Backend | None = None) -> None:
        self.url = decider_url
        self.labelsets = labelsets or LabelSets()
        self._call = caller or (lambda u, b, t: http_caller(u, b, t))
        self.timeout_s = timeout_s
        self.backend: Backend = backend or DeciderSidecar()

    async def decide(self, *, model: str, labelset_id: str, ctx: Any) -> dict:
        """The x_decision record: {labelset_id, probs, chosen, abstain, abstain_floor, ...}.

        `model` is the caller's X model id, kept in the record; the backend reports
        what actually served the call under `served` (model, revision, ms, usage)."""
        ls = self.labelsets.get(labelset_id)
        t0 = time.monotonic()
        rec: dict = {"labelset_id": ls.id, "model": model, "abstain_floor": ls.abstain_floor,
                     "qtype": ls.qtype, "backend": self.backend.name, "served": None, "error": None}
        try:
            body = self.backend.request(ls, ctx)
            raw = await self._call(self.url, body, self.timeout_s)
            probs, served = self.backend.parse(raw, ls)
            probs = calibrate(probs, ls.temperature)
        except Exception as exc:  # noqa: BLE001 - every failure must fail closed
            rec.update(probs={}, chosen=None, abstain=True, error=f"{type(exc).__name__}: {exc}"[:200])
            rec["ms"] = int((time.monotonic() - t0) * 1000)
            return rec
        chosen = max(probs, key=probs.get)
        abstain = chosen == OUT_OF_SCOPE or probs[chosen] < ls.abstain_floor
        rec.update(probs=probs, chosen=None if abstain else chosen, abstain=abstain, served=served,
                   ms=int((time.monotonic() - t0) * 1000))
        return rec

    async def gate(self, *, model: str, action: dict, ctx: Any = None,
                   baseline: str | None = None) -> dict:
        """Verdict on a proposed side effect. Restrictive-only against `baseline`."""
        if baseline is not None and baseline not in _RANK:
            raise ValueError(f"baseline must be one of {VERDICTS}")
        base = baseline or "escalate"
        rec = await self.decide(model=model, labelset_id=GATE.id, ctx={"action": action, "ctx": ctx})
        x = rec["chosen"]
        reason = "x abstained or failed: escalate" if x is None else f"x said {x}"
        if x == "allow" and rec["probs"].get("allow", 0.0) < ALLOW_FLOOR:
            x, reason = None, f"x allow below allow floor {ALLOW_FLOOR}: escalate"
        x = x or "escalate"
        # tighten only: the stricter of baseline and X
        verdict = x if _RANK[x] > _RANK[base] else base
        return {"verdict": verdict, "baseline": base, "x_verdict": x, "reason": reason,
                "tightened": _RANK[verdict] > _RANK[base], "decision": rec}
