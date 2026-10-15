"""X endpoints (GEN-008): /v1/x/decide and /v1/gate over the oasis-decider sidecar.

Design: docs/OX-MODELS-INTERFACE-TRAINING-DESIGN.md sections 1-3. X holds authority,
O holds ideas. These rules are enforced HERE, in code, not left to callers:

  1. Closed label set. Every labelset carries an explicit out-of-scope option
     (`OUT_OF_SCOPE`). A decider that puts its mass there, or whose top
     probability is below the labelset's abstain floor, yields abstain=True.
  2. Restrictive-only. For /v1/gate, X may only TIGHTEN a baseline verdict
     (allow -> escalate -> deny). It can never loosen one. With no baseline the
     default baseline is "escalate", so a bare X "allow" is only honoured when it
     clears the allow floor (high confidence).
  3. Fail closed. Decider unreachable, timeout, malformed answer, unknown label:
     gate -> escalate (never allow), decide -> abstain.
  4. Calibration is a post-step: `temperature` per question type is applied to the
     decider's probabilities here (one temperature per type, refit on held-out
     labels). The decider's own confidence is not trusted.

ASSUMPTION (unverified): the decider source (oasis-claw) is outside my reach, so
the request body sent to `{decider_url}` and the response keys read back are taken
from the audit-row field names in the console mail (decVerdict, decVerdictP,
decVerdictProbs, decVerdictConfidence, dec_<qid>). The adapter in `_parse_answer`
is the only place that knows the response shape; fix it there once the real
schema is read. See tests for the shapes accepted.
"""

from __future__ import annotations

import math
import time
from typing import Any, Awaitable, Callable, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

OUT_OF_SCOPE = "out_of_scope"
VERDICTS = ("allow", "escalate", "deny")
_RANK = {"allow": 0, "escalate": 1, "deny": 2}


class LabelSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    labels: list[str] = Field(min_length=1)
    qtype: Literal["choice", "noul", "score"] = "choice"
    abstain_floor: float = Field(0.5, ge=0.0, le=1.0)
    temperature: float = Field(1.0, gt=0.0)

    def with_oos(self) -> list[str]:
        return self.labels if OUT_OF_SCOPE in self.labels else [*self.labels, OUT_OF_SCOPE]


# Built-in label sets. Callers may register more via LabelSets.register().
GATE = LabelSet(id="gate.v1", labels=list(VERDICTS), abstain_floor=0.5)
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


def _parse_answer(raw: Any, labels: list[str]) -> dict[str, float]:
    """Return {label: p} from the decider response, or raise ValueError.

    Accepted shapes (see module ASSUMPTION): a top-level probs map under
    `decVerdictProbs` / `probs`; or a single `decVerdict` + `decVerdictP`
    (remaining mass spread evenly). Unknown labels are an error, not ignored."""
    if not isinstance(raw, dict):
        raise ValueError("decider answer is not an object")
    probs = raw.get("decVerdictProbs") or raw.get("probs")
    if isinstance(probs, dict) and probs:
        out = {str(k): float(v) for k, v in probs.items()}
    elif raw.get("decVerdict") is not None and raw.get("decVerdictP") is not None:
        top, p = str(raw["decVerdict"]), float(raw["decVerdictP"])
        rest = [l for l in labels if l != top]
        out = {top: p, **{l: (1 - p) / len(rest) for l in rest}} if rest else {top: p}
    else:
        raise ValueError("decider answer has no probabilities")
    unknown = set(out) - set(labels)
    if unknown:
        raise ValueError(f"decider returned unknown labels: {sorted(unknown)}")
    if any(not (0.0 <= v <= 1.0) or math.isnan(v) for v in out.values()):
        raise ValueError("probability out of range")
    s = sum(out.values())
    if s <= 0:
        raise ValueError("probabilities sum to zero")
    return {l: out.get(l, 0.0) / s for l in labels}


Caller = Callable[[str, dict, float], Awaitable[Any]]


async def http_caller(url: str, body: dict, timeout_s: float, transport=None) -> Any:
    async with httpx.AsyncClient(timeout=timeout_s, transport=transport, trust_env=False,
                                 follow_redirects=False) as c:
        r = await c.post(url, json=body)
        r.raise_for_status()
        return r.json()


class XService:
    def __init__(self, decider_url: str, labelsets: LabelSets | None = None,
                 caller: Caller | None = None, timeout_s: float = 10.0) -> None:
        self.url = decider_url
        self.labelsets = labelsets or LabelSets()
        self._call = caller or (lambda u, b, t: http_caller(u, b, t))
        self.timeout_s = timeout_s

    async def decide(self, *, model: str, labelset_id: str, ctx: Any) -> dict:
        """The x_decision record: {labelset_id, probs, chosen, abstain, abstain_floor, ...}."""
        ls = self.labelsets.get(labelset_id)
        labels = ls.with_oos()
        t0 = time.monotonic()
        rec: dict = {"labelset_id": ls.id, "model": model, "abstain_floor": ls.abstain_floor,
                     "qtype": ls.qtype, "error": None}
        try:
            raw = await self._call(self.url, {"model": model, "labelset_id": ls.id,
                                              "labels": labels, "ctx": ctx}, self.timeout_s)
            probs = calibrate(_parse_answer(raw, labels), ls.temperature)
        except Exception as exc:  # noqa: BLE001 - every failure must fail closed
            rec.update(probs={}, chosen=None, abstain=True, error=f"{type(exc).__name__}: {exc}"[:200])
            rec["ms"] = int((time.monotonic() - t0) * 1000)
            return rec
        chosen = max(probs, key=probs.get)
        abstain = chosen == OUT_OF_SCOPE or probs[chosen] < ls.abstain_floor
        rec.update(probs=probs, chosen=None if abstain else chosen, abstain=abstain,
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
