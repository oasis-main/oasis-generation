"""X service tests (GEN-008, T12). Decider is a stub caller; no network.

The stub answers in the oasis-decider sidecar shape (laya 0.4.0). The wire
contract against the real sidecar code is in test_decider_contract.py.
"""

import asyncio
import itertools
import json
import math

import httpx
import pytest
from pydantic import ValidationError

from oasis_generation.xservice import (ALLOW_FLOOR, DEFAULT_INSTRUCTIONS, GATE, OUT_OF_SCOPE,
                                       OUT_OF_SCOPE_DESCRIPTION, VERDICTS, DeciderSidecar, LabelSet,
                                       LabelSets, XService, calibrate, encode_body, http_caller)

QID = DeciderSidecar.QID
CTX = {"request": "check the repo", "tool": "exec", "params": {"cmd": "git status"}}
_RANK = {v: i for i, v in enumerate(VERDICTS)}


def run(c):
    return asyncio.run(c)


def answer(probs, qid=QID, qtype="choice", **top):
    """A sidecar 200 body in the laya 0.4.0 shape (agent.py _decode_answers)."""
    a = {"type": qtype, "choice": next(iter(probs), None), "probabilities": probs,
         "confidence": 0.5, "answer_confidence": 0.5, "action": {"act_probability": 0.5}}
    body = {"answers": {qid: a}, "usage": {"input_tokens": 99, "truncated": False}, "ms": 12.5,
            "model": "convaiinnovations/laya-typed-decisions", "revision": "e929ae5"}
    body.update(top)
    return body


def P(**p):
    """Gate probabilities; out_of_scope gets 0.0 unless given."""
    p.setdefault(OUT_OF_SCOPE, 0.0)
    return answer(p)


def svc(reply=None, exc=None, **kw):
    """XService with a stub caller. `.calls` lists every (url, body) the backend got."""
    calls = []

    async def caller(url, body, t):
        calls.append((url, body))
        if exc:
            raise exc
        return reply(body) if callable(reply) else reply
    s = XService("http://decider/v1/decide", caller=caller, **kw)
    s.calls = calls
    return s


def with_labels(ls):
    reg = LabelSets()
    reg.register(ls)
    return reg


# ---- request mapping: gateway {labelset, labels, ctx} -> sidecar {state, questions} ----

def test_request_is_state_and_questions_only():
    """T12: the pre-fix body {model, labelset_id, labels, ctx} must never be sent again."""
    s = svc(P(allow=0.1, escalate=0.8, deny=0.1))
    run(s.decide(model="m", labelset_id="gate.v1", ctx=CTX))
    (url, body), = s.calls
    assert set(body) == {"state", "questions"}
    assert body["state"] == CTX
    assert list(body["questions"]) == [QID]
    q = body["questions"][QID]
    assert q["type"] == "choice" and q["instructions"] == GATE.instructions
    assert list(q["criteria"]) == ["allow", "escalate", "deny", OUT_OF_SCOPE]
    assert q["criteria"]["deny"] == GATE.descriptions["deny"]
    assert q["criteria"][OUT_OF_SCOPE] == OUT_OF_SCOPE_DESCRIPTION


@pytest.mark.parametrize("ctx,state", [
    ({"a": 1}, {"a": 1}),
    ("rm -rf / was proposed", "rm -rf / was proposed"),
    ([{"role": "user", "content": "hi"}], {"ctx": [{"role": "user", "content": "hi"}]}),
])
def test_ctx_maps_to_state(ctx, state):
    assert DeciderSidecar().state(ctx) == state


@pytest.mark.parametrize("ctx", [None, "", "   ", {}, []])
def test_empty_ctx_abstains_without_a_call(ctx):
    s = svc(P(allow=1.0, escalate=0.0, deny=0.0))
    r = run(s.decide(model="m", labelset_id="gate.v1", ctx=ctx))
    assert r["abstain"] and r["chosen"] is None and "empty" in r["error"]
    assert s.calls == []


def test_ctx_of_another_type_abstains_without_a_call():
    s = svc(P(allow=1.0, escalate=0.0, deny=0.0))
    r = run(s.decide(model="m", labelset_id="gate.v1", ctx=42))
    assert r["abstain"] and "int" in r["error"] and s.calls == []


def test_state_limit_is_the_sidecars_measure_and_never_truncates():
    lim = DeciderSidecar.MAX_STATE_CHARS
    ad = DeciderSidecar()
    assert ad.state("x" * lim) == "x" * lim
    with pytest.raises(ValueError, match="does not truncate"):
        ad.state("x" * (lim + 1))
    pad = lim - len(json.dumps({"t": ""}))
    assert ad.state({"t": "x" * pad}) == {"t": "x" * pad}
    with pytest.raises(ValueError):
        ad.state({"t": "x" * (pad + 1)})
    # The sidecar measures an object with json.dumps defaults (ensure_ascii=True), where
    # "é" is 6 chars (é). The adapter must use the same measure, or a state it
    # accepts comes back as HTTP 400.
    n = pad // 6 + 1
    assert len(json.dumps({"t": "é" * n}, ensure_ascii=False)) <= lim < len(json.dumps({"t": "é" * n}))
    with pytest.raises(ValueError):
        ad.state({"t": "é" * n})


def test_oversized_ctx_abstains_without_a_call():
    s = svc(P(allow=1.0, escalate=0.0, deny=0.0))
    r = run(s.decide(model="m", labelset_id="gate.v1", ctx={"t": "x" * 7000}))
    assert r["abstain"] and "at most 6000" in r["error"] and s.calls == []


def test_custom_labelset_defaults_and_explicit_text():
    plain = LabelSet(id="route.v1", labels=["haiku", "sonnet"])
    body = DeciderSidecar().request(plain, "a request")
    q = body["questions"][QID]
    assert q["instructions"] == DEFAULT_INSTRUCTIONS
    assert q["criteria"] == {"haiku": "haiku", "sonnet": "sonnet", OUT_OF_SCOPE: OUT_OF_SCOPE_DESCRIPTION}
    told = LabelSet(id="route.v2", labels=["haiku", "sonnet"], instructions="  Pick the tier.  ",
                    descriptions={"haiku": "short reply", OUT_OF_SCOPE: "not a chat turn"})
    q = DeciderSidecar().request(told, "a request")["questions"][QID]
    assert q["instructions"] == "Pick the tier."
    assert q["criteria"] == {"haiku": "short reply", "sonnet": "sonnet", OUT_OF_SCOPE: "not a chat turn"}


def test_explicit_out_of_scope_label_is_not_doubled():
    ls = LabelSet(id="r", labels=["a", OUT_OF_SCOPE, "b"])
    assert list(DeciderSidecar().request(ls, "s")["questions"][QID]["criteria"]) == ["a", OUT_OF_SCOPE, "b"]


@pytest.mark.parametrize("qtype", ["noul", "score"])
def test_non_choice_labelset_abstains_on_this_backend(qtype):
    """Only a choice question carries the out-of-scope option (rule 1)."""
    ls = LabelSet(id=f"{qtype}.v1", labels=["no", "yes"], qtype=qtype)
    s = svc(answer({"no": 0.1, "yes": 0.9, OUT_OF_SCOPE: 0.0}), labelsets=with_labels(ls))
    r = run(s.decide(model="m", labelset_id=ls.id, ctx="s"))
    assert r["abstain"] and "only 'choice'" in r["error"] and s.calls == []


def test_option_count_limit():
    lim = DeciderSidecar.MAX_CHOICE_OPTIONS
    ok = LabelSet(id="ok", labels=[f"l{i}" for i in range(lim - 1)])     # + out_of_scope = lim
    assert len(DeciderSidecar().request(ok, "s")["questions"][QID]["criteria"]) == lim
    big = LabelSet(id="big", labels=[f"l{i}" for i in range(lim)])
    s = svc(None, labelsets=with_labels(big))
    r = run(s.decide(model="m", labelset_id="big", ctx="s"))
    assert r["abstain"] and "2..20" in r["error"] and s.calls == []


def test_oversized_body_abstains_without_a_call():
    huge = LabelSet(id="huge", labels=["a", "b"], descriptions={"a": "d" * (70 * 1024)})
    s = svc(None, labelsets=with_labels(huge))
    r = run(s.decide(model="m", labelset_id="huge", ctx="s"))
    assert r["abstain"] and "bytes" in r["error"] and s.calls == []


def test_labelset_validation():
    with pytest.raises(ValidationError, match="unique"):
        LabelSet(id="d", labels=["a", "a"])
    with pytest.raises(ValidationError, match="unknown labels"):
        LabelSet(id="d", labels=["a"], descriptions={"b": "?"})
    LabelSet(id="d", labels=["a"], descriptions={OUT_OF_SCOPE: "fine"})


def test_labelset_file_entry_without_text_still_loads():
    """Older labelsets files have no instructions/descriptions; they must still load."""
    LabelSet.model_validate({"id": "route.v1", "labels": ["a", "b"], "abstain_floor": 0.6})


# ---- response mapping: sidecar answers -> {label: p} --------------------------------

def test_decide_chooses_top_above_floor():
    r = run(svc(P(allow=0.1, escalate=0.8, deny=0.1)).decide(model="m", labelset_id="gate.v1", ctx=CTX))
    assert r["chosen"] == "escalate" and r["abstain"] is False and r["error"] is None
    assert set(r["probs"]) == {"allow", "escalate", "deny", OUT_OF_SCOPE}


def test_decide_abstains_below_floor_and_on_oos():
    """Abstain is a valid answer, not an error."""
    r = run(svc(P(allow=0.4, escalate=0.35, deny=0.25)).decide(model="m", labelset_id="gate.v1", ctx=CTX))
    assert r["abstain"] and r["chosen"] is None and r["error"] is None
    r = run(svc(P(allow=0.0, escalate=0.0, deny=0.0, out_of_scope=1.0)).decide(
        model="m", labelset_id="gate.v1", ctx=CTX))
    assert r["abstain"] and r["chosen"] is None and r["error"] is None


def test_rounded_probabilities_are_renormalised():
    # laya rounds each p to 4 decimals, so a sum of 0.9999 is normal
    r = run(svc(P(allow=0.3333, escalate=0.3333, deny=0.3333)).decide(model="m", labelset_id="gate.v1", ctx=CTX))
    assert math.isclose(sum(r["probs"].values()), 1.0)


def test_served_metadata_is_recorded():
    r = run(svc(P(allow=0.1, escalate=0.8, deny=0.1)).decide(model="laya-x", labelset_id="gate.v1", ctx=CTX))
    assert r["model"] == "laya-x" and r["backend"] == "oasis-decider"
    assert r["served"] == {"model": "convaiinnovations/laya-typed-decisions", "revision": "e929ae5",
                           "ms": 12.5, "usage": {"input_tokens": 99, "truncated": False}}


def _gate_answer(probs):
    return answer(probs)["answers"][QID]


@pytest.mark.parametrize("bad", [
    None, [], {}, "ok",
    {"answers": []},
    {"answers": {}},
    {"answers": {"verdict": _gate_answer({"allow": 1.0, "escalate": 0, "deny": 0, OUT_OF_SCOPE: 0})}},
    answer({"allow": 1.0, "escalate": 0, "deny": 0, OUT_OF_SCOPE: 0}, qtype="noul"),
    {"answers": {QID: {"type": "choice", "choice": "allow"}}},
    {"answers": {QID: {"type": "choice", "probabilities": {}}}},
    P(allow=0.5, escalate=0.2, deny=0.2, sure=0.1),          # unknown label
    answer({"allow": 0.5, "escalate": 0.3, "deny": 0.2}),     # out_of_scope missing
    P(allow=-0.1, escalate=0.6, deny=0.5),
    P(allow=1.1, escalate=0.0, deny=0.0),
    P(allow=float("nan"), escalate=0.5, deny=0.5),
    P(allow=True, escalate=0.0, deny=0.0),
    P(allow="0.9", escalate=0.05, deny=0.05),
    P(allow=0, escalate=0, deny=0),
    # the pre-fix guessed shape is not the sidecar's
    {"decVerdictProbs": {"allow": 0.98, "escalate": 0.01, "deny": 0.01}},
])
def test_malformed_answer_abstains_with_error(bad):
    r = run(svc(bad).decide(model="m", labelset_id="gate.v1", ctx=CTX))
    assert r["abstain"] and r["chosen"] is None and r["error"] and r["probs"] == {}
    assert r["served"] is None


@pytest.mark.parametrize("probs", [{"allow": 0.5, "escalate": 0.3, "deny": 0.2},
                                   {"allow": 0.5, "escalate": 0.2, "deny": 0.2, OUT_OF_SCOPE: 0, "sure": 0.1}])
def test_label_mismatch_error_names_both_label_sets(probs):
    """A backend that saw other options than the labelset must say so plainly."""
    r = run(svc(answer(probs)).decide(model="m", labelset_id="gate.v1", ctx=CTX))
    assert r["abstain"] and "BackendError" in r["error"] and "the labelset has" in r["error"]


def test_decider_down_fails_closed():
    r = run(svc(exc=TimeoutError("slow")).decide(model="m", labelset_id="gate.v1", ctx=CTX))
    assert r["abstain"] and "TimeoutError" in r["error"]


def test_unknown_labelset_raises():
    with pytest.raises(KeyError):
        run(svc(P(allow=1)).decide(model="m", labelset_id="nope", ctx=CTX))


def test_labelset_registration_conflict():
    ls = LabelSets()
    ls.register(LabelSet(id="route.v1", labels=["a", "b"]))
    ls.register(LabelSet(id="route.v1", labels=["a", "b"]))   # identical: fine
    with pytest.raises(ValueError):
        ls.register(LabelSet(id="route.v1", labels=["a", "c"]))


def test_oos_always_added():
    assert OUT_OF_SCOPE in LabelSet(id="x", labels=["a"]).with_oos()


def test_calibrate_flattens_and_normalises():
    p = calibrate({"a": 0.9, "b": 0.1}, 1.0)
    q = calibrate({"a": 0.9, "b": 0.1}, 3.0)
    assert abs(sum(q.values()) - 1) < 1e-9 and q["a"] < p["a"]


# ---- HTTP caller: exact bytes, and HTTP errors abstain -------------------------------

def _http_svc(handler):
    t = httpx.MockTransport(handler)
    return XService("http://decider/v1/decide", caller=lambda u, b, s: http_caller(u, b, s, transport=t))


def test_http_caller_sends_the_encoded_body():
    seen = {}

    def handler(req):
        seen["ct"], seen["raw"] = req.headers["content-type"], req.content
        return httpx.Response(200, json=P(allow=0.1, escalate=0.8, deny=0.1))
    r = run(_http_svc(handler).decide(model="m", labelset_id="gate.v1", ctx={"t": "café"}))
    assert r["chosen"] == "escalate"
    assert seen["ct"] == "application/json"
    assert seen["raw"] == encode_body(DeciderSidecar().request(GATE, {"t": "café"}))


@pytest.mark.parametrize("status,err", [(400, {"error": "state must be a string or an object"}),
                                        (413, {"error": "bad Content-Length"}),
                                        (503, {"error": "busy: 8 calls pending"}),
                                        (500, {"error": "RuntimeError: boom"})])
def test_http_error_statuses_abstain(status, err):
    r = run(_http_svc(lambda req: httpx.Response(status, json=err)).decide(
        model="m", labelset_id="gate.v1", ctx=CTX))
    assert r["abstain"] and "HTTPStatusError" in r["error"] and str(status) in r["error"]


# ---- /v1/gate: restrictive-only -----------------------------------------------------

def test_gate_state_is_action_and_ctx():
    s = svc(P(allow=0.1, escalate=0.8, deny=0.1))
    run(s.gate(model="m", action={"tool": "exec"}, ctx="asked to tidy a README", baseline="allow"))
    assert s.calls[0][1]["state"] == {"action": {"tool": "exec"}, "ctx": "asked to tidy a README"}


def test_gate_never_loosens_baseline():
    # X is confident allow, baseline deny -> stays deny
    s = svc(P(allow=0.99, escalate=0.005, deny=0.005))
    g = run(s.gate(model="m", action={"tool": "x"}, baseline="deny"))
    assert g["verdict"] == "deny" and not g["tightened"]
    g = run(s.gate(model="m", action={"tool": "x"}, baseline="escalate"))
    assert g["verdict"] == "escalate"  # allow does not loosen escalate either


def test_gate_tightens():
    s = svc(P(allow=0.02, escalate=0.03, deny=0.95))
    g = run(s.gate(model="m", action={}, baseline="allow"))
    assert g["verdict"] == "deny" and g["tightened"]


def test_gate_allow_needs_baseline_allow_and_high_confidence():
    hi = svc(P(allow=0.97, escalate=0.02, deny=0.01))
    assert run(hi.gate(model="m", action={}, baseline="allow"))["verdict"] == "allow"
    # allow at 0.8: clears the abstain floor but not the allow floor -> escalate
    mid = svc(P(allow=0.8, escalate=0.1, deny=0.1))
    assert 0.8 < ALLOW_FLOOR
    g = run(mid.gate(model="m", action={}, baseline="allow"))
    assert g["verdict"] == "escalate" and g["tightened"]


def test_gate_default_baseline_is_escalate_so_bare_allow_cannot_pass():
    hi = svc(P(allow=0.99, escalate=0.005, deny=0.005))
    assert run(hi.gate(model="m", action={}))["verdict"] == "escalate"


@pytest.mark.parametrize("reply,exc", [(None, ConnectionError("down")), ({"answers": {}}, None),
                                       (P(allow=0.0, escalate=0.0, deny=0.0, out_of_scope=1.0), None)])
def test_gate_fails_closed_never_allow(reply, exc):
    g = run(svc(reply, exc=exc).gate(model="m", action={}, baseline="allow"))
    assert g["verdict"] == "escalate" and g["tightened"] and g["decision"]["abstain"]


_GRID = [0.0, 0.05, 0.5, 0.85, 0.9, 0.95, 1.0]


@pytest.mark.parametrize("baseline", [None, *VERDICTS])
def test_gate_property_verdict_never_looser_than_baseline(baseline):
    """Every answer on a grid: the verdict is never looser than the baseline, and an
    allow needs baseline allow plus p(allow) >= ALLOW_FLOOR."""
    base = baseline or "escalate"

    async def sweep():
        seen = set()
        for pa, pe, pd, po in itertools.product(_GRID, repeat=4):
            if pa + pe + pd + po == 0:
                continue
            g = await svc(P(allow=pa, escalate=pe, deny=pd, out_of_scope=po)).gate(
                model="m", action={"tool": "x"}, baseline=baseline)
            assert _RANK[g["verdict"]] >= _RANK[base], (pa, pe, pd, po, g)
            if g["verdict"] == "allow":
                assert base == "allow" and g["decision"]["probs"]["allow"] >= ALLOW_FLOOR
            seen.add(g["verdict"])
        return seen
    seen = run(sweep())
    # the sweep must reach the tighter verdicts, or it proves nothing
    assert "deny" in seen and ("allow" in seen) == (base == "allow")


def test_gate_bad_baseline():
    with pytest.raises(ValueError):
        run(svc(P(allow=1)).gate(model="m", action={}, baseline="maybe"))
