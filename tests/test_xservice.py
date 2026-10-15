"""X service tests (GEN-008). Decider is a stub caller; no network."""

import asyncio

import pytest

from oasis_generation.xservice import (ALLOW_FLOOR, OUT_OF_SCOPE, LabelSet, LabelSets, XService,
                                       calibrate)


def run(c):
    return asyncio.run(c)


def svc(answer=None, exc=None, **kw):
    async def caller(url, body, t):
        if exc:
            raise exc
        return answer
    return XService("http://decider/v1/decide", caller=caller, **kw)


def P(**p):
    return {"decVerdictProbs": p}


def test_decide_chooses_top_above_floor():
    r = run(svc(P(allow=0.1, escalate=0.8, deny=0.1, out_of_scope=0.0)).decide(
        model="m", labelset_id="gate.v1", ctx={}))
    assert r["chosen"] == "escalate" and r["abstain"] is False and r["error"] is None


def test_decide_abstains_below_floor_and_on_oos():
    s = svc(P(allow=0.4, escalate=0.35, deny=0.25))
    r = run(s.decide(model="m", labelset_id="gate.v1", ctx={}))
    assert r["abstain"] and r["chosen"] is None
    r = run(svc(P(allow=0.0, escalate=0.0, deny=0.0, out_of_scope=1.0)).decide(
        model="m", labelset_id="gate.v1", ctx={}))
    assert r["abstain"] and r["chosen"] is None


def test_single_verdict_shape_accepted():
    r = run(svc({"decVerdict": "deny", "decVerdictP": 0.9}).decide(model="m", labelset_id="gate.v1", ctx={}))
    assert r["chosen"] == "deny"


@pytest.mark.parametrize("bad", [None, [], {}, {"decVerdictProbs": {"allow": 1, "sure": 1}},
                                 {"decVerdictProbs": {"allow": -0.1, "deny": 1.1}},
                                 {"decVerdictProbs": {"allow": 0, "deny": 0}}])
def test_malformed_answer_abstains_with_error(bad):
    r = run(svc(bad).decide(model="m", labelset_id="gate.v1", ctx={}))
    assert r["abstain"] and r["chosen"] is None and r["error"]


def test_decider_down_fails_closed():
    r = run(svc(exc=TimeoutError("slow")).decide(model="m", labelset_id="gate.v1", ctx={}))
    assert r["abstain"] and "TimeoutError" in r["error"]


def test_unknown_labelset_raises():
    with pytest.raises(KeyError):
        run(svc(P(allow=1)).decide(model="m", labelset_id="nope", ctx={}))


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


def test_gate_never_loosens_baseline():
    # X is confident allow, baseline deny -> stays deny
    s = svc(P(allow=0.99, escalate=0.005, deny=0.005))
    g = run(s.gate(model="m", action={"tool": "x"}, baseline="deny"))
    assert g["verdict"] == "deny" and not g["tightened"]
    g = run(s.gate(model="m", action={"tool": "x"}, baseline="escalate"))
    assert g["verdict"] == "allow" or g["verdict"] == "escalate"
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


def test_gate_fails_closed_never_allow():
    g = run(svc(exc=ConnectionError("down")).gate(model="m", action={}, baseline="allow"))
    assert g["verdict"] == "escalate" and g["tightened"]


def test_gate_bad_baseline():
    with pytest.raises(ValueError):
        run(svc(P(allow=1)).gate(model="m", action={}, baseline="maybe"))
