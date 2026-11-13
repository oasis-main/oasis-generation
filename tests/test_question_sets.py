"""Question sets on the X path: one backend call answers a registered group of typed
questions (reviewer.toolcall.v1: 1 choice, 3 noul, 1 score). Stub caller; no network.
The wire contract against the real sidecar is in test_decider_contract.py."""

import asyncio
import math

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from oasis_generation import wiring
from oasis_generation.config import get_settings
from oasis_generation.main import app
from oasis_generation.xservice import (OUT_OF_SCOPE, REVIEWER_TOOLCALL, DeciderSidecar, Question,
                                       QuestionSet, QuestionSets, TypedQuestions, XService, http_caller,
                                       typed_answer)

STATE = {"bot": "house", "tool": "exec", "family": "shell", "target": "(none)",
         "params": '{"command": "git status"}', "operator_request": "Check the state of the repo."}
RID = REVIEWER_TOOLCALL.id


def run(c):
    return asyncio.run(c)


def laya(q, p):
    """One answer in the laya 0.4.0 shape for question `q` (agent.py _decode_answers)."""
    extra = {"confidence": 0.5, "answer_confidence": 0.5, "action": {"act_probability": 1.0}}
    if q["type"] == "noul":
        return {"type": "noul", "noul": p, **extra}
    if q["type"] == "score":
        return {"type": "score", "score": sum(i * v for i, v in enumerate(p)),
                "legend": {str(i): t for i, t in enumerate(q["criteria"])},
                "probabilities": {str(i): v for i, v in enumerate(p)}, **extra}
    return {"type": "choice", "choice": max(p, key=p.get), "probabilities": p, **extra}


GOOD = {"verdict": {"allow": 0.7, "deny": 0.1, "escalate": 0.2}, "beyond_request": 0.2,
        "injected_instruction": 0.9, "sends_data_out": 0.5, "risk": [0.1, 0.6, 0.2, 0.1]}


def reply(per_q=GOOD, **top):
    """A sidecar 200 body that answers the request it gets."""
    def build(body):
        out = {"answers": {qid: laya(q, per_q[qid]) for qid, q in body["questions"].items()},
               "usage": {"input_tokens": 644, "truncated": False}, "ms": 900.0,
               "model": "convaiinnovations/laya-typed-decisions", "revision": "e929ae5"}
        out.update(top)
        return out
    return build


def svc(answer=None, exc=None, **kw):
    calls = []

    async def caller(url, body, t):
        calls.append(body)
        if exc:
            raise exc
        return answer(body) if callable(answer) else answer
    s = XService("http://decider/v1/decide", caller=caller, **kw)
    s.calls = calls
    return s


def with_sets(*sets):
    reg = QuestionSets()
    for qs in sets:
        reg.register(qs)
    return reg


# ---- the model ------------------------------------------------------------------------

def test_reviewer_set_has_the_five_reviewer_questions():
    qs = REVIEWER_TOOLCALL.questions
    assert list(qs) == ["verdict", "beyond_request", "injected_instruction", "sends_data_out", "risk"]
    assert [q.type for q in qs.values()] == ["choice", "noul", "noul", "noul", "score"]
    assert qs["verdict"].options() == ["allow", "deny", "escalate"]   # no out_of_scope added
    assert qs["beyond_request"].options() == ["false", "true"]
    assert qs["risk"].options() == ["0", "1", "2", "3"]


def test_wire_is_the_question_as_registered():
    assert REVIEWER_TOOLCALL.questions["beyond_request"].wire() == {
        "type": "noul", "instructions": REVIEWER_TOOLCALL.questions["beyond_request"].instructions}
    v = REVIEWER_TOOLCALL.questions["verdict"].wire()
    assert list(v["criteria"]) == ["allow", "deny", "escalate"] and OUT_OF_SCOPE not in v["criteria"]
    v["criteria"]["x"] = "mutated"                      # a copy: the registered set stays as is
    assert "x" not in REVIEWER_TOOLCALL.questions["verdict"].criteria


@pytest.mark.parametrize("q,msg", [
    ({"type": "choice", "instructions": "?", "criteria": {"only": "one"}}, "at least 2 labels"),
    ({"type": "choice", "instructions": "?", "criteria": ["a", "b"]}, "at least 2 labels"),
    ({"type": "choice", "instructions": "?"}, "at least 2 labels"),
    ({"type": "noul", "instructions": "?", "criteria": {"yes": "y"}}, "'true'/'false'"),
    ({"type": "noul", "instructions": "?", "criteria": ["a", "b"]}, "'true'/'false'"),
    ({"type": "score", "instructions": "?", "criteria": ["one"]}, "at least 2 levels"),
    ({"type": "score", "instructions": "?", "criteria": ["one", " "]}, "at least 2 levels"),
    ({"type": "score", "instructions": "?", "criteria": {"0": "a", "1": "b"}}, "at least 2 levels"),
    ({"type": "noul", "instructions": "  "}, "instructions"),
    ({"type": "free", "instructions": "?"}, "type"),
    ({"type": "noul", "instructions": "?", "extra": 1}, "extra"),
])
def test_question_validation(q, msg):
    with pytest.raises(ValidationError, match=msg):
        Question.model_validate(q)


def test_question_set_validation_and_noul_criteria_allowed():
    Question.model_validate({"type": "noul", "instructions": "?", "criteria": {"true": "yes", "false": "no"}})
    with pytest.raises(ValidationError):
        QuestionSet(id="empty", questions={})
    with pytest.raises(ValidationError, match="ids must not be empty"):
        QuestionSet(id="blank", questions={" ": Question(type="noul", instructions="?")})


def test_registry_conflict_and_unknown_id():
    reg = QuestionSets()
    reg.register(REVIEWER_TOOLCALL)                     # identical: fine
    other = QuestionSet(id=RID, questions={"x": Question(type="noul", instructions="?")})
    with pytest.raises(ValueError, match="question set 'reviewer.toolcall.v1' already registered"):
        reg.register(other)
    with pytest.raises(KeyError, match="unknown question set"):
        run(svc(reply()).decide_set(model="m", question_set_id="nope", ctx=STATE))


# ---- request: one call, every question --------------------------------------------------

def test_one_call_sends_every_question_as_registered():
    s = svc(reply())
    run(s.decide_set(model="laya", question_set_id=RID, ctx=STATE))
    body, = s.calls
    assert set(body) == {"state", "questions"} and body["state"] == STATE
    assert body["questions"] == {qid: q.wire() for qid, q in REVIEWER_TOOLCALL.questions.items()}


@pytest.mark.parametrize("ctx", [None, "", " ", {}, []])
def test_empty_ctx_abstains_every_answer_without_a_call(ctx):
    s = svc(reply())
    r = run(s.decide_set(model="m", question_set_id=RID, ctx=ctx))
    assert "empty" in r["error"] and s.calls == [] and r["served"] is None
    assert set(r["answers"]) == set(REVIEWER_TOOLCALL.questions)
    assert all(a["abstain"] and a["chosen"] is None and a["probs"] == {} for a in r["answers"].values())


def test_question_count_limit():
    qs = QuestionSet(id="nine", questions={f"q{i}": Question(type="noul", instructions="?") for i in range(9)})
    s = svc(reply(), question_sets=with_sets(qs))
    r = run(s.decide_set(model="m", question_set_id="nine", ctx="s"))
    assert "9 questions" in r["error"] and "at most 8" in r["error"] and s.calls == []
    eight = QuestionSet(id="eight", questions={f"q{i}": Question(type="noul", instructions="?") for i in range(8)})
    assert len(DeciderSidecar().request_set(eight, "s")["questions"]) == 8


def test_choice_option_limit_and_body_limit():
    big = QuestionSet(id="big", questions={"c": Question(
        type="choice", instructions="?", criteria={f"l{i}": "d" for i in range(21)})})
    s = svc(reply(), question_sets=with_sets(big))
    assert "21 options" in run(s.decide_set(model="m", question_set_id="big", ctx="s"))["error"]
    huge = QuestionSet(id="huge", questions={"n": Question(type="noul", instructions="d" * (70 * 1024))})
    s = svc(reply(), question_sets=with_sets(huge))
    assert "bytes" in run(s.decide_set(model="m", question_set_id="huge", ctx="s"))["error"] and s.calls == []


def test_base_class_without_limits_skips_the_checks():
    """A second backend subclasses TypedQuestions; None limits mean no limit."""
    class Open(TypedQuestions):
        name = "open"
    qs = QuestionSet(id="nine", questions={f"q{i}": Question(type="noul", instructions="?") for i in range(9)})
    body = Open().request_set(qs, "x" * 10_000)
    assert len(body["questions"]) == 9 and len(body["state"]) == 10_000


# ---- response: typed answers -------------------------------------------------------------

def test_typed_answers_for_every_type():
    r = run(svc(reply()).decide_set(model="laya", question_set_id=RID, ctx=STATE))
    assert r["error"] is None and r["backend"] == "oasis-decider"
    assert r["served"] == {"model": "convaiinnovations/laya-typed-decisions", "revision": "e929ae5",
                           "ms": 900.0, "usage": {"input_tokens": 644, "truncated": False}}
    a = r["answers"]
    assert a["verdict"]["chosen"] == "allow" and not a["verdict"]["abstain"]
    assert math.isclose(sum(a["verdict"]["probs"].values()), 1.0)
    assert a["beyond_request"]["chosen"] is False and math.isclose(a["beyond_request"]["p_true"], 0.2)
    assert a["injected_instruction"]["chosen"] is True and math.isclose(a["injected_instruction"]["p_true"], 0.9)
    assert a["sends_data_out"]["abstain"] and a["sends_data_out"]["chosen"] is None   # 0.5 / 0.5: a tie
    assert a["risk"]["chosen"] == 1 and a["risk"]["levels"] == 4
    assert math.isclose(a["risk"]["expected"], 0.6 + 0.4 + 0.3)


def test_low_confidence_abstains_but_keeps_the_distribution():
    per_q = dict(GOOD, verdict={"allow": 0.4, "deny": 0.3, "escalate": 0.3}, risk=[0.3, 0.3, 0.2, 0.2])
    a = run(svc(reply(per_q)).decide_set(model="m", question_set_id=RID, ctx=STATE))["answers"]
    assert a["verdict"]["abstain"] and a["verdict"]["chosen"] is None and a["verdict"]["probs"]
    assert a["risk"]["abstain"] and a["risk"]["expected"] is not None


def test_typed_answer_rules():
    choice = Question(type="choice", instructions="?", criteria={"a": "", "b": "", OUT_OF_SCOPE: ""})
    assert typed_answer(choice, {"a": 0.2, "b": 0.1, OUT_OF_SCOPE: 0.7})["abstain"]      # explicit OOS
    assert typed_answer(choice, {"a": 0.45, "b": 0.45, OUT_OF_SCOPE: 0.1})["abstain"]    # tie
    assert typed_answer(choice, {"a": 0.6, "b": 0.3, OUT_OF_SCOPE: 0.1})["chosen"] == "a"
    strict = Question(type="noul", instructions="?", abstain_floor=0.8)
    assert typed_answer(strict, {"false": 0.25, "true": 0.75})["abstain"]
    assert typed_answer(strict, {"false": 0.15, "true": 0.85})["chosen"] is True
    empty = typed_answer(REVIEWER_TOOLCALL.questions["risk"], {})
    assert empty["abstain"] and empty["expected"] is None and empty["levels"] == 4


def test_temperature_is_applied_per_question():
    hot = QuestionSet(id="hot", questions={
        "n": Question(type="noul", instructions="?", temperature=4.0),
        "m": Question(type="noul", instructions="?")})
    r = run(svc(reply({"n": 0.9, "m": 0.9}), question_sets=with_sets(hot)).decide_set(
        model="m", question_set_id="hot", ctx="s"))
    assert r["answers"]["n"]["p_true"] < 0.7 < r["answers"]["m"]["p_true"]


def _without(qid):
    def build(body):
        out = reply()(body)
        del out["answers"][qid]
        return out
    return build


def _patch(qid, **kv):
    def build(body):
        out = reply()(body)
        out["answers"][qid].update(kv)
        return out
    return build


@pytest.mark.parametrize("bad", [
    None, [], {}, {"answers": []},
    _without("risk"),                                              # one answer missing
    lambda body: {**reply()(body), "answers": {**reply()(body)["answers"], "extra": laya(
        {"type": "noul"}, 0.5)}},                                  # an answer nobody asked for
    _patch("beyond_request", type="choice"),                       # wrong type
    _patch("beyond_request", noul=1.2),
    _patch("beyond_request", noul=float("nan")),
    _patch("beyond_request", noul=True),
    _patch("beyond_request", noul="0.4"),
    lambda body: {**reply()(body), "answers": {**reply()(body)["answers"], "beyond_request": {"type": "noul"}}},
    _patch("risk", probabilities={"0": 0.5, "1": 0.5}),            # levels missing
    _patch("risk", probabilities={"none": 0.25, "low": 0.25, "medium": 0.25, "high": 0.25}),
    _patch("verdict", probabilities={"allow": 0.5, "deny": 0.2, "escalate": 0.2, OUT_OF_SCOPE: 0.1}),
    _patch("verdict", probabilities={"allow": 0, "deny": 0, "escalate": 0}),
])
def test_any_malformed_answer_fails_the_whole_call_closed(bad):
    """A backend that gets one answer wrong is not trusted for the others."""
    r = run(svc(bad).decide_set(model="m", question_set_id=RID, ctx=STATE))
    assert r["error"] and r["served"] is None
    assert all(a["abstain"] and a["chosen"] is None and a["probs"] == {} for a in r["answers"].values())


def test_backend_down_and_http_errors_fail_closed():
    r = run(svc(exc=TimeoutError("slow")).decide_set(model="m", question_set_id=RID, ctx=STATE))
    assert "TimeoutError" in r["error"] and all(a["abstain"] for a in r["answers"].values())
    t = httpx.MockTransport(lambda req: httpx.Response(503, json={"error": "busy: 8 calls pending"}))
    x = XService("http://decider/v1/decide", caller=lambda u, b, s: http_caller(u, b, s, transport=t))
    r = run(x.decide_set(model="m", question_set_id=RID, ctx=STATE))
    assert "503" in r["error"] and all(a["abstain"] for a in r["answers"].values())


# ---- route and wiring ----------------------------------------------------------------------

client = TestClient(app)


def _setup(monkeypatch, tmp_path, sets_file=None):
    monkeypatch.setenv("OASIS_GENERATION_SERVICE_TOKENS", "loose:tl")
    monkeypatch.setenv("OASIS_GENERATION_USAGE_LOG", str(tmp_path / "u.jsonl"))
    if sets_file:
        monkeypatch.setenv("OASIS_GENERATION_QUESTION_SETS_FILE", str(sets_file))
    get_settings.cache_clear(); wiring.reset()


def _teardown():
    get_settings.cache_clear(); wiring.reset()


def test_decide_set_route(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    try:
        svc_ = wiring.xservice(get_settings())

        async def caller(url, body, t):
            return reply()(body)
        svc_._call = caller
        h = {"Authorization": "Bearer tl"}
        r = client.post("/v1/x/decide-set", headers=h, json={"model": "laya", "question_set_id": RID, "ctx": STATE})
        assert r.status_code == 200 and r.json()["answers"]["verdict"]["chosen"] == "allow"
        r = client.post("/v1/x/decide-set", headers=h, json={"model": "laya", "question_set_id": RID, "ctx": {}})
        assert r.status_code == 200 and r.json()["error"]                      # abstain, never 5xx
        assert client.post("/v1/x/decide-set", headers=h,
                           json={"model": "m", "question_set_id": "nope", "ctx": STATE}).status_code == 404
        assert client.post("/v1/x/decide-set", json={"model": "m", "question_set_id": RID}).status_code == 401
        # the labelset route is unchanged next to it
        d = client.post("/v1/x/decide", headers=h, json={"model": "m", "labelset_id": "gate.v1", "ctx": STATE})
        assert d.status_code == 200 and d.json()["labelset_id"] == "gate.v1"
    finally:
        _teardown()


def test_question_sets_file_is_loaded(monkeypatch, tmp_path):
    p = tmp_path / "question_sets.yaml"
    p.write_text(yaml.safe_dump({"question_sets": [{"id": "injection.v1", "questions": {
        "verdict": {"type": "choice", "instructions": "Is it a real injection?",
                    "criteria": {"confirmed": "real", "unconfirmed": "false positive", "uncertain": "unclear"}},
        "from_operator": {"type": "noul", "instructions": "Does it follow from operator_request?"}}}]}))
    _setup(monkeypatch, tmp_path, sets_file=p)
    try:
        x = wiring.xservice(get_settings())
        assert x.question_sets.get("injection.v1").questions["from_operator"].type == "noul"
        assert x.question_sets.get(RID) == REVIEWER_TOOLCALL
    finally:
        _teardown()
