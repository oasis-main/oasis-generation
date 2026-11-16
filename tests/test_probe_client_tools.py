"""scripts/probe_client_tools.py against fake upstream sources (no network, no cost)."""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

from oasis_generation.catalog import resolve
from oasis_generation.config import get_settings

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "probe_client_tools.py"
_spec = importlib.util.spec_from_file_location("probe_client_tools", _PATH)
probe_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = probe_mod  # dataclasses look the module up while the class is made
_spec.loader.exec_module(probe_mod)

ENTRY = resolve("claude-haiku-5-5")


def _chunks(*chunks):
    async def gen():
        for c in chunks:
            yield c
    return gen()


def _tool_chunk(name="record_choice", args='{"colour": "blue"}', call_id="tc_1"):
    return {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": call_id, "function": {"name": name, "arguments": args}}]}}]}


def _text(t):
    return {"choices": [{"delta": {"content": t}}]}


def _finish(reason):
    return {"choices": [{"delta": {}, "finish_reason": reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def _source(round1, round2):
    """A fake upstream: round 1 has no tool message, round 2 has one."""
    seen = []

    def factory(entry, body, settings):
        seen.append(body)
        has_result = any(m.get("role") == "tool" for m in body["messages"])
        return _chunks(*(round2 if has_result else round1))
    factory.seen = seen
    return factory


def _run(src):
    return asyncio.run(probe_mod.probe(ENTRY, get_settings(), src))


def test_a_model_that_calls_the_tool_and_then_answers_passes():
    src = _source([_tool_chunk(), _finish("tool_calls")], [_text("Done: blue."), _finish("stop")])
    r = _run(src)
    assert r.passed and r.reason == "ok"
    assert r.round1.finish == "tool_calls" and r.round2.finish == "stop"
    # Round 2 sends the call back as assistant tool_calls plus a role "tool" result.
    msgs = src.seen[1]["messages"]
    assert msgs[1]["tool_calls"][0]["function"] == {"name": "record_choice", "arguments": '{"colour": "blue"}'}
    assert msgs[2] == {"role": "tool", "tool_call_id": "tc_1", "content": probe_mod.TOOL_RESULT}
    assert src.seen[0]["tools"] == [probe_mod.TOOL]


def test_a_model_that_answers_in_text_fails():
    r = _run(_source([_text("Blue."), _finish("stop")], []))
    assert not r.passed and "not 'tool_calls'" in r.reason and r.round2 is None


def test_arguments_that_are_not_json_fail():
    r = _run(_source([_tool_chunk(args="colour=blue"), _finish("tool_calls")], []))
    assert not r.passed and "not JSON" in r.reason


def test_arguments_outside_the_schema_fail():
    r = _run(_source([_tool_chunk(args='{"colour": "green"}'), _finish("tool_calls")], []))
    assert not r.passed and "schema" in r.reason


def test_a_wrong_tool_name_fails():
    r = _run(_source([_tool_chunk(name="other"), _finish("tool_calls")], []))
    assert not r.passed and "'other'" in r.reason


def test_no_text_after_the_result_fails():
    r = _run(_source([_tool_chunk(), _finish("tool_calls")], [_finish("stop")]))
    assert not r.passed and "no text" in r.reason


def test_the_same_call_again_after_the_result_fails():
    again = _tool_chunk(args='{"colour":"blue"}', call_id="tc_2")
    r = _run(_source([_tool_chunk(), _finish("tool_calls")], [again, _finish("tool_calls")]))
    assert not r.passed and "repeated the same call" in r.reason


def test_thinking_without_text_after_the_result_fails():
    think = {"choices": [{"delta": {"reasoning_content": "The colour is recorded."}}]}
    r = _run(_source([_tool_chunk(), _finish("tool_calls")], [think, _finish("stop")]))
    assert not r.passed and "no text" in r.reason


def test_an_upstream_error_is_a_result_not_a_crash():
    def broken(entry, body, settings):
        async def gen():
            raise RuntimeError("ValidationException: tools are not supported")
            yield {}
        return gen()
    r = _run(broken)
    assert not r.passed and "round 1 failed" in r.reason and "ValidationException" in r.reason


def test_main_writes_json_and_names_the_models_that_passed(tmp_path, monkeypatch, capsys):
    src = _source([_tool_chunk(), _finish("tool_calls")], [_text("ok"), _finish("stop")])
    real = probe_mod.probe_all
    monkeypatch.setattr(probe_mod, "probe_all", lambda entries, settings: real(entries, settings, src))
    out = tmp_path / "probe.json"
    assert probe_mod.main(["--models", "claude-haiku-5-5", "--json", str(out)]) == 0
    assert "Set client_tools=True for: claude-haiku-5-5" in capsys.readouterr().out
    assert json.loads(out.read_text())[0]["passed"] is True
