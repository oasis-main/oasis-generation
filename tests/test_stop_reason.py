"""Stop reasons reach the client, the log and the usage ledger (2026-10-18).

On 2026-10-17 Nimbus's reviewer judge got empty replies and denied 4 calls.
Bedrock had returned stopReason "content_filtered" with 0 output tokens; the
gateway recorded that nowhere. These tests pin the raw stop reason to the
response, the WARNING log line and the /usage ledger row, streamed or not.
"""

import asyncio
import json
import logging

import botocore.session
from fastapi.testclient import TestClient

from oasis_generation import bedrock, main, mantle
from oasis_generation.config import get_settings

client = TestClient(main.app)


def setup_function() -> None:
    get_settings.cache_clear()


class _FakeStreamClient:
    def __init__(self, events):
        self._events = events

    def converse_stream(self, **_kwargs):
        return {"stream": iter(self._events)}


def _filtered_events():
    return [
        {"messageStart": {"role": "assistant"}},
        {"messageStop": {"stopReason": "content_filtered"}},
        {"metadata": {"usage": {"inputTokens": 13413, "outputTokens": 0, "totalTokens": 13413}}},
    ]


async def _collect(agen):
    return [chunk async for chunk in agen]


def _frames(chunks):
    out = []
    for c in chunks:
        for line in c.decode().splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                out.append(json.loads(line[len("data: "):]))
    return out


# ---- Converse mapping --------------------------------------------------------


def test_every_converse_stop_reason_is_mapped_explicitly():
    enum = botocore.session.get_session().get_service_model(
        "bedrock-runtime").shape_for("StopReason").enum
    missing = [r for r in enum if r not in bedrock._FINISH_REASON]
    assert not missing, f"unmapped Converse stopReasons: {missing}"
    assert bedrock._FINISH_REASON["refusal"] == "content_filter"
    assert bedrock._FINISH_REASON["content_filtered"] == "content_filter"
    assert bedrock._FINISH_REASON["model_context_window_exceeded"] == "length"


def test_unknown_stop_reason_is_stop_and_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="oasis_generation.bedrock"):
        assert bedrock._finish_reason("brand_new_reason") == "stop"
    assert "brand_new_reason" in caplog.text


def test_non_streamed_carries_raw_stop_reason():
    resp = {
        "output": {"message": {"role": "assistant", "content": []}},
        "stopReason": "content_filtered",
        "usage": {"inputTokens": 10, "outputTokens": 0, "totalTokens": 10},
    }
    out = bedrock.converse_to_openai(resp, "claude-sonnet-5", "id", 0)
    choice = out["choices"][0]
    assert choice["finish_reason"] == "content_filter"
    assert choice["upstream_stop_reason"] == "content_filtered"


def test_complete_logs_filtered_empty_turn(monkeypatch, caplog):
    class Fake:
        def converse(self, **_kw):
            return {
                "output": {"message": {"role": "assistant", "content": []}},
                "stopReason": "refusal",
                "usage": {"inputTokens": 10, "outputTokens": 0, "totalTokens": 10},
            }

    monkeypatch.setattr(bedrock, "_client", lambda region: Fake())
    body = {"messages": [{"role": "user", "content": "x"}]}
    with caplog.at_level(logging.WARNING, logger="oasis_generation.bedrock"):
        out = asyncio.run(bedrock.complete("m", body, "us-east-1", "claude-sonnet-5"))
    assert out["choices"][0]["finish_reason"] == "content_filter"
    assert "stop_reason=refusal output_tokens=0" in caplog.text


def test_stream_fills_outcome_and_final_frame(monkeypatch, caplog):
    monkeypatch.setattr(bedrock, "_client", lambda region: _FakeStreamClient(_filtered_events()))
    outcome: dict = {}
    # No include_usage: the ledger still needs the usage, the client does not.
    body = {"messages": [{"role": "user", "content": "x"}], "stream": True}
    with caplog.at_level(logging.WARNING, logger="oasis_generation.bedrock"):
        frames = _frames(asyncio.run(_collect(
            bedrock.stream("m", body, "us-east-1", "claude-sonnet-5", outcome=outcome))))
    final = frames[-1]
    assert final["choices"][0]["finish_reason"] == "content_filter"
    assert final["choices"][0]["upstream_stop_reason"] == "content_filtered"
    assert "usage" not in final
    assert outcome["stop_reason"] == "content_filtered"
    assert outcome["finish_reason"] == "content_filter"
    assert outcome["usage"]["completion_tokens"] == 0
    assert "stop_reason=content_filtered output_tokens=0" in caplog.text


def test_normal_stream_is_not_a_warning(monkeypatch, caplog):
    events = [
        {"contentBlockDelta": {"delta": {"text": "ok"}, "contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4}}},
    ]
    monkeypatch.setattr(bedrock, "_client", lambda region: _FakeStreamClient(events))
    body = {"messages": [{"role": "user", "content": "x"}], "stream": True}
    with caplog.at_level(logging.WARNING, logger="oasis_generation.bedrock"):
        asyncio.run(_collect(bedrock.stream("m", body, "us-east-1", "claude-sonnet-5")))
    assert "bedrock done" not in caplog.text


# ---- Mantle (Responses) mapping ---------------------------------------------


def test_mantle_refusal_is_content_filter():
    resp = {
        "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}],
        "status": "completed",
        "usage": {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5},
    }
    choice = mantle.responses_to_openai(resp, "gpt-5.6-sol", "id", 0)["choices"][0]
    assert choice["finish_reason"] == "content_filter"
    assert choice["upstream_stop_reason"] == "refusal"


def test_mantle_incomplete_content_filter():
    resp = {
        "output": [],
        "status": "incomplete",
        "incomplete_details": {"reason": "content_filter"},
        "usage": {},
    }
    choice = mantle.responses_to_openai(resp, "gpt-5.6-sol", "id", 0)["choices"][0]
    assert choice["finish_reason"] == "content_filter"
    assert choice["upstream_stop_reason"] == "incomplete:content_filter"


def test_mantle_stream_fills_outcome(monkeypatch):
    async def fake_complete(model_id, body, region, public_id, defaults=None):
        return mantle.responses_to_openai({
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
            "status": "completed",
            "usage": {"input_tokens": 50, "output_tokens": 2, "total_tokens": 52},
        }, public_id, "chatcmpl-x", 0)

    monkeypatch.setattr(mantle, "complete", fake_complete)
    outcome: dict = {}
    body = {"messages": [{"role": "user", "content": "x"}], "stream": True}
    frames = _frames(asyncio.run(_collect(
        mantle.stream("m", body, "us-east-1", "gpt-5.6-sol", outcome=outcome))))
    assert frames[-1]["choices"][0]["upstream_stop_reason"] == "completed"
    assert outcome == {
        "stop_reason": "completed",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 50, "completion_tokens": 2, "total_tokens": 52,
                  "prompt_tokens_details": {"cached_tokens": 0}},
    }


# ---- Usage ledger -------------------------------------------------------------


def _ledger(tmp_path):
    path = tmp_path / "usage" / "requests.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_ledger_streamed_row_has_stop_reason_and_tokens(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "USAGE_LOG", str(tmp_path / "usage" / "requests.jsonl"))
    monkeypatch.setattr(bedrock, "_client", lambda region: _FakeStreamClient(_filtered_events()))
    resp = client.post("/v1/chat/completions", json={
        "model": "claude-sonnet-5", "stream": True,
        "messages": [{"role": "user", "content": "x"}],
    })
    assert resp.status_code == 200
    rows = _ledger(tmp_path)
    assert len(rows) == 1  # one row per call, written when the stream ends
    row = rows[0]
    assert row["streamed"] is True
    assert row["stop_reason"] == "content_filtered"
    assert row["finish_reason"] == "content_filter"
    assert row["prompt_tokens"] == 13413
    assert row["completion_tokens"] == 0


def test_ledger_non_streamed_row_has_stop_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "USAGE_LOG", str(tmp_path / "usage" / "requests.jsonl"))

    class Fake:
        def converse(self, **_kw):
            return {
                "output": {"message": {"role": "assistant", "content": [{"text": "pong"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 4, "outputTokens": 1, "totalTokens": 5},
            }

    monkeypatch.setattr(bedrock, "_client", lambda region: Fake())
    resp = client.post("/v1/chat/completions", json={
        "model": "claude-sonnet-5", "messages": [{"role": "user", "content": "ping"}],
    })
    assert resp.status_code == 200
    row = _ledger(tmp_path)[0]
    assert row["streamed"] is False
    assert row["stop_reason"] == "end_turn"
    assert row["finish_reason"] == "stop"
    assert row["completion_tokens"] == 1
