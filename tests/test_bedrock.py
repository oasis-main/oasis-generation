"""Bedrock backend: OpenAI <-> Converse translation + routing (GEN-003).

No network — the boto3 client is monkeypatched. Response shapes mirror the live
Converse smoke test run 2026-07-15 against oasis-dev.
"""

import json

from oasis_generation import bedrock


# ---- OpenAI -> Converse -----------------------------------------------------


def test_system_split_and_role_mapping():
    body = {
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ],
        "max_tokens": 32,
    }
    kw = bedrock.openai_to_converse(body)
    assert kw["system"] == [{"text": "be terse"}]
    assert [m["role"] for m in kw["messages"]] == ["user", "assistant", "user"]
    assert kw["inferenceConfig"]["maxTokens"] == 32


def test_consecutive_same_role_merged():
    body = {
        "messages": [
            {"role": "user", "content": "part 1"},
            {"role": "user", "content": "part 2"},
        ]
    }
    kw = bedrock.openai_to_converse(body)
    # Converse requires alternating roles; the two user turns collapse to one.
    assert len(kw["messages"]) == 1
    assert kw["messages"][0]["content"] == [{"text": "part 1"}, {"text": "part 2"}]
    # maxTokens defaulted when the caller omits it.
    assert kw["inferenceConfig"]["maxTokens"] == bedrock._DEFAULT_MAX_TOKENS


def test_tools_mapped_to_toolconfig():
    body = {
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
    }
    kw = bedrock.openai_to_converse(body)
    spec = kw["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["name"] == "get_weather"
    assert spec["description"] == "Look up the weather"
    # JSON Schema nests one level down under "json".
    assert spec["inputSchema"]["json"]["properties"] == {"city": {"type": "string"}}
    # "auto" (default) omits toolChoice for cross-model compatibility.
    assert "toolChoice" not in kw["toolConfig"]


def test_tool_choice_mappings():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    base_body = {"messages": [{"role": "user", "content": "x"}], "tools": tools}

    assert bedrock.openai_to_converse({**base_body, "tool_choice": "required"})[
        "toolConfig"
    ]["toolChoice"] == {"any": {}}

    named = bedrock.openai_to_converse(
        {**base_body, "tool_choice": {"type": "function", "function": {"name": "f"}}}
    )
    assert named["toolConfig"]["toolChoice"] == {"tool": {"name": "f"}}

    # "none" drops tools entirely -> no toolConfig at all.
    assert "toolConfig" not in bedrock.openai_to_converse({**base_body, "tool_choice": "none"})


def test_assistant_tool_calls_and_results_roundtrip():
    body = {
        "messages": [
            {"role": "user", "content": "check my inbox"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_thread", "arguments": '{"id":"42"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Reply from Brian"},
            {"role": "user", "content": "thanks"},
        ]
    }
    kw = bedrock.openai_to_converse(body)
    roles = [m["role"] for m in kw["messages"]]
    # assistant turn carries the toolUse; the tool result rides in a user turn,
    # and the trailing user "thanks" merges into that same user turn.
    assert roles == ["user", "assistant", "user"]

    assistant = kw["messages"][1]["content"]
    assert assistant == [
        {"toolUse": {"toolUseId": "call_1", "name": "get_thread", "input": {"id": "42"}}}
    ]

    tool_turn = kw["messages"][2]["content"]
    assert tool_turn[0]["toolResult"]["toolUseId"] == "call_1"
    assert tool_turn[0]["toolResult"]["content"] == [{"text": "Reply from Brian"}]
    assert tool_turn[1] == {"text": "thanks"}


def test_list_content_and_inference_params():
    body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}],
        "temperature": 0.5,
        "top_p": 0.9,
        "stop": "END",
    }
    kw = bedrock.openai_to_converse(body)
    assert kw["messages"][0]["content"] == [{"text": "ab"}]
    assert kw["inferenceConfig"]["temperature"] == 0.5
    assert kw["inferenceConfig"]["topP"] == 0.9
    assert kw["inferenceConfig"]["stopSequences"] == ["END"]
    # Absent system when there are no system messages.
    assert "system" not in kw


# ---- Converse -> OpenAI -----------------------------------------------------

_CONVERSE_RESP = {
    "output": {"message": {"role": "assistant", "content": [{"text": "Hi! 👋"}]}},
    "stopReason": "max_tokens",
    "usage": {"inputTokens": 9, "outputTokens": 5, "totalTokens": 14,
              "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0},
}


def test_converse_to_openai_shape():
    out = bedrock.converse_to_openai(_CONVERSE_RESP, "claude-haiku-4-5", "chatcmpl-x", 1234)
    assert out["object"] == "chat.completion"
    assert out["model"] == "claude-haiku-4-5"
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "Hi! 👋"}
    assert out["choices"][0]["finish_reason"] == "length"  # max_tokens -> length
    assert out["usage"]["prompt_tokens"] == 9
    assert out["usage"]["completion_tokens"] == 5


def test_reasoning_content_surfaced():
    resp = {
        "output": {"message": {"content": [
            {"reasoningContent": {"reasoningText": {"text": "thinking"}}},
            {"text": "answer"},
        ]}},
        "stopReason": "end_turn",
        "usage": {},
    }
    out = bedrock.converse_to_openai(resp, "glm-5", "id", 0)
    msg = out["choices"][0]["message"]
    assert msg["content"] == "answer"
    assert msg["reasoning_content"] == "thinking"
    assert out["choices"][0]["finish_reason"] == "stop"


def test_converse_to_openai_tool_calls():
    resp = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "tool_9", "name": "get_thread", "input": {"id": "42"}}}
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {},
    }
    out = bedrock.converse_to_openai(resp, "claude-sonnet-4-6", "id", 0)
    msg = out["choices"][0]["message"]
    # Pure tool-call turn: content null, one OpenAI-shaped tool_call.
    assert msg["content"] is None
    assert msg["tool_calls"] == [
        {
            "id": "tool_9",
            "type": "function",
            "function": {"name": "get_thread", "arguments": '{"id": "42"}'},
        }
    ]
    assert out["choices"][0]["finish_reason"] == "tool_calls"


# ---- streaming translation --------------------------------------------------


class _FakeStreamClient:
    """Mimics boto3 bedrock-runtime.converse_stream: a dict with a sync
    iterable under "stream"."""

    def __init__(self, events):
        self._events = events

    def converse_stream(self, **_kwargs):
        return {"stream": iter(self._events)}


async def _collect(agen):
    return [chunk async for chunk in agen]


def _parse_sse(chunks: list[bytes]) -> list:
    out = []
    for c in chunks:
        for line in c.decode().splitlines():
            if line.startswith("data: "):
                out.append(line[len("data: ") :])
    return out


def test_stream_emits_openai_chunks(monkeypatch):
    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Hel"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "lo"}, "contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5}}},
    ]
    monkeypatch.setattr(bedrock, "_client", lambda region: _FakeStreamClient(events))

    import asyncio

    body = {"messages": [{"role": "user", "content": "hi"}], "stream": True,
            "stream_options": {"include_usage": True}}
    chunks = asyncio.run(_collect(bedrock.stream("m", body, "us-east-1", "glm-5")))
    datas = _parse_sse(chunks)
    assert datas[-1] == "[DONE]"
    texts = "".join(
        json.loads(d)["choices"][0]["delta"].get("content", "")
        for d in datas
        if d != "[DONE]"
    )
    assert texts == "Hello"
    # first frame opens the assistant role
    assert json.loads(datas[0])["choices"][0]["delta"] == {"role": "assistant"}
    # a usage frame is present because include_usage was requested
    assert any("usage" in json.loads(d) for d in datas if d != "[DONE]")


def test_stream_tool_calls(monkeypatch):
    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {
            "start": {"toolUse": {"toolUseId": "tool_1", "name": "get_thread"}},
            "contentBlockIndex": 0,
        }},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"id":'}}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '"42"}'}}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]
    monkeypatch.setattr(bedrock, "_client", lambda region: _FakeStreamClient(events))

    import asyncio

    body = {"messages": [{"role": "user", "content": "check inbox"}], "stream": True}
    chunks = asyncio.run(_collect(bedrock.stream("m", body, "us-east-1", "claude-sonnet-4-6")))
    frames = [json.loads(d) for d in _parse_sse(chunks) if d != "[DONE]"]

    # Collect every tool_calls delta across the stream.
    tc_deltas = [
        tc
        for f in frames
        for tc in f["choices"][0]["delta"].get("tool_calls", [])
    ]
    # The opening frame carries index/id/name; deltas carry argument fragments.
    assert tc_deltas[0]["index"] == 0
    assert tc_deltas[0]["id"] == "tool_1"
    assert tc_deltas[0]["function"]["name"] == "get_thread"
    args = "".join(tc["function"].get("arguments", "") for tc in tc_deltas)
    assert args == '{"id":"42"}'
    # Terminal frame maps tool_use -> tool_calls.
    assert frames[-1]["choices"][0]["finish_reason"] == "tool_calls"
