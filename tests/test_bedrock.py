"""Bedrock backend: OpenAI <-> Converse translation + routing (GEN-003).

No network — the boto3 client is monkeypatched. Response shapes mirror the live
Converse smoke test run 2026-07-15 against oasis-dev.
"""

import json

from oasis_generation import bedrock, catalog


def _inference(public_id: str):
    """The catalog's inference policy for a model (ties these tests to the real
    per-model config so drift is caught)."""
    return next(m for m in catalog.CATALOG if m.public_id == public_id).inference


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
    out = bedrock.converse_to_openai(_CONVERSE_RESP, "claude-sonnet-5", "chatcmpl-x", 1234)
    assert out["object"] == "chat.completion"
    assert out["model"] == "claude-sonnet-5"
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
    out = bedrock.converse_to_openai(resp, "claude-opus-4-8", "id", 0)
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
    chunks = asyncio.run(_collect(bedrock.stream("m", body, "us-east-1", "claude-sonnet-5")))
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


# ---- inference policy: thinking / effort / temperature (param unification) ---


def test_adaptive_claude_injects_thinking_and_strips_sampling():
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,  # must be dropped — opus-4-8 400s on it
        "top_p": 0.9,
    }
    kw = bedrock.openai_to_converse(body, _inference("claude-opus-4-8"))
    amrf = kw["additionalModelRequestFields"]
    assert amrf["thinking"] == {"type": "adaptive"}
    assert amrf["output_config"] == {"effort": "high"}  # balanced default
    assert "temperature" not in kw["inferenceConfig"]
    assert "topP" not in kw["inferenceConfig"]
    assert kw["inferenceConfig"]["maxTokens"] == 16000  # balanced profile


def test_profile_deep_uses_top_effort_and_max_tokens():
    body = {"messages": [{"role": "user", "content": "hi"}], "profile": "deep"}
    opus = bedrock.openai_to_converse(body, _inference("claude-opus-4-8"))
    assert opus["additionalModelRequestFields"]["output_config"] == {"effort": "xhigh"}
    assert opus["inferenceConfig"]["maxTokens"] == 32000
    # Sonnet 5 caps deep effort at high.
    sonnet = bedrock.openai_to_converse(body, _inference("claude-sonnet-5"))
    assert sonnet["additionalModelRequestFields"]["output_config"] == {"effort": "high"}


def test_reasoning_effort_override_beats_profile_default():
    inf = _inference("claude-opus-4-8")
    low = bedrock.openai_to_converse(
        {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low"}, inf
    )
    assert low["additionalModelRequestFields"]["output_config"] == {"effort": "low"}
    # OpenAI "minimal" maps to Anthropic "low".
    minimal = bedrock.openai_to_converse(
        {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "minimal"}, inf
    )
    assert minimal["additionalModelRequestFields"]["output_config"] == {"effort": "low"}


def test_explicit_max_tokens_overrides_profile():
    kw = bedrock.openai_to_converse(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 500},
        _inference("claude-opus-4-8"),
    )
    assert kw["inferenceConfig"]["maxTokens"] == 500


def test_native_model_injects_nothing_and_keeps_temperature():
    kw = bedrock.openai_to_converse(
        {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.3},
        _inference("glm-5"),
    )
    assert "additionalModelRequestFields" not in kw  # native thinking, no injection
    assert kw["inferenceConfig"]["temperature"] == 0.3  # glm-5 allows sampling
    assert kw["inferenceConfig"]["maxTokens"] == 8192  # native balanced default


def test_no_inference_defaults_is_legacy_passthrough():
    kw = bedrock.openai_to_converse(
        {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.5}
    )  # defaults=None
    assert "additionalModelRequestFields" not in kw
    assert kw["inferenceConfig"]["temperature"] == 0.5
    assert kw["inferenceConfig"]["maxTokens"] == bedrock._DEFAULT_MAX_TOKENS


# ---- image input (Converse multimodal) --------------------------------------


def test_image_data_uri_becomes_converse_image_block():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]}
    blocks = bedrock.openai_to_converse(body)["messages"][0]["content"]
    assert blocks[0] == {"text": "what is this?"}
    assert blocks[1] == {"image": {"format": "png", "source": {"bytes": b"\x00\x00\x00"}}}


def test_jpeg_and_bare_string_image_url():
    # image_url as a bare string (not {"url":...}); jpg normalizes to jpeg.
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": "data:image/jpg;base64,AAAA"},
    ]}]}
    blocks = bedrock.openai_to_converse(body)["messages"][0]["content"]
    assert blocks == [{"image": {"format": "jpeg", "source": {"bytes": b"\x00\x00\x00"}}}]


def test_http_image_url_dropped_no_ssrf():
    # Remote URLs are NOT fetched by the gateway (SSRF-safe) -> only text survives.
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    ]}]}
    assert bedrock.openai_to_converse(body)["messages"][0]["content"] == [{"text": "look"}]


def test_malformed_or_missing_images_dropped():
    assert bedrock._image_block("data:image/png;base64,!!!!") is None
    assert bedrock._image_block("not a data uri") is None
    assert bedrock._image_block(None) is None


# ---- Prompt caching (2026-10-01) -------------------------------------------

import pytest


@pytest.fixture(autouse=True)
def _fresh_prefix_memory():
    bedrock._prefix_seen.clear()
    yield
    bedrock._prefix_seen.clear()


CP = {"cachePoint": {"type": "default"}}


def _cache_body(system="You are a bot.", last="second"):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": last},
        ],
        "tools": [{"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}],
    }


def _conv(body, **kw):
    return bedrock.openai_to_converse(body, prompt_cache=True, cache_key="m", **kw)


def test_first_sighting_places_no_cache_points():
    assert "cachePoint" not in json.dumps(_conv(_cache_body()))


def test_recurring_prefix_gets_tools_system_and_rolling_points():
    _conv(_cache_body())
    kw = _conv(_cache_body(last="third"))
    assert kw["toolConfig"]["tools"][-1] == CP
    assert kw["system"][-1] == CP
    assert kw["messages"][-1]["content"][-1] == CP
    assert json.dumps(kw).count("cachePoint") == 3


def test_changing_system_prompt_never_caches_system_or_history():
    # The reviewer-judge shape: a per-call nonce inside the system prompt.
    _conv(_cache_body(system="nonce 1"))
    kw = _conv(_cache_body(system="nonce 2"))
    assert kw["toolConfig"]["tools"][-1] == CP          # tools still recur
    assert "cachePoint" not in json.dumps(kw["system"])
    assert "cachePoint" not in json.dumps(kw["messages"])


def test_one_shot_call_gets_no_rolling_point():
    body = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "q1"}]}
    _conv(body)
    kw = _conv({"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "q2"}]})
    assert kw["system"][-1] == CP
    assert "cachePoint" not in json.dumps(kw["messages"])


def test_prefix_memory_expires_after_ttl():
    kw = bedrock.openai_to_converse(_cache_body(), prompt_cache=True, cache_key="m")
    bedrock._prefix_seen.update({k: v - 301 for k, v in bedrock._prefix_seen.items()})
    kw = _conv(_cache_body())
    assert "cachePoint" not in json.dumps(kw)


def test_prefix_memory_is_per_model():
    _conv(_cache_body())
    kw = bedrock.openai_to_converse(_cache_body(), prompt_cache=True, cache_key="other")
    assert "cachePoint" not in json.dumps(kw)


def test_no_cache_points_when_disabled():
    bedrock.openai_to_converse(_cache_body())
    assert "cachePoint" not in json.dumps(bedrock.openai_to_converse(_cache_body()))


def test_no_message_cache_point_after_assistant_prefill():
    body = _cache_body()
    body["messages"].append({"role": "assistant", "content": "prefill"})
    _conv(body)
    kw = _conv(body)
    assert "cachePoint" not in json.dumps(kw["messages"])


def test_cache_point_follows_a_tool_result_turn():
    body = _cache_body()
    body["messages"][-1] = {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
    }
    body["messages"].append({"role": "tool", "tool_call_id": "c1", "content": "result"})
    _conv(body)
    kw = _conv(body)
    last = kw["messages"][-1]["content"]
    assert "toolResult" in last[0] and last[-1] == CP


def test_supports_prompt_cache_claude_only(monkeypatch):
    monkeypatch.delenv("OASIS_GENERATION_PROMPT_CACHE", raising=False)
    assert bedrock.supports_prompt_cache("us.anthropic.claude-sonnet-5")
    for other in ("us.amazon.nova-micro-v1:0", "us.openai.gpt-6-astra", "zai.glm-5",
                  "us.meta.llama3-3-70b-instruct-v1:0"):
        assert not bedrock.supports_prompt_cache(other)
    monkeypatch.setenv("OASIS_GENERATION_PROMPT_CACHE", "0")
    assert not bedrock.supports_prompt_cache("us.anthropic.claude-sonnet-5")
