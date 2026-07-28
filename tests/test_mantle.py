"""Bedrock-mantle backend: OpenAI <-> Responses API translation (GPT-5.6-sol).

No network — pure translation. Response shapes mirror the live probe run
2026-07-24 against the bedrock-mantle endpoint.
"""

from oasis_generation import catalog, mantle


def _inf(public_id: str):
    return next(m for m in catalog.CATALOG if m.public_id == public_id).inference


def test_openai_to_responses_translation():
    body = {
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
    }
    p = mantle.openai_to_responses("openai.gpt-5.6-sol", body, _inf("gpt-5.6-sol"))
    assert p["model"] == "openai.gpt-5.6-sol"
    assert p["instructions"] == "be terse"
    assert p["store"] is False  # privacy: prompts not persisted
    # input items carry the right role + part type (input_text vs output_text)
    assert [(i["role"], i["content"][0]["type"], i["content"][0]["text"]) for i in p["input"]] == [
        ("user", "input_text", "hi"),
        ("assistant", "output_text", "hello"),
        ("user", "input_text", "bye"),
    ]
    # balanced default profile -> effort medium, verbosity medium, max 16000
    assert p["reasoning"] == {"effort": "medium"}
    assert p["text"] == {"verbosity": "medium"}
    assert p["max_output_tokens"] == 16000
    # gpt-5.6-sol rejects temperature -> never forwarded even if the caller sends it
    p_temp = mantle.openai_to_responses(
        "m", {"messages": [{"role": "user", "content": "x"}], "temperature": 0.7}, _inf("gpt-5.6-sol")
    )
    assert "temperature" not in p_temp


def test_profile_and_effort_override():
    inf = _inf("gpt-5.6-sol")
    deep = mantle.openai_to_responses("m", {"messages": [{"role": "user", "content": "x"}], "profile": "deep"}, inf)
    assert deep["reasoning"] == {"effort": "high"}
    assert deep["text"] == {"verbosity": "high"}
    assert deep["max_output_tokens"] == 32000
    # reasoning_effort override maps onto the Responses ladder (max -> xhigh)
    over = mantle.openai_to_responses(
        "m", {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "max"}, inf
    )
    assert over["reasoning"] == {"effort": "xhigh"}


def test_responses_effort_mapping():
    assert mantle._responses_effort("minimal") == "low"  # openai wire has no 'minimal' here
    assert mantle._responses_effort("max") == "xhigh"
    assert mantle._responses_effort("high") == "high"
    assert mantle._responses_effort("bogus") == "medium"
    assert mantle._responses_effort(None) is None


def test_responses_to_openai_shape():
    resp = {
        "output": [
            {"type": "reasoning", "content": []},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "42"}]},
        ],
        "status": "completed",
        "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
    }
    out = mantle.responses_to_openai(resp, "gpt-5.6-sol", "chatcmpl-x", 1234)
    assert out["object"] == "chat.completion"
    assert out["model"] == "gpt-5.6-sol"
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "42"}
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}


def test_incomplete_maps_to_length():
    resp = {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "partial"}]}],
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "usage": {},
    }
    out = mantle.responses_to_openai(resp, "m", "id", 0)
    assert out["choices"][0]["finish_reason"] == "length"


def test_no_user_content_gives_empty_string_input():
    p = mantle.openai_to_responses("m", {"messages": [{"role": "system", "content": "sys only"}]}, _inf("gpt-5.6-sol"))
    assert p["input"] == ""
    assert p["instructions"] == "sys only"


# ---- tools ------------------------------------------------------------------


def test_tools_translate_to_flat_responses_schema():
    body = {
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        "tool_choice": "required",
    }
    p = mantle.openai_to_responses("m", body, _inf("gpt-5.6-sol"))
    # Responses tools are FLAT: {type, name, description, parameters} (no nested "function").
    assert p["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Look up weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    assert p["tool_choice"] == "required"


def test_tool_choice_named_and_none():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    named = mantle.openai_to_responses(
        "m",
        {"messages": [{"role": "user", "content": "x"}], "tools": tools,
         "tool_choice": {"type": "function", "function": {"name": "f"}}},
        None,
    )
    assert named["tool_choice"] == {"type": "function", "name": "f"}
    # tool_choice "none" drops tools entirely.
    off = mantle.openai_to_responses(
        "m", {"messages": [{"role": "user", "content": "x"}], "tools": tools, "tool_choice": "none"}, None
    )
    assert "tools" not in off


def test_assistant_tool_calls_and_tool_result_roundtrip_to_responses():
    body = {
        "messages": [
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "18C sunny"},
        ]
    }
    items = mantle.openai_to_responses("m", body, None)["input"]
    assert items[0] == {"type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": "weather in Paris?"}]}
    assert items[1] == {"type": "function_call", "call_id": "call_1",
                        "name": "get_weather", "arguments": '{"city":"Paris"}'}
    assert items[2] == {"type": "function_call_output", "call_id": "call_1", "output": "18C sunny"}


def test_response_function_call_becomes_openai_tool_calls():
    resp = {
        "output": [
            {"type": "function_call", "call_id": "call_9", "id": "fc_1",
             "name": "get_weather", "arguments": '{"city":"Paris"}', "status": "completed"}
        ],
        "status": "completed",
        "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
    }
    out = mantle.responses_to_openai(resp, "gpt-5.6-sol", "id", 0)
    msg = out["choices"][0]["message"]
    assert msg["content"] is None  # pure tool-call turn
    assert msg["tool_calls"] == [
        {"id": "call_9", "type": "function",
         "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
    ]
    assert out["choices"][0]["finish_reason"] == "tool_calls"


# ---- images -----------------------------------------------------------------


def test_image_url_content_becomes_input_image():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ]
    }
    items = mantle.openai_to_responses("m", body, _inf("gpt-5.6-sol"))["input"]
    assert items[0]["content"] == [
        {"type": "input_text", "text": "what is this?"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
    ]
