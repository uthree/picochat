"""Tool calling: the format layer (render/parse/registry) and the API
server's OpenAI tools/tool_calls mapping. Offline (ScriptedModel)."""

import json

from fastapi.testclient import TestClient

from picochat.inference.api import create_app
from picochat.model.tools import (
    ToolRegistry,
    parse_tool_calls,
    render_tool_call,
    render_tool_system,
    strip_tool_calls,
)
from picochat.tokenizer import TOOL_CALL_END, TOOL_CALL_START
from tests.test_generate import ByteTokenizer, ScriptedModel

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


# ---------------------------------------------------------------------------
# format layer
# ---------------------------------------------------------------------------
def test_render_tool_system_declares_tools():
    sys = render_tool_system([WEATHER_TOOL], "You are helpful.")
    assert "You are helpful." in sys
    assert "# Tools" in sys and "get_weather" in sys
    assert TOOL_CALL_START in sys and TOOL_CALL_END in sys
    # bare (unwrapped) specs are accepted too
    bare = render_tool_system([WEATHER_TOOL["function"]])
    assert "get_weather" in bare


def test_render_and_parse_roundtrip():
    call = render_tool_call("get_weather", {"city": "Tokyo"})
    assert call.startswith(TOOL_CALL_START) and call.endswith(TOOL_CALL_END)
    parsed = parse_tool_calls("sure! " + call)
    assert parsed == [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]


def test_parse_multiple_calls():
    text = render_tool_call("a", {"x": 1}) + " and " + render_tool_call("b", {"y": 2})
    assert parse_tool_calls(text) == [
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {"y": 2}},
    ]


def test_parse_drops_malformed_calls():
    assert parse_tool_calls(f"{TOOL_CALL_START}not json{TOOL_CALL_END}") == []
    # valid JSON but not a call object
    assert parse_tool_calls(f"{TOOL_CALL_START}[1,2,3]{TOOL_CALL_END}") == []
    # missing name
    assert (
        parse_tool_calls(f'{TOOL_CALL_START}{{"arguments":{{}}}}{TOOL_CALL_END}') == []
    )
    # a good call among bad ones survives
    good = render_tool_call("f", {})
    assert parse_tool_calls(f"{TOOL_CALL_START}bad{TOOL_CALL_END}" + good) == [
        {"name": "f", "arguments": {}}
    ]


def test_strip_tool_calls_leaves_prose():
    text = "Let me look. " + render_tool_call("f", {"a": 1}) + " done."
    assert strip_tool_calls(text) == "Let me look.  done."


def test_tool_registry_executes_and_guards():
    reg = ToolRegistry()
    reg.register(WEATHER_TOOL, lambda city: f"Sunny in {city}")
    assert reg.specs == [WEATHER_TOOL]
    assert reg.execute({"name": "get_weather", "arguments": {"city": "Osaka"}}) == (
        "Sunny in Osaka"
    )
    # non-string returns are JSON-encoded
    reg.register({"name": "add", "parameters": {}}, lambda a, b: a + b)
    assert reg.execute({"name": "add", "arguments": {"a": 2, "b": 3}}) == "5"
    # unknown tool and raising tool both degrade to an error string
    assert reg.execute({"name": "nope", "arguments": {}}).startswith("error: unknown")
    reg.register({"name": "boom", "parameters": {}}, lambda: 1 / 0)
    assert "ZeroDivisionError" in reg.execute({"name": "boom", "arguments": {}})


# ---------------------------------------------------------------------------
# API: OpenAI tools / tool_calls
# ---------------------------------------------------------------------------
def _client_scripting(reply: str, max_seq_len=4096):
    """A ScriptedModel that emits `reply` byte-for-byte (ByteTokenizer maps
    each byte to a token id), so the API sees exactly that assistant text."""
    tok = ByteTokenizer()
    script = list(reply.encode("utf-8")) + [tok.encode_single_token("<|im_end|>")]
    app = create_app(
        ScriptedModel(script),
        tok,
        device="cpu",
        max_seq_len=max_seq_len,
        model_id="t",
    )
    return TestClient(app)


def test_api_returns_tool_calls():
    reply = "I'll check. " + render_tool_call("get_weather", {"city": "Tokyo"})
    client = _client_scripting(reply)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "t",
            "messages": [{"role": "user", "content": "weather in Tokyo?"}],
            "tools": [WEATHER_TOOL],
            "max_tokens": 400,
        },
    )
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Tokyo"}
    # the natural-language part is preserved as content
    assert "check" in choice["message"]["content"]


def test_api_plain_reply_when_no_tool_call_emitted():
    client = _client_scripting("It is sunny today.")
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "t",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [WEATHER_TOOL],
            "max_tokens": 400,
        },
    )
    body = r.json()["choices"][0]
    assert body["finish_reason"] == "stop"
    assert "tool_calls" not in body["message"]
    assert body["message"]["content"] == "It is sunny today."


def test_api_tool_result_turn_roundtrips():
    # A follow-up request carrying the prior assistant tool_calls + a tool
    # result turn must be accepted (renders back to ChatML without error).
    client = _client_scripting("It is sunny in Tokyo.")
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "t",
            "messages": [
                {"role": "user", "content": "weather in Tokyo?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Tokyo"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 24C"},
            ],
            "tools": [WEATHER_TOOL],
            "max_tokens": 400,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "It is sunny in Tokyo."


def test_api_tools_force_non_streaming():
    reply = render_tool_call("get_weather", {"city": "Tokyo"})
    client = _client_scripting(reply)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "t",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [WEATHER_TOOL],
            "stream": True,  # ignored when tools are present
            "max_tokens": 400,
        },
    )
    # a JSON body (not an SSE stream) comes back
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"
