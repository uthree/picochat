import json

from fastapi.testclient import TestClient

from picochat.api import create_app
from picochat.engine import SamplingConfig
from picochat.tokenizer import IM_END
from tests.test_generate import ByteTokenizer, ScriptedModel


def make_client(
    script, max_seq_len=4096, default_sampling=None, model_id="picochat-test"
):
    tok = ByteTokenizer()
    model = ScriptedModel(script)
    app = create_app(
        model,
        tok,
        device="cpu",
        max_seq_len=max_seq_len,
        model_id=model_id,
        default_sampling=default_sampling,
    )
    return TestClient(app), tok


def chat_request(**overrides):
    body = {"model": "picochat-test", "messages": [{"role": "user", "content": "hi"}]}
    body.update(overrides)
    return body


def test_list_models_returns_configured_id():
    client, _ = make_client([65])
    res = client.get("/v1/models")
    assert res.status_code == 200
    assert res.json()["data"][0]["id"] == "picochat-test"


def test_chat_completions_non_streaming_returns_content_and_stop_reason():
    tok = ByteTokenizer()
    im_end = tok.encode_single_token(IM_END)
    client, tok = make_client(
        [ord("h"), ord("i"), im_end], default_sampling=SamplingConfig(temperature=0.0)
    )
    res = client.post("/v1/chat/completions", json=chat_request())
    assert res.status_code == 200
    body = res.json()
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["model"] == "picochat-test"
    assert body["usage"]["completion_tokens"] == 2
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 2


def test_chat_completions_finish_reason_length_when_budget_exhausted():
    client, _ = make_client(
        [ord("a")], default_sampling=SamplingConfig(temperature=0.0, max_new_tokens=3)
    )
    res = client.post("/v1/chat/completions", json=chat_request(max_tokens=3))
    body = res.json()
    assert body["choices"][0]["message"]["content"] == "aaa"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 3


def test_chat_completions_per_request_sampling_overrides_default():
    # default max_new_tokens=1 would cut generation short; the request's
    # max_tokens should override it and let the stop token end things instead
    tok = ByteTokenizer()
    im_end = tok.encode_single_token(IM_END)
    client, _ = make_client(
        [ord("h"), ord("i"), im_end],
        default_sampling=SamplingConfig(temperature=0.0, max_new_tokens=1),
    )
    res = client.post("/v1/chat/completions", json=chat_request(max_tokens=10))
    body = res.json()
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_rejects_empty_messages():
    client, _ = make_client([65])
    res = client.post(
        "/v1/chat/completions", json={"model": "picochat-test", "messages": []}
    )
    assert res.status_code == 400
    assert res.json()["detail"]["error"]["code"] == "empty_messages"


def test_chat_completions_rejects_prompt_that_fills_context():
    client, _ = make_client([65], max_seq_len=4)  # the ChatML prompt alone exceeds this
    res = client.post("/v1/chat/completions", json=chat_request())
    assert res.status_code == 400
    assert res.json()["detail"]["error"]["code"] == "context_length_exceeded"


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(json.loads(payload))
    return events


def test_chat_completions_streaming_reassembles_content_and_ends_with_done():
    tok = ByteTokenizer()
    im_end = tok.encode_single_token(IM_END)
    client, _ = make_client(
        [ord("h"), ord("i"), im_end], default_sampling=SamplingConfig(temperature=0.0)
    )
    res = client.post("/v1/chat/completions", json=chat_request(stream=True))
    assert res.status_code == 200
    events = _parse_sse(res.text)
    assert events[-1] == "[DONE]"
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    content = "".join(
        e["choices"][0]["delta"].get("content", "")
        for e in events[:-1]
        if isinstance(e, dict)
    )
    assert content == "hi"
    assert events[-2]["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_streaming_buffers_multi_byte_utf8_split_across_tokens():
    # "é" is 0xC3 0xA9 in UTF-8, delivered as two separate tokens: each chunk
    # must only ever contain complete characters, never a lone partial byte.
    tok = ByteTokenizer()
    im_end = tok.encode_single_token(IM_END)
    client, _ = make_client(
        [0xC3, 0xA9, im_end], default_sampling=SamplingConfig(temperature=0.0)
    )
    res = client.post("/v1/chat/completions", json=chat_request(stream=True))
    events = _parse_sse(res.text)
    contents = [
        e["choices"][0]["delta"]["content"]
        for e in events
        if isinstance(e, dict) and "content" in e["choices"][0]["delta"]
    ]
    assert contents == [
        "é"
    ]  # buffered into a single complete chunk, not two partial ones
