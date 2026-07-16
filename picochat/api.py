"""OpenAI-compatible Chat Completions API (see scripts/api.py).

Implements the two endpoints an OpenAI-compatible client needs to talk to a
picochat checkpoint -- GET /v1/models and POST /v1/chat/completions, either
streamed as server-sent events or returned as one JSON response -- which is
enough for tools like OpenCode's `@ai-sdk/openai-compatible` provider.
Request `messages` are plain ChatML turns: picochat.tokenizer.render_chat_prompt
already expects exactly this {"role", "content"} shape, so no translation
layer is needed beyond Pydantic's request parsing.

Generation runs one request at a time (see create_app's generation_lock):
TransformerLM.decode() is a plain function over an explicit KV cache with no
shared mutable state, so concurrent calls would be memory-safe, but
serializing keeps GPU memory bounded and avoids two requests interleaving
their decode steps on one accelerator. Batched/concurrent serving is future
work.
"""

import asyncio
import codecs
import json
import queue
import threading
import time
import uuid
from typing import AsyncIterator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tiktoken import Encoding

from picochat.engine import SamplingConfig, generate
from picochat.tokenizer import render_chat_prompt


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = (
        None  # extension beyond the OpenAI spec (vLLM/Ollama also accept it)
    )
    max_tokens: int | None = None


def _resolve_sampling(
    req: ChatCompletionRequest, default: SamplingConfig
) -> SamplingConfig:
    """Per-request overrides on top of the server's --temperature/--top-k/...
    defaults; fields left unset in the request (None) fall back to them."""
    return SamplingConfig(
        temperature=req.temperature
        if req.temperature is not None
        else default.temperature,
        top_k=req.top_k if req.top_k is not None else default.top_k,
        top_p=req.top_p if req.top_p is not None else default.top_p,
        max_new_tokens=req.max_tokens
        if req.max_tokens is not None
        else default.max_new_tokens,
    )


def _completion_budget(
    prompt_len: int, sampling: SamplingConfig, max_seq_len: int
) -> int:
    """Mirrors generate()'s own budget cap (max_new_tokens vs remaining
    context) so the API can tell "hit a stop token" and "ran out of budget"
    apart for `finish_reason`, without generate() itself reporting it."""
    return min(sampling.max_new_tokens, max_seq_len - prompt_len)


def _error(message: str, code: str, status: int) -> HTTPException:
    # Matches the OpenAI error envelope so clients that parse it don't choke.
    return HTTPException(
        status_code=status,
        detail={
            "error": {"message": message, "type": "invalid_request_error", "code": code}
        },
    )


def _run_generation(
    model, tokenizer, prompt_ids, sampling, device, max_seq_len
) -> tuple[list[int], bool]:
    """Consume generate() fully. Returns (token_ids, stopped_on_stop_token) --
    the latter distinguishes finish_reason "stop" from "length"."""
    budget = _completion_budget(len(prompt_ids), sampling, max_seq_len)
    token_ids = list(
        generate(
            model, tokenizer, prompt_ids, sampling, device, max_seq_len=max_seq_len
        )
    )
    return token_ids, len(token_ids) < budget


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _stream_chat_completion(
    model,
    tokenizer,
    prompt_ids: list[int],
    sampling: SamplingConfig,
    device,
    max_seq_len: int,
    completion_id: str,
    created: int,
    model_name: str,
    lock: asyncio.Lock,
) -> AsyncIterator[bytes]:
    def chunk(delta: dict, finish_reason: str | None) -> dict:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    async with lock:
        q: queue.Queue = queue.Queue()
        budget = _completion_budget(len(prompt_ids), sampling, max_seq_len)

        def produce() -> None:
            # Tokens don't align with UTF-8 codepoints, so a multi-byte
            # character split across two tokens must be buffered until it's
            # complete rather than decoded (and emitted) one token at a time.
            # errors="replace" (not the default "strict") matters here: an
            # undertrained/base checkpoint can emit a token sequence that
            # never resolves to valid UTF-8, which would otherwise raise
            # inside this thread and leave the consumer below awaiting a
            # "done" that never comes -- the `finally` is a second backstop
            # for any other unexpected error.
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            n = 0
            try:
                for token_id in generate(
                    model,
                    tokenizer,
                    prompt_ids,
                    sampling,
                    device,
                    max_seq_len=max_seq_len,
                ):
                    n += 1
                    text = decoder.decode(tokenizer.decode_single_token_bytes(token_id))
                    if text:
                        q.put(("text", text))
            finally:
                q.put(("done", n < budget))

        threading.Thread(target=produce, daemon=True).start()

        yield _sse(chunk({"role": "assistant"}, None))
        stopped = True
        while True:
            kind, payload = await asyncio.to_thread(q.get)
            if kind == "done":
                stopped = payload
                break
            yield _sse(chunk({"content": payload}, None))
        yield _sse(chunk({}, "stop" if stopped else "length"))
        yield b"data: [DONE]\n\n"


def create_app(
    model: torch.nn.Module,
    tokenizer: Encoding,
    device: torch.device | str,
    max_seq_len: int,
    model_id: str,
    default_sampling: SamplingConfig | None = None,
) -> FastAPI:
    default_sampling = default_sampling or SamplingConfig()
    generation_lock = asyncio.Lock()
    app = FastAPI(title="picochat")

    @app.get("/v1/models")
    def list_models() -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "picochat",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if not req.messages:
            raise _error("messages must not be empty", "empty_messages", 400)
        messages = [m.model_dump() for m in req.messages]
        prompt_ids = render_chat_prompt(messages, tokenizer)
        sampling = _resolve_sampling(req, default_sampling)
        if _completion_budget(len(prompt_ids), sampling, max_seq_len) <= 0:
            raise _error(
                f"prompt ({len(prompt_ids)} tokens) leaves no room to generate "
                f"within this model's max_seq_len={max_seq_len}",
                "context_length_exceeded",
                400,
            )

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if req.stream:
            return StreamingResponse(
                _stream_chat_completion(
                    model,
                    tokenizer,
                    prompt_ids,
                    sampling,
                    device,
                    max_seq_len,
                    completion_id,
                    created,
                    req.model,
                    generation_lock,
                ),
                media_type="text/event-stream",
            )

        async with generation_lock:
            token_ids, stopped = await asyncio.to_thread(
                _run_generation,
                model,
                tokenizer,
                prompt_ids,
                sampling,
                device,
                max_seq_len,
            )
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": tokenizer.decode(token_ids),
                    },
                    "finish_reason": "stop" if stopped else "length",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(token_ids),
                "total_tokens": len(prompt_ids) + len(token_ids),
            },
        }

    return app
