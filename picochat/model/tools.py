"""Tool (function) calling: the Hermes / Qwen2.5 convention, rendered on top
of picochat's ChatML tokenizer.

Wire format:
  - Tools are declared to the model in the system prompt as JSON schema
    (render_tool_system): a `# Tools` section listing each function's
    signature, plus the instruction to emit calls between the
    TOOL_CALL_START/END special tokens.
  - The assistant calls a tool by emitting
    `<|tool_call|>{"name": ..., "arguments": {...}}<|/tool_call|>`. Because
    the delimiters are single special tokens, TOOL_CALL_END is a natural stop
    signal and parse_tool_calls can recover the calls unambiguously.
  - The caller runs the tool and appends the result as a `tool` ChatML turn
    ({"role": "tool", "content": ...}); render_turn already handles it, and
    encode_conversation loss-masks it (a tool result is environment output,
    not the model's to produce -- only the assistant's own call tokens train).

This module is the pure format layer (render/parse/execute-a-registry). The
API server maps it to OpenAI `tools`/`tool_calls` (see
picochat.inference.api), and SFT data can teach it by including tool turns in
the conversation JSONL.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from picochat.tokenizer import TOOL_CALL_END, TOOL_CALL_START

# Assistant content between the call delimiters. DOTALL so multi-line JSON
# arguments are captured; non-greedy so adjacent calls stay separate.
_TOOL_CALL_RE = re.compile(
    re.escape(TOOL_CALL_START) + r"\s*(.*?)\s*" + re.escape(TOOL_CALL_END),
    re.DOTALL,
)


def render_tool_system(tools: list[dict], base_system: str | None = None) -> str:
    """Build the system-prompt text that declares `tools` to the model.

    `tools` are JSON-schema function specs in the OpenAI shape -- either the
    bare `{"name", "description", "parameters"}` or the wrapped
    `{"type": "function", "function": {...}}` (both accepted). The returned
    string is `base_system` (if any) followed by a `# Tools` section listing
    each spec as JSON and the call-format instruction; pass it as the
    conversation's system turn."""
    specs = [t.get("function", t) for t in tools]
    lines = [base_system.strip()] if base_system and base_system.strip() else []
    lines.append(
        "# Tools\n\n"
        "You may call one or more of the following functions to help answer "
        "the user. Each is described by its JSON schema:"
    )
    for spec in specs:
        lines.append(json.dumps(spec, ensure_ascii=False))
    lines.append(
        "To call a function, emit a call wrapped exactly like\n"
        f"{TOOL_CALL_START}"
        '{"name": <function-name>, "arguments": <arguments-json-object>}'
        f"{TOOL_CALL_END}\n"
        "You may emit several calls. After the results are returned to you as "
        "`tool` messages, use them to write your final answer."
    )
    return "\n\n".join(lines)


def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from an assistant response. Returns a list of
    `{"name": str, "arguments": dict}` for every well-formed
    TOOL_CALL_START..END block; blocks whose payload isn't a JSON object with
    a string `name` are skipped (a small model will sometimes emit malformed
    calls -- dropping them is safer than raising into the serving loop).
    `arguments` defaults to an empty dict when absent."""
    calls = []
    for match in _TOOL_CALL_RE.findall(text):
        try:
            obj = json.loads(match)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            continue
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            continue
        calls.append({"name": obj["name"], "arguments": args})
    return calls


def strip_tool_calls(text: str) -> str:
    """The assistant text with every tool-call block removed -- the
    natural-language part of a response that also called tools."""
    return _TOOL_CALL_RE.sub("", text).strip()


def render_tool_call(name: str, arguments: dict) -> str:
    """The exact assistant-side call string for a (name, arguments) pair --
    used to build SFT targets and to echo a parsed call back."""
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"{TOOL_CALL_START}{payload}{TOOL_CALL_END}"


class ToolRegistry:
    """A name -> callable map with JSON-schema specs, for executing the calls
    the model emits. Purely local: the caller decides which Python functions
    to expose. `execute` never raises into the serving loop -- a bad call
    name or a tool exception becomes an error string the model sees as the
    tool result and can recover from."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[Callable[..., object], dict]] = {}

    def register(self, spec: dict, fn: Callable[..., object]) -> None:
        """Register `fn` under the name in its OpenAI-shaped `spec`
        ({"name", ...} or {"function": {"name", ...}})."""
        function = spec.get("function", spec)
        self._tools[function["name"]] = (fn, spec)

    @property
    def specs(self) -> list[dict]:
        return [spec for _, spec in self._tools.values()]

    def execute(self, call: dict) -> str:
        """Run one parsed call ({"name", "arguments"}) and return its result
        as a string (JSON for non-string returns), or an error message the
        model can react to."""
        name = call["name"]
        entry = self._tools.get(name)
        if entry is None:
            return f"error: unknown tool {name!r} (available: {list(self._tools)})"
        fn, _ = entry
        try:
            result = fn(**call["arguments"])
        except Exception as e:  # tool bugs must not kill the serving loop
            return f"error: tool {name!r} raised {type(e).__name__}: {e}"
        return (
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False)
        )
