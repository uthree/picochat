"""Interactive chat TUI (textual) for a trained picochat checkpoint.

Loads a model + tokenizer from a checkpoint and opens a ChatML conversation:
each submitted line becomes a user turn, the full history is rendered via
picochat.data.sft.render_chat_prompt (ending in the `<|im_start|>assistant\\n`
cue) and the reply streams into the log until the model emits `<|im_end|>`
(or `<|end_of_text|>`/the token budget). Slash commands control the session:

    /reset               clear the conversation (keeps the system prompt)
    /system <text>       set the system prompt and reset the conversation
    /set <key> <value>   temperature, top_k, top_p, max_new_tokens
    /help                list commands
    /quit                exit (also Ctrl+Q); Esc stops a running generation

    python scripts/base_chat.py --checkpoint weights/sft-stage1/last.ckpt \\
        --system "You are a helpful assistant."

Like base_eval.py, the architecture is rebuilt from the checkpoint's embedded
model_config. A base (pretrain-only) checkpoint has never seen ChatML turns,
so it will ramble; this chat is primarily for SFT checkpoints.
"""

import argparse
from pathlib import Path

import torch
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Input, Static
from textual.worker import get_current_worker

from picochat.data.sft import render_chat_prompt
from picochat.generate import SamplingConfig, generate
from picochat.model.gpt import load_gpt_checkpoint

HELP = """\
/reset               clear the conversation (keeps the system prompt)
/system <text>       set the system prompt and reset the conversation
/set <key> <value>   temperature, top_k, top_p, max_new_tokens
/help                list commands
/quit                exit (also Ctrl+Q); Esc stops a running generation"""


class ChatApp(App):
    """Streaming ChatML chat over a TransformerLM, with slash commands."""

    CSS = """
    #log { height: 1fr; padding: 0 1; }
    #status { height: 1; padding: 0 2; color: $text-muted; }
    .msg { padding: 0 1; margin: 1 1 0 1; border: round $surface-lighten-2; }
    .user { border: round $accent; }
    .assistant { border: round $success; }
    .notice { color: $text-muted; margin: 1 2 0 2; }
    """
    BINDINGS = [
        ("ctrl+q", "quit", "quit"),
        ("escape", "stop_generation", "stop generation"),
    ]

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: torch.device | str = "cpu",
        sampling: SamplingConfig | None = None,
        system: str | None = None,
        banner: str | None = None,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sampling = sampling or SamplingConfig()
        self.system = system
        self.messages: list[dict] = []  # user/assistant turns only
        self.banner = banner
        self.max_seq_len = max_seq_len
        self._generating = False

    # --- UI scaffolding ----------------------------------------------------
    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="log")
        yield Static(id="status")
        yield Input(placeholder="message picochat (/help for commands)")

    def on_mount(self) -> None:
        if self.banner:
            self._add_widget(Static(Text(self.banner), classes="notice"))
        if self.system:
            self._notice(f"system prompt: {self.system}")
        self._refresh_status()
        self.query_one(Input).focus()

    def _refresh_status(self) -> None:
        busy = "  •  generating... (Esc stops)" if self._generating else ""
        self.query_one("#status", Static).update(self.sampling.describe() + busy)

    def _add_widget(self, widget: Static) -> Static:
        log = self.query_one("#log", VerticalScroll)
        log.mount(widget)
        log.scroll_end(animate=False)
        return widget

    def _add_message(self, role: str, text: str) -> Static:
        widget = Static(Text(text), classes=f"msg {role}")
        widget.border_title = "you" if role == "user" else "picochat"
        return self._add_widget(widget)

    def _notice(self, text: str) -> None:
        self._add_widget(Static(Text(text), classes="notice"))

    # --- input handling ----------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if not text:
            return
        if text.startswith("/"):
            self._handle_command(text)
            return
        if self._generating:
            self._notice("still generating -- Esc to stop it first")
            return
        self.messages.append({"role": "user", "content": text})
        self._add_message("user", text)
        self._generating = True
        self._refresh_status()
        self._generate_reply()

    def _handle_command(self, text: str) -> None:
        cmd, _, arg = text.partition(" ")
        arg = arg.strip()
        if cmd == "/reset":
            self.messages.clear()
            self._notice("conversation cleared")
        elif cmd == "/system":
            self.system = arg or None
            self.messages.clear()
            self._notice(
                f"system prompt set: {arg}" if arg else "system prompt removed"
            )
            self._notice("conversation cleared")
        elif cmd == "/set":
            key, _, raw = arg.partition(" ")
            try:
                if not key or not raw.strip():
                    raise ValueError("usage: /set <key> <value>")
                self.sampling.update(key, raw.strip())
                self._refresh_status()
                self._notice(self.sampling.describe())
            except ValueError as e:
                self._notice(f"error: {e}")
        elif cmd == "/help":
            self._notice(HELP)
        elif cmd in ("/quit", "/exit"):
            self.exit()
        else:
            self._notice(f"unknown command {cmd} (/help for commands)")

    def action_stop_generation(self) -> None:
        self.workers.cancel_group(self, "generate")

    # --- generation --------------------------------------------------------
    def _build_prompt(self) -> tuple[list[int], bool]:
        """Render the ChatML prompt, dropping the oldest turns while it
        overflows the model's context window -- the prompt must fit
        max_seq_len (the RoPE tables) with some room left to reply. Returns
        (ids, trimmed). self.messages itself is never modified."""
        system = [{"role": "system", "content": self.system}] if self.system else []
        # Leave room for the reply: the full budget when it is modest, but a
        # long history shouldn't be dropped just because max_new_tokens is
        # large -- generate() caps the reply to whatever room remains.
        reserve = min(self.sampling.max_new_tokens, max(self.max_seq_len // 4, 1))
        keep = list(self.messages)
        trimmed = False
        while True:
            ids = render_chat_prompt(system + keep, self.tokenizer)
            if len(ids) + reserve <= self.max_seq_len or len(keep) <= 1:
                break
            keep = keep[1:]
            trimmed = True
        if len(ids) >= self.max_seq_len:
            # A single oversized turn: hard-truncate, keeping the leading BOS
            # and the tail (which ends in the assistant cue).
            ids = [ids[0], *ids[-(self.max_seq_len - reserve - 1) :]]
            trimmed = True
        return ids, trimmed

    @work(thread=True, exclusive=True, group="generate")
    def _generate_reply(self) -> None:
        """Streams one assistant reply in a worker thread; every UI touch
        goes through call_from_thread."""
        worker = get_current_worker()
        prompt_ids, trimmed = self._build_prompt()
        if trimmed:
            self.call_from_thread(
                self._notice, "context window full: oldest turns were dropped"
            )

        widget = self.call_from_thread(self._add_message, "assistant", "...")
        reply_ids: list[int] = []
        # Streamed display decodes per token (a multi-byte character split
        # across tokens shows replacement chars until complete)...
        shown = ""
        for token_id in generate(
            self.model,
            self.tokenizer,
            prompt_ids,
            self.sampling,
            self.device,
            max_seq_len=self.max_seq_len,
        ):
            if worker.is_cancelled:
                break
            reply_ids.append(token_id)
            shown += self.tokenizer.decode_single_token_bytes(token_id).decode(
                "utf-8", errors="replace"
            )
            self.call_from_thread(self._stream_update, widget, shown)
        # ...but the history entry is decoded from the full id sequence, so
        # the next turn's prompt re-encodes clean text. A stopped (partial)
        # reply is kept: it is what the user saw.
        reply = self.tokenizer.decode(reply_ids)
        self.messages.append({"role": "assistant", "content": reply})
        self.call_from_thread(self._finish_reply, widget, reply, worker.is_cancelled)

    def _stream_update(self, widget: Static, text: str) -> None:
        widget.update(Text(text))
        self.query_one("#log", VerticalScroll).scroll_end(animate=False)

    def _finish_reply(self, widget: Static, reply: str, stopped: bool) -> None:
        widget.update(Text(reply + (" [stopped]" if stopped else "")))
        self._generating = False
        self._refresh_status()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="path to a .ckpt file")
    p.add_argument("--tokenizer", type=str, default="weights/tokenizer.json")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--temperature", type=float, default=0.8, help="0 -> greedy decoding"
    )
    p.add_argument("--top-k", type=int, default=50, help="0 -> disabled")
    p.add_argument("--top-p", type=float, default=1.0, help="1.0 -> disabled")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--system",
        type=str,
        default=None,
        help="optional system prompt, prepended as a ChatML system turn",
    )
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"loading model from {args.checkpoint} (device={device}) ...", flush=True)
    gpt, tokenizer = load_gpt_checkpoint(args.checkpoint, args.tokenizer, device)

    sampling = SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if 0 < args.top_p < 1 else None,
        max_new_tokens=args.max_new_tokens,
    )
    logo = Path("assets/logo_ascii.txt")
    banner = logo.read_text() if logo.exists() else None

    app = ChatApp(
        gpt.model,
        tokenizer,
        device=device,
        sampling=sampling,
        system=args.system,
        banner=banner,
        max_seq_len=(gpt.hparams["model_config"] or {}).get("max_seq_len", 4096),
    )
    app.run()


if __name__ == "__main__":
    main()
