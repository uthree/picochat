"""Drive the chat TUI headlessly via textual's run_test pilot: a scripted
model stands in for the LM so replies are deterministic."""

import asyncio

import torch
from textual.widgets import Input, Static

from picochat.inference.engine import SamplingConfig
from picochat.tokenizer import IM_END
from scripts.chat import ChatApp
from tests.test_generate import ByteTokenizer, ScriptedModel


def make_app(script=None, **kwargs) -> ChatApp:
    tok = ByteTokenizer()
    script = script or [65, 66, tok.encode_single_token(IM_END)]  # "AB"
    return ChatApp(
        ScriptedModel(script),
        tok,
        sampling=SamplingConfig(temperature=0.0),
        **kwargs,
    )


async def submit(app: ChatApp, pilot, text: str) -> None:
    app.query_one(Input).value = text
    await pilot.press("enter")
    await pilot.pause()


def test_chat_roundtrip_streams_reply_into_history():
    async def go():
        app = make_app()
        async with app.run_test() as pilot:
            await submit(app, pilot, "hi")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.messages == [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "AB"},
            ]
            rendered = [str(w.content) for w in app.query(".msg").results(Static)]
            assert rendered == ["hi", "AB"]

    asyncio.run(go())


def test_set_command_updates_sampling():
    async def go():
        app = make_app()
        async with app.run_test() as pilot:
            await submit(app, pilot, "/set temperature 0.25")
            assert app.sampling.temperature == 0.25
            await submit(app, pilot, "/set top_k off")
            assert app.sampling.top_k is None
            # bad input: config untouched, an error notice appears
            await submit(app, pilot, "/set top_p 7")
            assert app.sampling.top_p is None
            notices = [str(w.content) for w in app.query(".notice").results(Static)]
            assert any("error" in n for n in notices)

    asyncio.run(go())


def test_reset_and_system_commands_clear_context():
    async def go():
        app = make_app(system="be brief")
        async with app.run_test() as pilot:
            await submit(app, pilot, "hi")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert len(app.messages) == 2
            await submit(app, pilot, "/reset")
            assert app.messages == []
            assert app.system == "be brief"  # /reset keeps the system prompt
            await submit(app, pilot, "/system be verbose")
            assert app.system == "be verbose"

    asyncio.run(go())


def test_multi_turn_survives_small_context_window():
    # Regression: turn 2+ used to decode past max_seq_len and crash the
    # worker (position N exceeds max_seq_len assertion). With a real model,
    # a tiny window and the default 256-token budget, several turns must
    # stream replies without overrunning the RoPE tables.
    from picochat.model import TransformerLM

    async def go():
        torch.manual_seed(0)
        lm = TransformerLM(
            vocab_size=512,
            d_model=32,
            n_heads=4,
            n_layers=2,
            max_seq_len=64,
            window_size=8,
            grad_checkpoint=False,
        ).eval()
        app = ChatApp(lm, ByteTokenizer(), max_seq_len=64)
        async with app.run_test() as pilot:
            for i, msg in enumerate(["hello", "second message", "third"]):
                await submit(app, pilot, msg)
                await app.workers.wait_for_complete()
                await pilot.pause()
                assert app.messages[-1]["role"] == "assistant"
                assert len(app.messages) == 2 * (i + 1)

    asyncio.run(go())


def test_prompt_trimming_drops_oldest_turns():
    async def go():
        app = make_app(max_seq_len=48)
        async with app.run_test() as pilot:
            await submit(app, pilot, "first message padded out")
            await app.workers.wait_for_complete()
            await pilot.pause()
            await submit(app, pilot, "second message padded out")
            await app.workers.wait_for_complete()
            await pilot.pause()
            # the full history no longer fits 48 positions: the prompt must
            # shrink (oldest turns dropped) while the history stays intact
            ids, trimmed = app._build_prompt()
            assert trimmed
            assert len(ids) < 48
            assert len(app.messages) == 4
            notices = [str(w.content) for w in app.query(".notice").results(Static)]
            assert any("dropped" in n for n in notices)

    asyncio.run(go())


def test_theme_defaults_to_ansi_dark():
    async def go():
        app = make_app()
        async with app.run_test():
            assert app.theme == "ansi-dark"

    asyncio.run(go())


def test_theme_flag_and_command():
    async def go():
        # --theme equivalent: applied on mount
        app = make_app(theme="nord")
        async with app.run_test() as pilot:
            assert app.theme == "nord"
            # /theme switches at runtime
            await submit(app, pilot, "/theme ansi-light")
            assert app.theme == "ansi-light"
            # unknown theme: error notice, theme unchanged
            await submit(app, pilot, "/theme not-a-theme")
            assert app.theme == "ansi-light"
            notices = [str(w.content) for w in app.query(".notice").results(Static)]
            assert any("unknown theme" in n for n in notices)

    asyncio.run(go())


def test_invalid_theme_flag_falls_back_with_notice():
    async def go():
        # --theme with a typo must not abort: fall back to the default and
        # list what is available
        app = make_app(theme="not-a-theme")
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.theme == "ansi-dark"
            notices = [str(w.content) for w in app.query(".notice").results(Static)]
            assert any("unknown theme" in n and "ansi-light" in n for n in notices)

    asyncio.run(go())


def test_status_shows_context_usage():
    async def go():
        app = make_app(max_seq_len=100)
        async with app.run_test() as pilot:
            status = str(app.query_one("#status", Static).content)
            assert "context=" in status and "/100" in status
            before = int(status.split("context=")[1].split("/")[0])
            await submit(app, pilot, "hello")
            await app.workers.wait_for_complete()
            await pilot.pause()
            status = str(app.query_one("#status", Static).content)
            after = int(status.split("context=")[1].split("/")[0])
            assert after > before  # the exchange consumed context window

    asyncio.run(go())


def test_tab_accepts_command_completion():
    async def go():
        app = make_app()
        async with app.run_test() as pilot:
            await pilot.press("/", "r", "e", "s")
            await pilot.pause(0.1)  # let the suggester deliver its suggestion
            await pilot.press("tab")
            assert app.query_one(Input).value == "/reset"

    asyncio.run(go())


def test_unknown_command_notice():
    async def go():
        app = make_app()
        async with app.run_test() as pilot:
            await submit(app, pilot, "/frobnicate")
            assert app.messages == []
            notices = [str(w.content) for w in app.query(".notice").results(Static)]
            assert any("unknown command" in n for n in notices)

    asyncio.run(go())
