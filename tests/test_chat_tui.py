"""Drive the chat TUI headlessly via textual's run_test pilot: a scripted
model stands in for the LM so replies are deterministic."""

import asyncio

from textual.widgets import Input, Static

from picochat.generate import SamplingConfig
from picochat.tokenizer import IM_END
from scripts.base_chat import ChatApp
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


def test_unknown_command_notice():
    async def go():
        app = make_app()
        async with app.run_test() as pilot:
            await submit(app, pilot, "/frobnicate")
            assert app.messages == []
            notices = [str(w.content) for w in app.query(".notice").results(Static)]
            assert any("unknown command" in n for n in notices)

    asyncio.run(go())
