import pytest
import torch

from picochat.inference.engine import SamplingConfig, generate, sample
from picochat.tokenizer import IM_END, SPECIAL_TOKENS


class ByteTokenizer:
    """1 byte = 1 token; special tokens get ids >= 256."""

    def __init__(self):
        self._special = {tok: 256 + i for i, tok in enumerate(SPECIAL_TOKENS)}

    def encode_ordinary(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def encode_single_token(self, token: str) -> int:
        return self._special[token]

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        # ids past the byte range (special/out-of-script tokens a random
        # model may sample) render as "?"
        return bytes([token_id]) if token_id < 256 else b"?"

    def decode(self, ids: list[int]) -> str:
        joined = b"".join(self.decode_single_token_bytes(i) for i in ids)
        return joined.decode("utf-8", errors="replace")


class ScriptedModel(torch.nn.Module):
    """decode() returns one-hot-ish logits forcing a scripted token sequence
    (with greedy sampling), regardless of the input ids."""

    def __init__(self, script: list[int], vocab_size: int = 300):
        super().__init__()
        self.script = list(script)
        self.vocab_size = vocab_size
        self.calls = 0

    def decode(self, x, cache=None, pos=0):
        logits = torch.full((x.shape[0], x.shape[1], self.vocab_size), -100.0)
        # wrap around so multi-turn tests replay the same scripted reply
        logits[:, -1, self.script[self.calls % len(self.script)]] = 100.0
        self.calls += 1
        return logits, None, pos + x.shape[1]


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------
def test_sample_greedy_at_zero_temperature():
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    cfg = SamplingConfig(temperature=0.0)
    assert sample(logits, cfg).item() == 1


def test_sample_top_k_restricts_support():
    torch.manual_seed(0)
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0]])
    cfg = SamplingConfig(temperature=1.0, top_k=2)
    drawn = {sample(logits, cfg).item() for _ in range(200)}
    assert drawn <= {0, 1}
    assert len(drawn) == 2  # both survivors actually reachable


def test_sample_top_p_restricts_support():
    torch.manual_seed(0)
    # probs (0.5, 0.3, 0.2): mass before token 2 is 0.8 >= 0.6, so top_p=0.6
    # keeps exactly tokens 0 and 1
    logits = torch.tensor([[0.5, 0.3, 0.2]]).log()
    cfg = SamplingConfig(temperature=1.0, top_k=None, top_p=0.6)
    drawn = {sample(logits, cfg).item() for _ in range(200)}
    assert drawn <= {0, 1}
    assert len(drawn) == 2


def test_sample_top_p_always_keeps_most_probable():
    # top token's own mass (0.9) exceeds top_p: it must survive anyway
    logits = torch.tensor([[0.9, 0.1]]).log()
    cfg = SamplingConfig(temperature=1.0, top_k=None, top_p=0.5)
    drawn = {sample(logits, cfg).item() for _ in range(50)}
    assert drawn == {0}


# ---------------------------------------------------------------------------
# SamplingConfig.update
# ---------------------------------------------------------------------------
def test_update_parses_and_validates():
    cfg = SamplingConfig()
    cfg.update("temperature", "0.5")
    assert cfg.temperature == 0.5
    cfg.update("top_k", "10")
    assert cfg.top_k == 10
    cfg.update("top_k", "off")
    assert cfg.top_k is None
    cfg.update("top_p", "0.9")
    assert cfg.top_p == 0.9
    cfg.update("top_p", "none")
    assert cfg.top_p is None
    cfg.update("max_new_tokens", "32")
    assert cfg.max_new_tokens == 32


@pytest.mark.parametrize(
    "key,raw",
    [
        ("temperature", "-1"),
        ("temperature", "abc"),
        ("top_k", "0"),
        ("top_p", "1.5"),
        ("top_p", "0"),
        ("max_new_tokens", "0"),
        ("bogus", "1"),
    ],
)
def test_update_rejects_bad_input(key, raw):
    cfg = SamplingConfig()
    before = SamplingConfig(**vars(cfg))
    with pytest.raises(ValueError):
        cfg.update(key, raw)
    assert vars(cfg) == vars(before)  # failed updates leave the config intact


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
def test_generate_stops_at_im_end():
    tok = ByteTokenizer()
    model = ScriptedModel([65, 66, tok.encode_single_token(IM_END), 67])
    cfg = SamplingConfig(temperature=0.0, max_new_tokens=10)
    out = list(generate(model, tok, [1, 2, 3], cfg))
    assert out == [65, 66]


def test_generate_respects_token_budget():
    tok = ByteTokenizer()
    model = ScriptedModel([65] * 20)
    cfg = SamplingConfig(temperature=0.0, max_new_tokens=5)
    assert list(generate(model, tok, [1], cfg)) == [65] * 5


def test_generate_caps_budget_at_max_seq_len():
    # A real TransformerLM asserts if decode runs past its RoPE tables; the
    # cap must stop generation instead. vocab < 256 keeps the byte
    # tokenizer's stop-token ids unreachable, so only the cap can end it.
    from picochat.model import TransformerLM

    torch.manual_seed(0)
    lm = TransformerLM(
        vocab_size=200,
        d_model=32,
        n_heads=4,
        n_layers=2,
        max_seq_len=32,
        window_size=8,
        grad_checkpoint=False,
    ).eval()
    tok = ByteTokenizer()
    cfg = SamplingConfig(temperature=1.0, top_k=None, max_new_tokens=100)
    prompt = list(range(1, 11))
    out = list(generate(lm, tok, prompt, cfg, max_seq_len=32))
    assert len(out) == 32 - len(prompt)


def test_generate_yields_nothing_when_prompt_fills_window():
    tok = ByteTokenizer()
    model = ScriptedModel([65] * 5)
    cfg = SamplingConfig(temperature=0.0)
    assert list(generate(model, tok, [1] * 8, cfg, max_seq_len=8)) == []
    assert model.calls == 0  # not even the prefill runs


def test_generate_is_lazy():
    # Breaking out of the loop aborts decoding: nothing runs past the break.
    tok = ByteTokenizer()
    model = ScriptedModel([65] * 20)
    cfg = SamplingConfig(temperature=0.0, max_new_tokens=10)
    for i, _ in enumerate(generate(model, tok, [1], cfg)):
        if i == 1:
            break
    # prompt prefill + 1st loop iteration + the decode feeding iteration 2
    assert model.calls <= 4


# ---------------------------------------------------------------------------
# anti-repetition penalties
# ---------------------------------------------------------------------------
def test_sample_repetition_penalty_shifts_greedy_choice():
    # token 0 barely leads; once it is in the history the multiplicative
    # penalty drops it below token 1 -- greedy decoding escapes the loop.
    logits = torch.tensor([[1.0, 0.95, -2.0]])
    cfg = SamplingConfig(temperature=0.0, repetition_penalty=1.2)
    history = torch.tensor([[0]])
    assert int(sample(logits, cfg, history)) == 1
    # without history (or with the penalty off) the raw argmax wins
    assert int(sample(logits, cfg)) == 0
    assert int(sample(logits, SamplingConfig(temperature=0.0), history)) == 0


def test_sample_repetition_penalty_scales_negative_logits_down():
    # a seen token with a NEGATIVE logit must get less likely, not more
    logits = torch.tensor([[-0.5, -1.0]])
    cfg = SamplingConfig(temperature=0.0, repetition_penalty=2.0)
    history = torch.tensor([[0]])
    assert (
        int(sample(logits, cfg, history)) == 0
    )  # -0.5*2 = -1.0, ties broken by argmax
    strong = SamplingConfig(temperature=0.0, repetition_penalty=3.0)
    assert int(sample(logits, strong, history)) == 1  # -1.5 < -1.0


def test_sample_frequency_penalty_counts_occurrences():
    logits = torch.tensor([[1.0, 0.8]])
    history = torch.tensor([[0, 0, 0]])  # token 0 emitted three times
    cfg = SamplingConfig(temperature=0.0, frequency_penalty=0.1)
    # 1.0 - 3*0.1 = 0.7 < 0.8
    assert int(sample(logits, cfg, history)) == 1
    once = torch.tensor([[0]])  # 1.0 - 0.1 = 0.9 > 0.8
    assert int(sample(logits, cfg, once)) == 0


def test_sample_presence_penalty_is_flat():
    logits = torch.tensor([[1.0, 0.8]])
    cfg = SamplingConfig(temperature=0.0, presence_penalty=0.3)
    # present once or thrice, the penalty is the same 0.3
    for history in (torch.tensor([[0]]), torch.tensor([[0, 0, 0]])):
        assert int(sample(logits, cfg, history)) == 1


def test_generate_repetition_penalty_breaks_loops():
    # A constant-logits model loops on one token forever under plain greedy;
    # with the penalty the stream must visit other tokens.
    class LoopyModel(torch.nn.Module):
        n_mtp = 0

        def decode(self, x=None, cache=None, pos=0, inputs_embeds=None):
            logits = torch.zeros(1, 1, 8)
            logits[..., 3] = 5.0  # always prefers token 3
            logits[..., 4] = 4.9
            logits[..., 5] = 4.8
            return logits, cache, pos + 1

    tok = ByteTokenizer()
    plain = list(
        generate(
            LoopyModel(),
            tok,
            [1, 2],
            SamplingConfig(temperature=0.0, max_new_tokens=6),
        )
    )
    assert set(plain) == {3}
    shaped = list(
        generate(
            LoopyModel(),
            tok,
            [1, 2],
            SamplingConfig(temperature=0.0, max_new_tokens=6, repetition_penalty=2.0),
        )
    )
    assert len(set(shaped)) > 1  # escaped the loop


def test_update_and_describe_cover_penalties():
    cfg = SamplingConfig()
    assert not cfg.penalized()
    cfg.update("repetition_penalty", "1.3")
    cfg.update("frequency_penalty", "0.5")
    cfg.update("presence_penalty", "off")
    assert cfg.repetition_penalty == 1.3 and cfg.frequency_penalty == 0.5
    assert cfg.penalized()
    with pytest.raises(ValueError):
        cfg.update("repetition_penalty", "-1")
    assert "repetition_penalty=1.3" in cfg.describe()
