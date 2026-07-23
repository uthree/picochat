"""Multimodal integration: part rendering / loss masks (model.multimodal),
the JSONL dataset + collate (data.multimodal), SFT training through
inputs_embeds, the engine's embeds prefill, and the OpenAI content-part API.
Runs on tiny random-weight encoders -- no network, no pretrained downloads.
"""

import base64
import io
import json
import math
import struct
import wave

import pytest
import torch
from torch import nn

from picochat.data.multimodal import MultimodalSFTDataset, mm_collate
from picochat.model import TransformerLM
from picochat.model.audio import AudioConfig, AudioProcessor
from picochat.model.multimodal import (
    MediaAdapters,
    encode_mm_conversation,
    render_mm_prompt,
    scatter_embeds,
    splice_media_embeds,
)
from picochat.model.vision import VisionEncoder, VisionEncoderConfig
from picochat.tokenizer import (
    AUDIO_TOKEN,
    IMAGE_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    load_tokenizer,
    render_chat_prompt,
    train_tokenizer,
)

D_MODEL = 32


class TinyAudioEncoder(nn.Module):
    """Contract-shaped stand-in for an audio encoder: `processor`,
    `num_tokens`, `projector`, `tower_parameters`, and forward
    (B, n_mels, T) -> (B, num_tokens(T), d_model)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.processor = AudioProcessor(AudioConfig(n_mels=16))
        self.downsample = self.processor.cfg.downsample
        self.tower = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.projector = nn.Linear(16 * self.downsample, d_model)

    def num_tokens(self, n_frames: int) -> int:
        return n_frames // self.downsample

    def tower_parameters(self):
        return self.tower.parameters()

    def forward(self, mel):
        x = self.tower(mel)
        b, c, t = x.shape
        n = self.num_tokens(t)
        x = x[:, :, : n * self.downsample].reshape(b, c, n, self.downsample)
        x = x.permute(0, 2, 1, 3).reshape(b, n, c * self.downsample)
        return self.projector(x)


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    corpus = ["hello world what is said here", "a spoken description of an image"] * 30
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    train_tokenizer(
        iter(corpus), vocab_size=400, save_as=path, special_tokens=SPECIAL_TOKENS
    )
    return load_tokenizer(path)


@pytest.fixture()
def encoders():
    torch.manual_seed(0)
    audio = TinyAudioEncoder(D_MODEL)
    vision = VisionEncoder(
        VisionEncoderConfig(
            image_size=32, patch_size=8, d_encoder=16, n_layers=1, n_heads=2, d_ffn=32
        ),
        D_MODEL,
    )
    return audio, vision


def _media(audio, vision):
    return MediaAdapters.from_encoders(audio, vision)


def _wav_seconds(sec=0.5):
    return torch.sin(torch.arange(int(16000 * sec)) / 30.0)


def _pil_image():
    from PIL import Image

    return Image.new("RGB", (40, 24), (200, 30, 30))


MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": _wav_seconds()},
            {"type": "text", "text": "what is said here"},
            {"type": "image", "image": None},  # filled per test
        ],
    },
    {"role": "assistant", "content": "hello world"},
]


def _messages():
    msgs = json.loads(
        json.dumps(
            [{**m, "content": m["content"]} for m in MESSAGES], default=lambda o: None
        )
    )
    msgs[0]["content"][0]["audio"] = _wav_seconds()
    msgs[0]["content"][2]["image"] = _pil_image()
    return msgs


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def test_render_mm_prompt_text_only_matches_text_renderer(tok, encoders):
    messages = [{"role": "user", "content": "hello world"}]
    ids, mels, images = render_mm_prompt(messages, tok, _media(*encoders))
    assert ids == render_chat_prompt(messages, tok)
    assert mels == [] and images == []


def test_render_mm_prompt_placeholders_match_encoders(tok, encoders):
    audio, vision = encoders
    ids, mels, images = render_mm_prompt(_messages(), tok, _media(audio, vision))
    n_audio = sum(1 for i in ids if i == tok.encode_single_token(AUDIO_TOKEN))
    n_image = sum(1 for i in ids if i == tok.encode_single_token(IMAGE_TOKEN))
    assert len(mels) == 1 and len(images) == 1
    assert n_audio == audio(mels[0][None]).shape[1]
    assert n_image == vision.tokens_per_image
    assert images[0].shape == (3, 32, 32)  # preprocessed pixels, not the PIL image


def test_render_rejects_media_without_encoder(tok):
    with pytest.raises(ValueError):
        render_mm_prompt(_messages(), tok, MediaAdapters())


# ---------------------------------------------------------------------------
# SFT encoding: loss mask + truncation
# ---------------------------------------------------------------------------
def test_encode_mm_conversation_masks_media_and_user(tok, encoders):
    pad = tok.encode_single_token(PAD_TOKEN)
    ids, labels, mels, images = encode_mm_conversation(
        _messages(), tok, max_length=512, pad_id=pad, media=_media(*encoders)
    )
    assert len(ids) == len(labels)
    # every media placeholder position is loss-masked
    for i, t in enumerate(ids):
        if t in (
            tok.encode_single_token(AUDIO_TOKEN),
            tok.encode_single_token(IMAGE_TOKEN),
        ):
            assert labels[i] == pad
    # the assistant text is trainable (labels echo the ids there)
    trainable = [t for t, y in zip(ids, labels) if y != pad]
    decoded = tok.decode([t for t in trainable if not tok.is_special_token(t)])
    assert "hello world" in decoded
    assert len(mels) == 1 and len(images) == 1


def test_encode_mm_conversation_truncates_whole_turns(tok, encoders):
    pad = tok.encode_single_token(PAD_TOKEN)
    msgs = _messages() + [
        {"role": "user", "content": "again " * 50},
        {"role": "assistant", "content": "yes " * 50},
    ]
    full = encode_mm_conversation(msgs, tok, 4096, pad, _media(*encoders))
    ids_full = full[0]
    # tight budget: the trailing turns must drop *whole*, media stays aligned
    small = encode_mm_conversation(
        msgs, tok, len(ids_full) - 10, pad, _media(*encoders)
    )
    ids_small = small[0]
    n_audio = sum(1 for i in ids_small if i == tok.encode_single_token(AUDIO_TOKEN))
    assert n_audio == 0 or len(small[2]) == 1
    assert len(ids_small) < len(ids_full)


def test_encode_mm_conversation_returns_none_when_nothing_trainable(tok, encoders):
    pad = tok.encode_single_token(PAD_TOKEN)
    msgs = [{"role": "user", "content": "question with no answer"}]
    assert encode_mm_conversation(msgs, tok, 128, pad, _media(*encoders)) is None


# ---------------------------------------------------------------------------
# scatter / splice
# ---------------------------------------------------------------------------
def test_scatter_embeds_replaces_and_validates(tok):
    ids = torch.tensor([[1, 2, 7, 7, 3]])
    embeds = torch.zeros(1, 5, 4)
    feats = torch.ones(2, 4)
    out = scatter_embeds(embeds, ids, feats, token_id=7)
    assert torch.equal(out[0, 2:4], torch.ones(2, 4))
    assert torch.equal(out[0, :2], torch.zeros(2, 4))
    with pytest.raises(ValueError):
        scatter_embeds(embeds, ids, torch.ones(3, 4), token_id=7)


def test_splice_media_embeds_end_to_end(tok, encoders):
    audio, vision = encoders
    ids, mels, images = render_mm_prompt(_messages(), tok, _media(audio, vision))
    x = torch.tensor([ids])
    lm = TransformerLM(vocab_size=tok.n_vocab, d_model=D_MODEL, n_heads=4, n_layers=2)
    embeds = splice_media_embeds(
        lm.embed(x),
        x,
        tok,
        audio_encoder=audio,
        mels=mels,
        vision_encoder=vision,
        images=images,
    )
    assert embeds.shape == (1, len(ids), D_MODEL)
    # placeholder rows differ from the raw token embedding, text rows don't
    raw = lm.embed(x)
    mask = (x == tok.encode_single_token(AUDIO_TOKEN)) | (
        x == tok.encode_single_token(IMAGE_TOKEN)
    )
    assert not torch.allclose(embeds[mask], raw[mask])
    assert torch.allclose(embeds[~mask], raw[~mask])


# ---------------------------------------------------------------------------
# dataset + collate
# ---------------------------------------------------------------------------
def _write_wav(path, sec=0.3):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        n = int(16000 * sec)
        w.writeframes(
            b"".join(struct.pack("<h", int(9000 * math.sin(i / 25))) for i in range(n))
        )


def _write_dataset(tmp_path):
    _write_wav(tmp_path / "a.wav")
    _pil_image().save(tmp_path / "i.png")
    records = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": "a.wav"},
                        {"type": "text", "text": "what is said"},
                    ],
                },
                {"role": "assistant", "content": "hello world"},
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": "i.png"},
                        {"type": "text", "text": "describe"},
                    ],
                },
                {"role": "assistant", "content": "a spoken description of an image"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "text only"},
                {"role": "assistant", "content": "hello"},
            ]
        },
    ]
    path = tmp_path / "sft.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def test_dataset_and_collate(tok, encoders, tmp_path):
    pad = tok.encode_single_token(PAD_TOKEN)
    ds = MultimodalSFTDataset(
        _write_dataset(tmp_path), tok, _media(*encoders), max_length=512, pad_id=pad
    )
    assert len(ds) == 3
    batch = mm_collate([ds[0], ds[1], ds[2]], pad_id=pad)
    b, length = batch["input_ids"].shape
    assert b == 3 and batch["labels"].shape == (3, length)
    assert len(batch["mels"]) == 1 and len(batch["images"]) == 1
    # pad tail: doc_ids increment per pad so nothing attends into it
    row = batch["input_ids"][2]
    n_pad = int((row == pad).sum())
    assert n_pad > 0
    assert batch["doc_ids"][2, -1] == n_pad


# ---------------------------------------------------------------------------
# SFT training through inputs_embeds
# ---------------------------------------------------------------------------
def test_sft_module_multimodal_backward(tok, encoders, tmp_path):
    from picochat.training import SFTModule

    audio, vision = encoders
    pad = tok.encode_single_token(PAD_TOKEN)
    lm = TransformerLM(vocab_size=tok.n_vocab, d_model=D_MODEL, n_heads=4, n_layers=2)
    sft = SFTModule(
        lm,
        pad_idx=pad,
        compile=False,
        tokenizer=tok,
        audio_encoder=audio,
        vision_encoder=vision,
    )
    ds = MultimodalSFTDataset(
        _write_dataset(tmp_path), tok, _media(audio, vision), max_length=512, pad_id=pad
    )
    batch = mm_collate([ds[0], ds[1]], pad_id=pad)
    loss = sft._loss(
        batch["input_ids"],
        batch["labels"],
        batch["doc_ids"],
        batch["mels"],
        batch["images"],
    )
    assert torch.isfinite(loss)
    loss.backward()
    # stage-1 freezing: projector learns, tower stays frozen
    assert all(p.grad is not None for p in audio.projector.parameters())
    assert all(p.grad is not None for p in vision.projector.parameters())
    assert all(not p.requires_grad for p in audio.tower_parameters())
    assert all(not p.requires_grad for p in vision.tower_parameters())
    # frozen towers stay out of the optimizer, projectors are in
    muon, adam_groups = sft._muon_param_split()
    opt_ids = {id(p) for g in adam_groups for p in g["params"]} | {id(p) for p in muon}
    assert all(id(p) in opt_ids for p in audio.projector.parameters())
    assert all(id(p) not in opt_ids for p in vision.tower_parameters())


def test_sft_module_text_batch_unaffected(tok, encoders):
    from picochat.training import SFTModule

    audio, vision = encoders
    pad = tok.encode_single_token(PAD_TOKEN)
    lm = TransformerLM(vocab_size=tok.n_vocab, d_model=D_MODEL, n_heads=4, n_layers=2)
    sft = SFTModule(
        lm,
        pad_idx=pad,
        compile=False,
        tokenizer=tok,
        audio_encoder=audio,
        vision_encoder=vision,
    )
    x = torch.randint(len(SPECIAL_TOKENS), tok.n_vocab, (2, 12))
    loss = sft._loss(x, x)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# inference: embeds prefill + API content parts
# ---------------------------------------------------------------------------
def test_generate_with_prompt_embeds(tok, encoders):
    from picochat.inference.engine import SamplingConfig, generate

    audio, vision = encoders
    lm = TransformerLM(
        vocab_size=tok.n_vocab, d_model=D_MODEL, n_heads=4, n_layers=2
    ).eval()
    ids, mels, images = render_mm_prompt(_messages(), tok, _media(audio, vision))
    x = torch.tensor([ids])
    with torch.no_grad():
        embeds = splice_media_embeds(
            lm.embed(x),
            x,
            tok,
            audio_encoder=audio,
            mels=mels,
            vision_encoder=vision,
            images=images,
        )
    cfg = SamplingConfig(temperature=0, max_new_tokens=4)
    tokens = list(generate(lm, tok, ids, cfg, prompt_embeds=embeds, max_seq_len=1024))
    assert len(tokens) <= 4
    assert all(isinstance(t, int) for t in tokens)


def _b64_wav(sec=0.3):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        n = int(16000 * sec)
        w.writeframes(
            b"".join(struct.pack("<h", int(9000 * math.sin(i / 25))) for i in range(n))
        )
    return base64.b64encode(buf.getvalue()).decode()


def _data_uri_png():
    buf = io.BytesIO()
    _pil_image().save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _mm_request():
    return {
        "model": "t",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": _b64_wav(), "format": "wav"},
                    },
                    {"type": "image_url", "image_url": {"url": _data_uri_png()}},
                ],
            }
        ],
        "max_tokens": 4,
    }


def test_api_multimodal_request(tok, encoders):
    from fastapi.testclient import TestClient

    from picochat.inference.api import create_app

    audio, vision = encoders
    lm = TransformerLM(
        vocab_size=tok.n_vocab, d_model=D_MODEL, n_heads=4, n_layers=2
    ).eval()
    app = create_app(
        lm,
        tok,
        device="cpu",
        max_seq_len=1024,
        model_id="t",
        audio_encoder=audio,
        vision_encoder=vision,
    )
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json=_mm_request())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["prompt_tokens"] > 0


def test_api_rejects_media_without_encoders(tok):
    from fastapi.testclient import TestClient

    from picochat.inference.api import create_app

    lm = TransformerLM(
        vocab_size=tok.n_vocab, d_model=D_MODEL, n_heads=4, n_layers=2
    ).eval()
    app = create_app(lm, tok, device="cpu", max_seq_len=1024, model_id="t")
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json=_mm_request())
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "unsupported_modality"
