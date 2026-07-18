"""Verify the audio input path: the log-mel front end, the from-scratch encoder
at ~100 ms/token, Qwen-style prompt rendering with expanded AUDIO placeholders,
and an end-to-end forward where gradients reach the encoder through the spliced
soft tokens."""

import torch

from picochat import audio
from picochat.audio import AudioConfig, AudioEncoder, AudioProcessor
from picochat.gpt import TransformerLM
from picochat.tokenizer import AUDIO_TOKEN, IMAGE_TOKEN, SPECIAL_TOKENS, train_tokenizer

SR = 16000


def test_downsample_matches_ms_per_token():
    assert AudioConfig(ms_per_token=100).downsample == 10  # 100ms / 10ms hop
    assert AudioConfig(ms_per_token=40).downsample == 4


def test_log_mel_shape():
    cfg = AudioConfig()
    wav = torch.randn(SR)  # 1 second
    mel = audio.log_mel_spectrogram(wav, cfg)
    assert mel.shape[0] == 1 and mel.shape[1] == cfg.n_mels
    # ~10 ms hop over 1 s -> ~101 frames
    assert 95 <= mel.shape[2] <= 105
    assert torch.isfinite(mel).all()


def test_encoder_output_length_is_100ms_per_token():
    cfg = AudioConfig(ms_per_token=100)
    enc = AudioEncoder(cfg, d_model=64)
    mel = torch.randn(1, cfg.n_mels, 101)  # ~1 s
    out = enc(mel)
    assert out.shape == (1, 101 // cfg.downsample, 64)  # ~10 tokens for 1 s


def test_processor_num_tokens_matches_encoder():
    cfg = AudioConfig(ms_per_token=100)
    proc = AudioProcessor(cfg)
    enc = AudioEncoder(cfg, d_model=32)
    mel = proc.mel(torch.randn(SR * 2))  # 2 s
    assert proc.num_tokens(mel.shape[-1]) == enc(mel).shape[1]


def test_scatter_audio_embeds_replaces_placeholder_positions():
    d = 8
    input_ids = torch.tensor([[5, 99, 99, 6, 99]])  # 99 == AUDIO placeholder
    text = torch.zeros(1, 5, d)
    audio_embeds = torch.arange(3 * d, dtype=torch.float).reshape(3, d)
    out = audio.scatter_audio_embeds(text, input_ids, audio_embeds, audio_token_id=99)
    assert torch.equal(out[0, 1], audio_embeds[0])
    assert torch.equal(out[0, 2], audio_embeds[1])
    assert torch.equal(out[0, 4], audio_embeds[2])
    assert torch.equal(out[0, 0], torch.zeros(d))  # non-placeholder untouched


def test_scatter_rejects_count_mismatch():
    input_ids = torch.tensor([[99, 99]])
    text = torch.zeros(1, 2, 4)
    try:
        audio.scatter_audio_embeds(text, input_ids, torch.zeros(1, 4), 99)
        assert False, "expected a mismatch error"
    except ValueError:
        pass


def _tokenizer(tmp_path):
    # A tiny real tokenizer so the audio special tokens get ids.
    corpus = ["hello world this is a tiny corpus for tests"] * 50
    path = tmp_path / "tok.json"
    train_tokenizer(
        iter(corpus), vocab_size=320, save_as=path, special_tokens=SPECIAL_TOKENS
    )
    from picochat.tokenizer import load_tokenizer

    return load_tokenizer(path)


def test_render_audio_prompt_expands_placeholders(tmp_path):
    tok = _tokenizer(tmp_path)
    proc = AudioProcessor(AudioConfig(ms_per_token=100))
    wav = torch.randn(SR)  # 1 s -> ~10 audio tokens
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this sound"},
                {"type": "audio", "audio": wav},
            ],
        }
    ]
    ids, mels = audio.render_audio_prompt(messages, tok, proc)
    assert len(mels) == 1
    n_audio = proc.num_tokens(mels[0].shape[-1])
    assert ids.count(tok.encode_single_token(AUDIO_TOKEN)) == n_audio


def test_image_tokens_exist_but_unused():
    # Reserved for future image support; distinct from the audio placeholder.
    assert IMAGE_TOKEN in SPECIAL_TOKENS
    assert IMAGE_TOKEN != AUDIO_TOKEN


def test_end_to_end_forward_grads_reach_encoder(tmp_path):
    torch.manual_seed(0)
    tok = _tokenizer(tmp_path)
    cfg = AudioConfig(ms_per_token=100)
    proc = AudioProcessor(cfg)
    d_model = 64
    lm = TransformerLM(vocab_size=tok.n_vocab, d_model=d_model, n_heads=4, n_layers=2)
    enc = AudioEncoder(cfg, d_model=d_model)

    messages = [
        {"role": "user", "content": [{"type": "audio", "audio": torch.randn(SR)}]}
    ]
    ids, mels = audio.render_audio_prompt(messages, tok, proc)
    input_ids = torch.tensor([ids])

    audio_embeds = enc(mels[0])[0]  # (n_audio, d_model)
    text_embeds = lm.embed(input_ids)
    merged = audio.scatter_audio_embeds(
        text_embeds, input_ids, audio_embeds, tok.encode_single_token(AUDIO_TOKEN)
    )

    logits = lm(inputs_embeds=merged)
    assert logits.shape == (1, len(ids), tok.n_vocab)
    logits.float().sum().backward()
    # gradient must flow back through the spliced soft tokens into the encoder
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
