"""Verify the audio input path: the log-mel front end, the from-scratch encoder
at ~100 ms/token, Qwen-style prompt rendering with expanded AUDIO placeholders,
and an end-to-end forward where gradients reach the encoder through the spliced
soft tokens."""

import pytest
import torch

from picochat.model import audio
from picochat.model.audio import (
    AudioConfig,
    AudioEncoder,
    AudioProcessor,
    WhisperAudioEncoder,
    WhisperEncoderConfig,
    load_whisper_encoder_state,
    mel_filterbank,
)
from picochat.model import TransformerLM
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


def test_downsample_rejects_sub_frame_ms_per_token():
    # 5 ms per token < the 10 ms mel hop: zero frames per token is impossible
    with pytest.raises(ValueError):
        AudioConfig(ms_per_token=5).downsample


def test_render_audio_prompt_rejects_unknown_part_type(tmp_path):
    tok = _tokenizer(tmp_path)
    proc = AudioProcessor(AudioConfig(ms_per_token=100))
    messages = [{"role": "user", "content": [{"type": "video", "video": None}]}]
    with pytest.raises(ValueError, match="unknown content part type"):
        audio.render_audio_prompt(messages, tok, proc)


def test_slaney_filterbank_shape_and_envelope():
    fb = mel_filterbank(80, 400, SR, scale="slaney")
    assert fb.shape == (80, 201)
    assert torch.isfinite(fb).all() and (fb >= 0).all()
    # slaney normalization: every filter has positive area, and the per-bin
    # column sums form a smooth positive envelope over the covered band (no
    # dead bins between the first and last filter edges)
    row_sums = fb.sum(dim=1)
    assert (row_sums > 0).all()
    col_sums = fb.sum(dim=0)
    inner = col_sums[2:-1]  # bins strictly inside the filterbank's support
    assert (inner > 0).all()
    # area normalization makes the envelope smooth: flat over the linear
    # (< 1 kHz) region, then gently decaying through the log region -- never
    # jumping bin to bin
    linear_region = inner[: 900 // 40]  # 40 Hz bins, stay below the 1 kHz knee
    assert linear_region.max() / linear_region.min() < 1.05
    assert (inner.diff() <= 2e-3 * inner.max()).all()  # monotone-ish decay


def test_mel_filterbank_rejects_unknown_scale():
    with pytest.raises(ValueError, match="unknown mel scale"):
        mel_filterbank(80, 400, SR, scale="mystery")


# Tiny config so Whisper-encoder tests run on random weights with no network.
_TINY = dict(n_mels=8, d_encoder=16, n_layers=2, n_heads=2, d_ffn=32, max_frames=100)


def test_whisper_encoder_shapes_and_num_tokens():
    cfg = WhisperEncoderConfig(**_TINY)
    enc = WhisperAudioEncoder(cfg, d_model=24, ms_per_token=100)
    assert enc.frames_per_token == 5  # 100 ms / 20 ms encoder frames
    # odd and even mel lengths, including a trailing partial fold to drop
    for t in (101, 100, 57, 10, 9):
        out = enc(torch.randn(2, cfg.n_mels, t))
        n = enc.num_tokens(t)
        assert out.shape == (2, n, 24)
        t_enc = (t - 1) // 2 + 1  # conv2 k3 s2 p1 output length
        assert n == t_enc // 5


def test_whisper_encoder_accepts_unbatched_and_variable_length():
    cfg = WhisperEncoderConfig(**_TINY)
    enc = WhisperAudioEncoder(cfg, d_model=12)
    a = enc(torch.randn(cfg.n_mels, 73))
    b = enc(torch.randn(cfg.n_mels, 199))
    assert a.shape == (1, enc.num_tokens(73), 12)
    assert b.shape == (1, enc.num_tokens(199), 12)
    # longer than max_frames post-conv must fail loudly, not attend garbage
    with pytest.raises(ValueError, match="max_frames"):
        enc(torch.randn(cfg.n_mels, 2 * cfg.max_frames + 3))


def test_whisper_encoder_gradients_flow_to_tower_and_projector():
    cfg = WhisperEncoderConfig(**_TINY)
    enc = WhisperAudioEncoder(cfg, d_model=16)
    enc(torch.randn(1, cfg.n_mels, 60)).sum().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, name
    assert any(p.grad.abs().sum() > 0 for p in enc.tower_parameters())
    assert any(p.grad.abs().sum() > 0 for p in enc.projector.parameters())


def test_whisper_tower_and_projector_partition_parameters():
    enc = WhisperAudioEncoder(WhisperEncoderConfig(**_TINY), d_model=16)
    tower = {id(p) for p in enc.tower_parameters()}
    proj = {id(p) for p in enc.projector.parameters()}
    every = {id(p) for p in enc.parameters()}
    assert tower.isdisjoint(proj)
    assert tower | proj == every


def test_encoder_processor_attribute_consistency():
    # Both encoder classes carry the front end they expect; token counts from
    # the encoder must match the actual forward output on processor features.
    wav = torch.randn(SR)  # 1 s
    cfg = AudioConfig(ms_per_token=100)
    scratch = AudioEncoder(cfg, d_model=16)
    mel = scratch.processor.mel(wav)
    assert scratch.processor.cfg is cfg
    assert scratch(mel).shape[1] == scratch.processor.num_tokens(mel.shape[-1])

    wcfg = WhisperEncoderConfig(**_TINY)
    whisper = WhisperAudioEncoder(wcfg, d_model=16, ms_per_token=100)
    assert whisper.processor.cfg.n_mels == wcfg.n_mels
    assert whisper.processor.cfg.mel_scale == "slaney"
    wmel = whisper.processor.mel(wav)
    assert wmel.shape[0] == wcfg.n_mels
    assert whisper(wmel).shape[1] == whisper.num_tokens(wmel.shape[-1])


def _local_to_hf_key(local: str) -> str:
    """Inverse of the loader's mapping, for fabricating an HF-style checkpoint."""
    if local == "positional_embedding":
        return "model.encoder.embed_positions.weight"
    for ours, hf in {
        "ln_post.": "layer_norm.",
        "attn.q_proj": "self_attn.q_proj",
        "attn.k_proj": "self_attn.k_proj",
        "attn.v_proj": "self_attn.v_proj",
        "attn.out_proj": "self_attn.out_proj",
        "attn_ln": "self_attn_layer_norm",
        "mlp_ln": "final_layer_norm",
    }.items():
        local = local.replace(ours, hf)
    return "model.encoder." + local


def test_whisper_loader_maps_hf_keys(tmp_path):
    from safetensors.torch import load_file, save_file

    cfg = WhisperEncoderConfig(**_TINY)
    torch.manual_seed(0)
    source = WhisperAudioEncoder(cfg, d_model=16)  # plays the pretrained model
    torch.manual_seed(1)
    target = WhisperAudioEncoder(cfg, d_model=16)

    # fabricate a full-model HF checkpoint: encoder tower under HF names, plus
    # a decoder weight the loader must ignore
    hf_state = {
        _local_to_hf_key(k): v.clone()
        for k, v in source.state_dict().items()
        if not k.startswith("projector.")
    }
    assert "model.encoder.layers.1.self_attn.k_proj.weight" in hf_state  # mapped right
    hf_state["model.decoder.embed_tokens.weight"] = torch.zeros(4, cfg.d_encoder)
    path = tmp_path / "model.safetensors"
    save_file(hf_state, str(path))

    projector_before = [p.clone() for p in target.projector.parameters()]
    load_whisper_encoder_state(target, load_file(str(path)))

    src, tgt = source.state_dict(), target.state_dict()
    for k in src:
        if k.startswith("projector."):
            continue
        assert torch.equal(src[k], tgt[k]), k  # every tower weight loaded
    for before, after in zip(projector_before, target.projector.parameters()):
        assert torch.equal(before, after)  # adapter untouched by the loader


def test_whisper_loader_fails_loudly_on_mapping_drift(tmp_path):
    cfg = WhisperEncoderConfig(**_TINY)
    enc = WhisperAudioEncoder(cfg, d_model=16)
    # an encoder key the mapping does not know must raise, not be skipped
    with pytest.raises(KeyError, match="unrecognized"):
        load_whisper_encoder_state(enc, {"model.encoder.mystery.weight": torch.ones(1)})
    # a known key with a missing sibling (incomplete checkpoint) must also fail
    with pytest.raises(RuntimeError, match="left unloaded"):
        load_whisper_encoder_state(
            enc, {"model.encoder.conv1.weight": torch.randn(16, 8, 3)}
        )


def test_decode_prefill_from_inputs_embeds_matches_forward():
    # The audio-conditioned generation path: prefill the KV cache from spliced
    # embeddings instead of token ids, then check the logits agree with a plain
    # forward over the same embeddings.
    torch.manual_seed(0)
    lm = TransformerLM(vocab_size=32, d_model=16, n_heads=2, n_layers=2).eval()
    ids = torch.randint(0, 32, (1, 8))
    embeds = lm.embed(ids)
    with torch.no_grad():
        ref = lm(inputs_embeds=embeds)
        logits, cache, pos = lm.decode(inputs_embeds=embeds)
    assert pos == 8
    assert len(cache) > 0
    torch.testing.assert_close(logits, ref, rtol=1e-4, atol=1e-5)
