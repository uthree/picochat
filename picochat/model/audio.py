"""Audio input (multimodal): a log-mel front end, a from-scratch audio encoder,
and the plumbing that splices its outputs into the token stream.

The integration follows the current de-facto (LLaVA / Qwen2-Audio): an encoder
turns audio into a short sequence of feature vectors, a projector maps them to
`d_model`, and those become *soft tokens* placed at AUDIO_TOKEN placeholder
positions in an otherwise ordinary ChatML sequence -- the transformer then
attends over text and audio uniformly (see TransformerLM's `inputs_embeds`
path). The wire format mirrors Qwen2-Audio: `<|audio_bos|>` + `<|AUDIO|>` * n +
`<|audio_eos|>`, where n is the clip's audio-token count.

Front end: 16 kHz, 128-mel, 25 ms window / 10 ms hop (Qwen2-Audio's numbers),
so mel frames land every 10 ms. `AudioConfig.ms_per_token` then sets how many
frames the encoder folds into one token -- 100 ms (the default) folds 10 frames,
i.e. ~10 audio tokens/second. That is coarser than Qwen2-Audio's ~40 ms (kinder
to a small model's context budget, fine for understanding rather than verbatim
transcription); lower it toward 40-80 ms if fidelity matters more than length.

Two encoders share that contract. AudioEncoder is small and trained from
scratch (a conv for local context + frame-stacking + MLP): no external
weights, matching picochat's from-scratch philosophy, at a lower quality
ceiling. WhisperAudioEncoder is an in-repo port of the OpenAI Whisper encoder
(Qwen2-Audio's choice) whose tower loads pretrained weights via
load_pretrained_audio_encoder while its projector trains fresh -- use
mel_scale="slaney" features with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from picochat.tokenizer import (
    AUDIO_BOS_TOKEN,
    AUDIO_EOS_TOKEN,
    AUDIO_TOKEN,
    BOS_TOKEN,
    render_turn,
)


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 400  # 25 ms window at 16 kHz
    hop_length: int = 160  # 10 ms hop at 16 kHz
    n_mels: int = 128
    ms_per_token: int = 100  # audio folded per output token (~10 tokens/sec)
    # "htk" (the from-scratch encoder's default) or "slaney". Whisper's feature
    # extractor uses librosa-style slaney-scale, slaney-normalized filters, so
    # AudioConfig(n_mels=80, mel_scale="slaney") reproduces WhisperFeatureExtractor
    # output for the pretrained WhisperAudioEncoder below.
    mel_scale: str = "htk"

    @property
    def downsample(self) -> int:
        """Mel frames folded into one audio token (ms_per_token / hop_ms)."""
        hop_ms = 1000 * self.hop_length / self.sample_rate
        ds = round(self.ms_per_token / hop_ms)
        if ds < 1:
            raise ValueError("ms_per_token is smaller than one mel frame")
        return ds


def _hz_to_mel(f: float) -> float:
    return 2595.0 * math.log10(1.0 + f / 700.0)


def _mel_to_hz(m: Tensor) -> Tensor:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


# Slaney mel scale (librosa's default, ported here so we do not depend on
# librosa): linear below 1 kHz at 200/3 mel per Hz, logarithmic above with 27
# steps per octave-of-6.4 (log(6.4)/27 per mel). These exact constants are what
# librosa -- and therefore OpenAI Whisper's feature extractor -- uses, so any
# deviation would break parity with pretrained Whisper weights.
_SLANEY_F_SP = 200.0 / 3.0
_SLANEY_MIN_LOG_HZ = 1000.0
_SLANEY_MIN_LOG_MEL = _SLANEY_MIN_LOG_HZ / _SLANEY_F_SP  # = 15.0
_SLANEY_LOGSTEP = math.log(6.4) / 27.0


def _hz_to_mel_slaney(f: float) -> float:
    if f < _SLANEY_MIN_LOG_HZ:
        return f / _SLANEY_F_SP
    return _SLANEY_MIN_LOG_MEL + math.log(f / _SLANEY_MIN_LOG_HZ) / _SLANEY_LOGSTEP


def _mel_to_hz_slaney(m: Tensor) -> Tensor:
    linear = m * _SLANEY_F_SP
    log = _SLANEY_MIN_LOG_HZ * torch.exp(_SLANEY_LOGSTEP * (m - _SLANEY_MIN_LOG_MEL))
    return torch.where(m < _SLANEY_MIN_LOG_MEL, linear, log)


def mel_filterbank(
    n_mels: int, n_fft: int, sample_rate: int, scale: str = "htk"
) -> Tensor:
    """Triangular mel filterbank, (n_mels, n_fft//2 + 1).

    scale="htk": HTK mel scale, unnormalized triangles (the from-scratch path).
    scale="slaney": slaney mel scale with slaney area normalization (each
    triangle scaled by 2 / bandwidth), matching librosa/Whisper. Triangles are
    built in Hz space in both cases (librosa's default too)."""
    if scale not in ("htk", "slaney"):
        raise ValueError(f"unknown mel scale: {scale!r}")
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0.0, sample_rate / 2, n_freqs)
    hz_to_mel = _hz_to_mel if scale == "htk" else _hz_to_mel_slaney
    mel_to_hz = _mel_to_hz if scale == "htk" else _mel_to_hz_slaney
    mel_pts = torch.linspace(
        hz_to_mel(0.0), hz_to_mel(sample_rate / 2), n_mels + 2, dtype=torch.float64
    )
    hz_pts = mel_to_hz(mel_pts)
    fb = torch.zeros(n_mels, n_freqs, dtype=torch.float64)
    freqs = freqs.double()
    for i in range(n_mels):
        lower, center, upper = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        left = (freqs - lower) / (center - lower)
        right = (upper - freqs) / (upper - center)
        fb[i] = torch.clamp(torch.minimum(left, right), min=0.0)
        if scale == "slaney":
            # area normalization: peak height 2 / bandwidth, so each filter
            # integrates to ~the same energy regardless of its width
            fb[i] *= 2.0 / (upper - lower)
    return fb.float()


def log_mel_spectrogram(waveform: Tensor, cfg: AudioConfig) -> Tensor:
    """Waveform (n_samples,) or (B, n_samples) -> log-mel (B, n_mels, T), with
    Whisper's log10 + dynamic-range normalization to keep values ~[-1, 1].

    Deviations from Whisper's own feature extractor, both deliberate: (1) we do
    not pad/truncate to 30 s -- clips stay their natural length and the encoder
    slices its positional embedding instead (see WhisperAudioEncoder); (2) we
    keep every centered STFT frame (T = n_samples // hop + 1) where Whisper
    drops the last one (T = n_samples // hop). Frame t is bit-identical between
    the two for the frames both produce, so with mel_scale="slaney" the shared
    region matches WhisperFeatureExtractor output exactly."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    window = torch.hann_window(cfg.n_fft, device=waveform.device)
    stft = torch.stft(
        waveform,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window=window,
        center=True,
        return_complex=True,
    )
    power = stft.abs() ** 2  # (B, n_freqs, T)
    fb = mel_filterbank(cfg.n_mels, cfg.n_fft, cfg.sample_rate, cfg.mel_scale).to(
        waveform.device
    )
    mel = torch.einsum("mf,bft->bmt", fb, power)
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0)
    return (log_spec + 4.0) / 4.0


class AudioProcessor:
    """Front end + token bookkeeping shared by data prep and inference: turns a
    waveform into log-mel features and reports how many audio tokens a clip of
    a given mel length becomes (so the AUDIO placeholder is repeated exactly
    that many times)."""

    def __init__(self, cfg: AudioConfig | None = None):
        self.cfg = cfg or AudioConfig()

    def mel(self, waveform: Tensor) -> Tensor:
        """(n_samples,) -> (n_mels, T)."""
        return log_mel_spectrogram(waveform, self.cfg)[0]

    def num_tokens(self, n_frames: int) -> int:
        return n_frames // self.cfg.downsample


class AudioEncoder(nn.Module):
    """From-scratch audio encoder: a depthwise-ish conv gives each mel frame
    local temporal context, then `downsample` frames are stacked and an MLP
    projects the stack to one `d_model` soft token. Input (B, n_mels, T) ->
    output (B, T // downsample, d_model)."""

    def __init__(self, cfg: AudioConfig, d_model: int, d_hidden: int | None = None):
        super().__init__()
        self.cfg = cfg
        # The front end this encoder expects, exposed so rendering code can ask
        # the encoder itself for features and token counts (multimodal.py does
        # `enc.processor.mel(wav)`) instead of threading a config alongside.
        self.processor = AudioProcessor(cfg)
        self.downsample = cfg.downsample
        d_hidden = d_hidden or 4 * d_model
        self.conv = nn.Conv1d(cfg.n_mels, cfg.n_mels, kernel_size=3, padding=1)
        self.proj = nn.Sequential(
            nn.Linear(cfg.n_mels * self.downsample, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, mel: Tensor) -> Tensor:
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        x = F.gelu(self.conv(mel))  # (B, n_mels, T)
        b, c, t = x.shape
        ds = self.downsample
        t2 = t - (t % ds)  # drop a trailing partial token
        # stack `ds` consecutive frames along the feature dim
        x = x[:, :, :t2].reshape(b, c, t2 // ds, ds)
        x = x.permute(0, 2, 1, 3).reshape(b, t2 // ds, c * ds)
        return self.proj(x)  # (B, T//ds, d_model)


@dataclass
class WhisperEncoderConfig:
    """Architecture of a pretrained OpenAI Whisper *encoder* (defaults are
    whisper-small). Field names mirror the concepts, not the HF config keys;
    load_pretrained_audio_encoder translates from the checkpoint's config.json
    (num_mel_bins / d_model / encoder_layers / encoder_attention_heads /
    encoder_ffn_dim / max_source_positions)."""

    n_mels: int = 80
    d_encoder: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ffn: int = 3072
    max_frames: int = 1500  # post-conv frames for 30 s (Whisper's fixed window)


def _sinusoid_positions(length: int, channels: int) -> Tensor:
    """Whisper's fixed sinusoidal positions: sin over the first channels//2
    dims, cos over the rest (concatenated, NOT interleaved -- this layout must
    match the checkpoint's embed_positions or every pretrained weight after the
    convs is attending over garbage). Timescales are log-spaced 1..10000."""
    if channels % 2 != 0:
        raise ValueError("sinusoidal positions need an even channel count")
    log_timescale_increment = math.log(10000.0) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length).float()[:, None] * inv_timescales[None, :]
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


class _WhisperSelfAttention(nn.Module):
    """Whisper encoder self-attention. One checkpoint quirk decides the shape
    of this module: k_proj has NO bias while q/v/out do (the key bias is
    redundant under softmax -- a constant added to every logit -- so Whisper
    omits it, and our parameter set must agree with the checkpoint's)."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError("d_encoder must be divisible by n_heads")
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        shape = (b, t, self.n_heads, d // self.n_heads)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)  # bidirectional: no mask
        return self.out_proj(o.transpose(1, 2).reshape(b, t, d))


class _WhisperEncoderLayer(nn.Module):
    """Pre-LN transformer block, exactly Whisper's: LN -> attn -> residual,
    LN -> Linear -> GELU -> Linear -> residual. GELU is the exact (erf) form,
    which is what both openai/whisper and the HF port use."""

    def __init__(self, d: int, n_heads: int, d_ffn: int):
        super().__init__()
        self.attn_ln = nn.LayerNorm(d)
        self.attn = _WhisperSelfAttention(d, n_heads)
        self.mlp_ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_ln(x))
        return x + self.fc2(F.gelu(self.fc1(self.mlp_ln(x))))


class WhisperAudioEncoder(nn.Module):
    """In-repo port of the OpenAI Whisper encoder plus a fresh adapter that
    turns its 50 fps output into `d_model` soft tokens at `ms_per_token`.

    Tower (loadable from a Whisper checkpoint): conv1 k3 s1 (n_mels -> d) +
    GELU, conv2 k3 s2 (d -> d) + GELU, fixed sinusoidal positions, n_layers
    pre-LN blocks, final LayerNorm. Adapter (always freshly initialized): fold
    `frames_per_token` consecutive encoder frames by reshape-concat, then a
    2-layer GELU MLP to d_model -- the same stack-and-project scheme as the
    from-scratch AudioEncoder, so downstream code treats both identically.

    Deviation from Whisper's training regime: Whisper always sees exactly 30 s
    (3000 mel frames -> 1500 encoder frames) and its positional embedding is
    only ever used in full. We instead run on the clip's natural length and
    slice the positional embedding to the actual post-conv frame count. That is
    out-of-distribution for the pretrained attention layers, but for a soft-
    token front end (understanding, not verbatim transcription) it works well
    in practice and saves quadratic attention cost on 30x-padded silence; the
    adapter + finetuning absorb the shift.
    """

    def __init__(
        self, cfg: WhisperEncoderConfig, d_model: int, ms_per_token: int = 100
    ):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_encoder
        # Encoder frames sit 2 mel hops apart (the stride-2 conv2); at
        # Whisper's fixed 16 kHz / 160-sample hop that is 20 ms per frame.
        self.frames_per_token = round(ms_per_token / 20)
        if self.frames_per_token < 1:
            raise ValueError("ms_per_token is smaller than one encoder frame (20 ms)")
        # The front end this encoder expects (same contract as AudioEncoder's
        # `processor`): Whisper's mel count and slaney filters over the default
        # 16 kHz / 400 / 160 STFT, which are also Whisper's numbers. Use
        # processor.mel for features but *this module's* num_tokens for counts
        # (the fold here runs on post-conv frames, not raw mel frames).
        self.processor = AudioProcessor(
            AudioConfig(
                n_mels=cfg.n_mels, ms_per_token=ms_per_token, mel_scale="slaney"
            )
        )
        self.conv1 = nn.Conv1d(cfg.n_mels, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)
        # Persistent buffer (not a Parameter: Whisper's positions are fixed,
        # never trained) so the checkpoint's embed_positions loads straight
        # into it through load_state_dict -- identical to our sinusoids, but
        # loading it keeps the parity guarantee mechanical rather than assumed.
        self.register_buffer(
            "positional_embedding", _sinusoid_positions(cfg.max_frames, d)
        )
        self.layers = nn.ModuleList(
            _WhisperEncoderLayer(d, cfg.n_heads, cfg.d_ffn) for _ in range(cfg.n_layers)
        )
        self.ln_post = nn.LayerNorm(d)
        # The adapter. Stage-1 training freezes everything above and trains
        # only this, so it must stay a single cleanly-separable submodule named
        # `projector` (see tower_parameters).
        self.projector = nn.Sequential(
            nn.Linear(d * self.frames_per_token, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def tower_parameters(self):
        """Parameters of the pretrained Whisper tower -- everything except the
        fresh projector. This is the set to freeze (or LR-scale) when adapting:
        together with projector.parameters() it partitions parameters()."""
        return (
            p
            for name, p in self.named_parameters()
            if not name.startswith("projector.")
        )

    def num_tokens(self, n_mel_frames: int) -> int:
        """Soft tokens produced for a clip of n_mel_frames mel frames; trailing
        encoder frames short of a full fold are dropped (like AudioEncoder)."""
        t_enc = (n_mel_frames - 1) // 2 + 1 if n_mel_frames > 0 else 0  # conv2 k3 s2 p1
        return t_enc // self.frames_per_token

    def forward(self, mel: Tensor) -> Tensor:
        """(n_mels, T) or (B, n_mels, T) -> (B, num_tokens(T), d_model)."""
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))  # (B, d, T'), T' = (T-1)//2 + 1
        x = x.permute(0, 2, 1)  # (B, T', d)
        b, t, d = x.shape
        if t > self.positional_embedding.shape[0]:
            raise ValueError(
                f"{t} encoder frames exceed max_frames={self.positional_embedding.shape[0]}"
                " (clip longer than Whisper's 30 s window -- chunk it upstream)"
            )
        x = x + self.positional_embedding[:t]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_post(x)
        fold = self.frames_per_token
        n_tok = t // fold  # drop a trailing partial token
        x = x[:, : n_tok * fold].reshape(b, n_tok, fold * d)
        return self.projector(x)


# HF checkpoint layout -> our module names, for the per-layer suffixes. Kept as
# a table (rather than clever string surgery) so mapping drift fails loudly in
# load_whisper_encoder_state instead of silently skipping a weight.
_HF_LAYER_KEY_MAP = {
    "self_attn.q_proj": "attn.q_proj",
    "self_attn.k_proj": "attn.k_proj",
    "self_attn.v_proj": "attn.v_proj",
    "self_attn.out_proj": "attn.out_proj",
    "self_attn_layer_norm": "attn_ln",
    "fc1": "fc1",
    "fc2": "fc2",
    "final_layer_norm": "mlp_ln",
}


def _map_hf_encoder_key(key: str) -> str | None:
    """Map one HF Whisper checkpoint key (model.encoder.*) to this module's
    parameter name. Returns None for keys that are not encoder weights (the
    decoder, proj_out, ...). Raises on encoder keys it does not recognize."""
    prefix = "model.encoder."
    if not key.startswith(prefix):
        return None
    rest = key[len(prefix) :]
    if rest == "embed_positions.weight":
        return "positional_embedding"
    head = rest.split(".", 1)[0]
    if head in ("conv1", "conv2"):
        return rest
    if head == "layer_norm":  # the final LayerNorm, after all blocks
        return "ln_post." + rest.split(".", 1)[1]
    if head == "layers":
        _, idx, sub = rest.split(".", 2)
        stem, leaf = sub.rsplit(".", 1)  # e.g. self_attn.q_proj + weight
        if stem in _HF_LAYER_KEY_MAP:
            return f"layers.{idx}.{_HF_LAYER_KEY_MAP[stem]}.{leaf}"
    raise KeyError(f"unrecognized Whisper encoder key: {key!r}")


def load_whisper_encoder_state(
    encoder: WhisperAudioEncoder, hf_state: dict[str, Tensor]
) -> None:
    """Load the encoder tower from an HF-layout Whisper state dict (full-model
    keys, `model.encoder.*`). The projector stays freshly initialized. Fails
    loudly if any encoder key cannot be mapped or any tower weight is left
    unloaded -- silent partial loads are how pretrained-parity quietly rots."""
    mapped: dict[str, Tensor] = {}
    for key, value in hf_state.items():
        local = _map_hf_encoder_key(key)
        if local is not None:
            mapped[local] = value
    missing, unexpected = encoder.load_state_dict(mapped, strict=False)
    if unexpected:
        raise RuntimeError(
            f"mapped keys not present in WhisperAudioEncoder: {unexpected}"
        )
    not_projector = [k for k in missing if not k.startswith("projector.")]
    if not_projector:
        raise RuntimeError(
            f"encoder weights left unloaded from checkpoint: {not_projector}"
        )


def load_pretrained_audio_encoder(
    d_model: int, repo_id: str = "openai/whisper-small", ms_per_token: int = 100
) -> WhisperAudioEncoder:
    """Download a Whisper checkpoint from the HF Hub and build a
    WhisperAudioEncoder with its encoder tower pretrained and a fresh
    projector. Only the encoder half of the seq2seq checkpoint is used; feed it
    features from AudioConfig(n_mels=cfg.n_mels, mel_scale="slaney")."""
    import json

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    with open(hf_hub_download(repo_id, "config.json")) as f:
        hf_cfg = json.load(f)
    cfg = WhisperEncoderConfig(
        n_mels=hf_cfg["num_mel_bins"],
        d_encoder=hf_cfg["d_model"],
        n_layers=hf_cfg["encoder_layers"],
        n_heads=hf_cfg["encoder_attention_heads"],
        d_ffn=hf_cfg["encoder_ffn_dim"],
        max_frames=hf_cfg["max_source_positions"],
    )
    encoder = WhisperAudioEncoder(cfg, d_model=d_model, ms_per_token=ms_per_token)
    state = load_file(hf_hub_download(repo_id, "model.safetensors"))
    load_whisper_encoder_state(encoder, state)
    return encoder


def scatter_audio_embeds(
    text_embeds: Tensor, input_ids: Tensor, audio_embeds: Tensor, audio_token_id: int
) -> Tensor:
    """Replace the embeddings at AUDIO_TOKEN positions with the encoder's audio
    soft tokens. `text_embeds` (B, L, d) is the token embedding of `input_ids`;
    `audio_embeds` (n_audio, d) is every clip's tokens concatenated in the order
    they appear. The number of AUDIO placeholders must equal n_audio."""
    mask = input_ids == audio_token_id
    n_slots = int(mask.sum())
    if n_slots != audio_embeds.shape[0]:
        raise ValueError(
            f"{n_slots} AUDIO placeholders but {audio_embeds.shape[0]} audio tokens"
        )
    out = text_embeds.clone()
    out[mask] = audio_embeds.to(out.dtype)
    return out


def render_audio_prompt(
    messages: list[dict],
    tokenizer,
    processor: AudioProcessor,
) -> tuple[list[int], list[Tensor]]:
    """Qwen2-Audio-style prompt rendering. Each message's `content` is either a
    string or a list of parts: {"type": "text", "text": ...} and
    {"type": "audio", "audio": <1-D waveform tensor>}. Every audio part expands
    to `<|audio_bos|>` + `<|AUDIO|>` * n + `<|audio_eos|>` (n = the clip's audio
    tokens). Returns (input_ids, mels) where mels are the per-clip log-mel
    features in placeholder order, ready for AudioEncoder + scatter_audio_embeds.

    The ChatML frame (leading BOS, `<|im_start|>{role}\\n...<|im_end|>\\n` per
    turn, trailing bare assistant header) comes from
    picochat.tokenizer.render_turn -- the same spans render_chat_prompt emits
    -- so this renderer can never drift from the text-only chat format; only
    the audio-part expansion is added here.
    """
    sp = tokenizer.encode_single_token
    ids = [sp(BOS_TOKEN)]
    mels: list[Tensor] = []
    for msg in messages:
        # render_turn with empty content: header = <|im_start|>{role}\n,
        # closing = <|im_end|>, tail = \n. The parts loop fills the body.
        header, closing, tail = render_turn(msg["role"], "", tokenizer)
        ids += header
        content = msg["content"]
        parts = (
            [{"type": "text", "text": content}] if isinstance(content, str) else content
        )
        for part in parts:
            if part["type"] == "text":
                ids += tokenizer.encode_ordinary(part["text"])
            elif part["type"] == "audio":
                mel = processor.mel(part["audio"])
                n = processor.num_tokens(mel.shape[-1])
                ids += [
                    sp(AUDIO_BOS_TOKEN),
                    *([sp(AUDIO_TOKEN)] * n),
                    sp(AUDIO_EOS_TOKEN),
                ]
                mels.append(mel)
            else:
                raise ValueError(f"unknown content part type: {part['type']!r}")
        ids += closing + tail
    header, _, _ = render_turn("assistant", "", tokenizer)
    ids += header
    return ids, mels
