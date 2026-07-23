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

Unlike Qwen2-Audio's frozen Whisper encoder, the encoder here is small and
trained from scratch (a conv for local context + frame-stacking + MLP): no
external weights, matching picochat's from-scratch philosophy, at a lower
quality ceiling.
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


def mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> Tensor:
    """Triangular mel filterbank, (n_mels, n_fft//2 + 1). HTK mel scale."""
    n_freqs = n_fft // 2 + 1
    freqs = torch.linspace(0.0, sample_rate / 2, n_freqs)
    mel_pts = torch.linspace(_hz_to_mel(0.0), _hz_to_mel(sample_rate / 2), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    fb = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        lower, center, upper = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        left = (freqs - lower) / (center - lower)
        right = (upper - freqs) / (upper - center)
        fb[i] = torch.clamp(torch.minimum(left, right), min=0.0)
    return fb


def log_mel_spectrogram(waveform: Tensor, cfg: AudioConfig) -> Tensor:
    """Waveform (n_samples,) or (B, n_samples) -> log-mel (B, n_mels, T), with
    Whisper's log10 + dynamic-range normalization to keep values ~[-1, 1]."""
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
    fb = mel_filterbank(cfg.n_mels, cfg.n_fft, cfg.sample_rate).to(waveform.device)
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
