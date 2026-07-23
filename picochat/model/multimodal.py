"""Multimodal conversation assembly: the one place that turns part-structured
chat messages (text / audio / image parts) into token ids plus the media
features whose soft tokens replace the placeholder embeddings.

A message's `content` is either a plain string (text-only, exactly the
tokenizer.py ChatML path) or a list of parts:

    {"type": "text",  "text":  str}
    {"type": "audio", "audio": 1-D waveform Tensor (16 kHz mono)}
    {"type": "image", "image": PIL.Image | (3, H, W) Tensor}

An audio part renders as `<|audio_bos|>` + `<|AUDIO|>` * n + `<|audio_eos|>`
and an image part as `<|vision_start|>` + `<|image_pad|>` * n +
`<|vision_end|>` (the Qwen2-Audio / Qwen2-VL wire format); the embeddings at
the placeholder positions are replaced by the encoders' soft tokens via
scatter_embeds before the transformer runs (TransformerLM's `inputs_embeds`
path). The ChatML frame itself (BOS, turn headers, `<|im_end|>`, the
assistant cue) comes from picochat.tokenizer.render_turn, so this module can
never drift from the text-only chat format.

Placeholder counts must match the encoders exactly, so rendering is
parameterized by a MediaAdapters bundle built from the actual encoders in
use (see from_encoders); the data pipeline and the inference server both go
through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from picochat.tokenizer import (
    AUDIO_BOS_TOKEN,
    AUDIO_EOS_TOKEN,
    AUDIO_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    IMAGE_TOKEN,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
    Tokenizer,
    render_turn,
)


@dataclass
class MediaAdapters:
    """How rendering turns raw media into (features, placeholder count).

    audio_processor: waveform -> log-mel features (picochat.model.audio
      .AudioProcessor); None -> audio parts are rejected.
    audio_num_tokens: mel-frame count -> soft-token count. This must be the
      *encoder's* arithmetic (the from-scratch encoder folds mel frames
      directly, the Whisper encoder halves them in its conv stem first), so
      it is taken from the encoder, not derived here.
    preprocess_image: PIL.Image | Tensor -> (3, S, S) normalized pixels
      (picochat.model.vision.preprocess_image); None -> image parts rejected.
    image_tokens: soft tokens per image (the vision encoder's
      tokens_per_image -- fixed-resolution encoders emit a constant count).
    """

    audio_processor: object | None = None
    audio_num_tokens: Callable[[int], int] | None = None
    preprocess_image: Callable[[object], Tensor] | None = None
    image_tokens: int | None = None

    @classmethod
    def from_encoders(cls, audio_encoder=None, vision_encoder=None) -> "MediaAdapters":
        """Bundle the adapters off the actual encoder modules: audio encoders
        expose `.processor` (an AudioProcessor) and `.num_tokens`; vision
        encoders expose `.preprocess` and `.tokens_per_image`."""
        kwargs = {}
        if audio_encoder is not None:
            kwargs["audio_processor"] = audio_encoder.processor
            kwargs["audio_num_tokens"] = audio_encoder.num_tokens
        if vision_encoder is not None:
            kwargs["preprocess_image"] = vision_encoder.preprocess
            kwargs["image_tokens"] = vision_encoder.tokens_per_image
        return cls(**kwargs)


def _as_parts(content) -> list[dict]:
    return [{"type": "text", "text": content}] if isinstance(content, str) else content


def render_parts(
    content,
    tokenizer: Tokenizer,
    media: MediaAdapters,
) -> tuple[list[int], list[bool], list[Tensor], list[Tensor]]:
    """One message body's parts -> (ids, is_text, mels, images). `is_text`
    marks, per token, whether it came from a text part -- the loss mask keeps
    only those (plus <|im_end|>) trainable in assistant turns; placeholder
    spans are inputs, never targets."""
    sp = tokenizer.encode_single_token
    ids: list[int] = []
    is_text: list[bool] = []
    mels: list[Tensor] = []
    images: list[Tensor] = []
    for part in _as_parts(content):
        if part["type"] == "text":
            text_ids = tokenizer.encode_ordinary(part["text"])
            ids += text_ids
            is_text += [True] * len(text_ids)
        elif part["type"] == "audio":
            if media.audio_processor is None or media.audio_num_tokens is None:
                raise ValueError("audio part but no audio encoder is configured")
            mel = media.audio_processor.mel(part["audio"])
            n = media.audio_num_tokens(mel.shape[-1])
            span = [sp(AUDIO_BOS_TOKEN), *([sp(AUDIO_TOKEN)] * n), sp(AUDIO_EOS_TOKEN)]
            ids += span
            is_text += [False] * len(span)
            mels.append(mel)
        elif part["type"] == "image":
            if media.preprocess_image is None or media.image_tokens is None:
                raise ValueError("image part but no vision encoder is configured")
            pixels = media.preprocess_image(part["image"])
            span = [
                sp(VISION_START_TOKEN),
                *([sp(IMAGE_TOKEN)] * media.image_tokens),
                sp(VISION_END_TOKEN),
            ]
            ids += span
            is_text += [False] * len(span)
            images.append(pixels)
        else:
            raise ValueError(f"unknown content part type: {part['type']!r}")
    return ids, is_text, mels, images


def render_mm_prompt(
    messages: list[dict],
    tokenizer: Tokenizer,
    media: MediaAdapters,
) -> tuple[list[int], list[Tensor], list[Tensor]]:
    """Token ids of a part-structured conversation ready for generation, plus
    the media features in placeholder order: `<|begin_of_text|>`, every turn,
    then the bare assistant header to cue the reply -- the multimodal analogue
    of tokenizer.render_chat_prompt (and identical to it for text-only
    messages)."""
    ids = [tokenizer.encode_single_token(BOS_TOKEN)]
    mels: list[Tensor] = []
    images: list[Tensor] = []
    for msg in messages:
        header, closing, tail = render_turn(msg["role"], "", tokenizer)
        body_ids, _, m, im = render_parts(msg["content"], tokenizer, media)
        ids += header + body_ids + closing + tail
        mels += m
        images += im
    header, _, _ = render_turn("assistant", "", tokenizer)
    ids += header
    return ids, mels, images


def encode_mm_conversation(
    messages: list[dict],
    tokenizer: Tokenizer,
    max_length: int,
    pad_id: int,
    media: MediaAdapters,
) -> tuple[list[int], list[int], list[Tensor], list[Tensor]] | None:
    """Tokenize one part-structured conversation into (input_ids, labels,
    mels, images) for SFT -- the multimodal analogue of
    tokenizer.encode_conversation, with the same loss-mask semantics: only
    assistant *text* (content + `<|im_end|>`) is trainable; headers, media
    placeholder spans and document delimiters get `pad_id` labels.

    One deliberate difference: truncation is per *turn*, not per token --
    a turn that would push the sequence past max_length is dropped along with
    everything after it. Token-level truncation could split a media
    placeholder span, which would desynchronize the placeholder count from
    the media features and break scatter_embeds. Returns None if nothing
    trainable survives."""
    bos = tokenizer.encode_single_token(BOS_TOKEN)
    eos = tokenizer.encode_single_token(EOS_TOKEN)
    input_ids: list[int] = [bos]
    labels: list[int] = [pad_id]
    mels: list[Tensor] = []
    images: list[Tensor] = []
    truncated = False
    for msg in messages:
        header, closing, tail = render_turn(msg["role"], "", tokenizer)
        body_ids, is_text, m, im = render_parts(msg["content"], tokenizer, media)
        turn_len = len(header) + len(body_ids) + len(closing) + len(tail)
        # +1 leaves room for the closing <|end_of_text|> of an untruncated
        # conversation, so the turn either fits whole or is dropped whole.
        if len(input_ids) + turn_len + 1 > max_length:
            truncated = True
            break
        is_assistant = msg["role"] == "assistant"
        input_ids += header + body_ids + closing + tail
        labels += [pad_id] * len(header)
        labels += [
            tok if (is_assistant and text) else pad_id
            for tok, text in zip(body_ids, is_text)
        ]
        labels += (closing if is_assistant else [pad_id] * len(closing)) + [
            pad_id
        ] * len(tail)
        mels += m
        images += im
    if not truncated:  # close the document like the pretraining corpus does
        input_ids.append(eos)
        labels.append(pad_id)
    if all(label == pad_id for label in labels):
        return None  # nothing survived to train on
    return input_ids, labels, mels, images


def build_encoders(spec: dict | None, d_model: int):
    """Build the media encoders from a config's `multimodal:` section and
    return (audio_encoder, vision_encoder, mm_config), where mm_config is the
    fully-resolved plain-dict recipe SFTModule saves in its hyper_parameters
    so inference can rebuild the same architectures from the checkpoint alone
    (training.checkpoint.load_mm_encoders) -- no Hub access at serve time.

    spec:
        audio:
            pretrained: openai/whisper-small   # or null -> from-scratch
            ms_per_token: 100
        vision:
            pretrained: google/siglip2-base-patch16-256   # or null -> random
    """
    from dataclasses import asdict

    audio_encoder = vision_encoder = None
    mm_config: dict = {}
    spec = spec or {}
    if "audio" in spec:
        from picochat.model.audio import (
            AudioConfig,
            AudioEncoder,
            load_pretrained_audio_encoder,
        )

        a = spec["audio"] or {}
        ms = a.get("ms_per_token", 100)
        if a.get("pretrained"):
            audio_encoder = load_pretrained_audio_encoder(
                d_model, repo_id=a["pretrained"], ms_per_token=ms
            )
            mm_config["audio"] = {
                "kind": "whisper",
                "config": asdict(audio_encoder.cfg),
                "ms_per_token": ms,
            }
        else:
            cfg = AudioConfig(ms_per_token=ms)
            audio_encoder = AudioEncoder(cfg, d_model)
            mm_config["audio"] = {"kind": "scratch", "config": asdict(cfg)}
    if "vision" in spec:
        from picochat.model.vision import (
            VisionEncoder,
            VisionEncoderConfig,
            load_pretrained_vision_encoder,
        )

        v = spec["vision"] or {}
        if v.get("pretrained"):
            vision_encoder = load_pretrained_vision_encoder(
                d_model, repo_id=v["pretrained"]
            )
        else:
            vision_encoder = VisionEncoder(VisionEncoderConfig(), d_model)
        mm_config["vision"] = {"config": asdict(vision_encoder.cfg)}
    return audio_encoder, vision_encoder, mm_config or None


def rebuild_encoders(mm_config: dict | None, d_model: int):
    """Reconstruct (randomly initialized) encoder architectures from a saved
    mm_config -- the inference-side inverse of build_encoders; the caller
    loads the trained weights from the checkpoint on top."""
    audio_encoder = vision_encoder = None
    if not mm_config:
        return None, None
    if "audio" in mm_config:
        a = mm_config["audio"]
        if a["kind"] == "whisper":
            from picochat.model.audio import WhisperAudioEncoder, WhisperEncoderConfig

            audio_encoder = WhisperAudioEncoder(
                WhisperEncoderConfig(**a["config"]), d_model, a["ms_per_token"]
            )
        else:
            from picochat.model.audio import AudioConfig, AudioEncoder

            audio_encoder = AudioEncoder(AudioConfig(**a["config"]), d_model)
    if "vision" in mm_config:
        from picochat.model.vision import VisionEncoder, VisionEncoderConfig

        vision_encoder = VisionEncoder(
            VisionEncoderConfig(**mm_config["vision"]["config"]), d_model
        )
    return audio_encoder, vision_encoder


def scatter_embeds(
    text_embeds: Tensor, input_ids: Tensor, features: Tensor, token_id: int
) -> Tensor:
    """Replace the embeddings at `token_id` placeholder positions with soft
    tokens. `text_embeds` (B, L, d) is the token embedding of `input_ids`;
    `features` (n, d) is every media item's soft tokens concatenated in
    placeholder order (row-major over the batch -- the order render_parts /
    the collate emit). The placeholder count must equal n."""
    mask = input_ids == token_id
    n_slots = int(mask.sum())
    if n_slots != features.shape[0]:
        raise ValueError(f"{n_slots} placeholders but {features.shape[0]} soft tokens")
    out = text_embeds.clone()
    out[mask] = features.to(out.dtype)
    return out


def splice_media_embeds(
    text_embeds: Tensor,
    input_ids: Tensor,
    tokenizer: Tokenizer,
    audio_encoder=None,
    mels: list[Tensor] | None = None,
    vision_encoder=None,
    images: list[Tensor] | None = None,
) -> Tensor:
    """Run the encoders over the batch's media (in placeholder order) and
    scatter their soft tokens into the token embeddings. Clips vary in length,
    so the audio encoder runs per clip; images share a fixed size and run as
    one batch. Shared by SFT training and the inference server."""
    embeds = text_embeds
    if mels:
        soft = torch.cat(
            [audio_encoder(mel[None].to(text_embeds.device))[0] for mel in mels]
        )
        embeds = scatter_embeds(
            embeds, input_ids, soft, tokenizer.encode_single_token(AUDIO_TOKEN)
        )
    if images:
        pixels = torch.stack(images).to(text_embeds.device)
        soft = vision_encoder(pixels).reshape(-1, text_embeds.shape[-1])
        embeds = scatter_embeds(
            embeds, input_ids, soft, tokenizer.encode_single_token(IMAGE_TOKEN)
        )
    return embeds
