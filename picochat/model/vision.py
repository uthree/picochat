"""Image input (multimodal): a SigLIP2 vision tower, pixel-shuffle token
merging, and a projector into the LM's embedding space.

The integration mirrors audio.py and the current de-facto VLM recipe
(LLaVA / SmolVLM / Qwen2-VL): a pretrained encoder turns an image into a short
sequence of feature vectors, a projector maps them to `d_model`, and those
become *soft tokens* placed at IMAGE_TOKEN placeholder positions in an
otherwise ordinary ChatML sequence (spliced via the same mechanism as
audio.scatter_audio_embeds). The wire format is `<|vision_start|>` +
`<|image_pad|>` * n + `<|vision_end|>` with n = `tokens_per_image`.

Unlike the from-scratch audio encoder, the vision tower is an in-repo port of
the SigLIP/SigLIP2 fixed-resolution ViT (google/siglip2-base-patch16-256):
image understanding leans hard on pretraining scale, so we load Google's
released weights instead of training our own -- but through our own module,
keeping picochat free of a `transformers` dependency. The port covers exactly
the pieces a VLM uses: patch embedding, learned position embeddings, the
pre-LN transformer stack and the final LayerNorm. The checkpoint's
attention-pooling head is contrastive-training equipment and is skipped -- VLMs
consume the full patch grid, not a pooled summary.

Token budget: 256px / patch16 gives a 16x16 = 256-patch grid, too long for a
small model's context. Pixel shuffle (SmolVLM / InternVL style) folds each
2x2 patch neighborhood into one token by concatenating on the channel dim:
64 tokens of dim 4*768, which the projector then maps to `d_model`. This is
lossless spatially (unlike pooling) and is the current standard reduction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# SigLIP normalizes to [-1, 1]: (x/255 - 0.5) / 0.5. Same value per channel.
SIGLIP_MEAN = 0.5
SIGLIP_STD = 0.5

# SigLIP's LayerNorm epsilon (HF `layer_norm_eps`); matching it matters for
# numerical parity with the released checkpoint.
_LN_EPS = 1e-6


@dataclass
class VisionEncoderConfig:
    """The SigLIP2-base-patch16-256 vision tower (defaults) + token merging."""

    image_size: int = 256
    patch_size: int = 16
    d_encoder: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ffn: int = 3072
    pixel_shuffle: int = 2  # fold pixel_shuffle^2 patches into one soft token

    @property
    def grid_size(self) -> int:
        """Patches per image side (16 for 256px / patch16)."""
        return self.image_size // self.patch_size

    @property
    def tokens_per_image(self) -> int:
        """Soft tokens the LM sees per image, after pixel-shuffle merging."""
        return (self.grid_size // self.pixel_shuffle) ** 2


def preprocess_image(
    img, cfg: VisionEncoderConfig | None = None, image_size: int | None = None
) -> Tensor:
    """PIL.Image or (3, H, W) uint8/float tensor -> normalized (3, S, S) float
    pixels, matching HF's SiglipImageProcessor: resize to a fixed square
    (bilinear), rescale to [0, 1], normalize with mean/std 0.5.

    PIL inputs are resized with PIL itself (exactly what the HF processor
    does -- PIL's uint8 rounding makes any other resizer disagree at the
    1/255 level); tensor inputs use antialiased bilinear interpolation, the
    float-domain equivalent."""
    size = (
        image_size
        if image_size is not None
        else (cfg or VisionEncoderConfig()).image_size
    )
    if isinstance(img, Tensor):
        if img.dim() != 3 or img.shape[0] != 3:
            raise ValueError(f"expected a (3, H, W) tensor, got {tuple(img.shape)}")
        x = img.float()
        if img.dtype == torch.uint8:
            x = x / 255.0
        if x.shape[-2:] != (size, size):
            x = F.interpolate(
                x[None], size=(size, size), mode="bilinear", antialias=True
            )[0]
    else:  # PIL.Image (duck-typed so PIL stays an optional import)
        from PIL import Image

        img = img.convert("RGB").resize((size, size), Image.BILINEAR)
        x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
        x = x.reshape(size, size, 3).permute(2, 0, 1).float() / 255.0
    return (x - SIGLIP_MEAN) / SIGLIP_STD


class _Attention(nn.Module):
    """Standard multi-head self-attention with separate q/k/v/out projections
    (all biased), matching the SigLIP checkpoint layout key-for-key so loading
    needs no weight surgery."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d_encoder {d} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        shape = (b, n, self.n_heads, d // self.n_heads)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        # bidirectional -- every patch sees every patch, no causal mask
        o = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(o.transpose(1, 2).reshape(b, n, d))


class _EncoderLayer(nn.Module):
    """Pre-LN transformer block: LN -> MHA -> residual, LN -> MLP -> residual.
    The MLP uses the tanh-approximated GELU -- SigLIP's `gelu_pytorch_tanh`;
    exact GELU drifts from the checkpoint at the 1e-3 level."""

    def __init__(self, cfg: VisionEncoderConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_encoder, eps=_LN_EPS)
        self.attn = _Attention(cfg.d_encoder, cfg.n_heads)
        self.norm2 = nn.LayerNorm(cfg.d_encoder, eps=_LN_EPS)
        self.fc1 = nn.Linear(cfg.d_encoder, cfg.d_ffn)
        self.fc2 = nn.Linear(cfg.d_ffn, cfg.d_encoder)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x)), approximate="tanh"))
        return x


class VisionEncoder(nn.Module):
    """SigLIP2 fixed-resolution vision tower + pixel shuffle + projector.
    Input (B, 3, S, S) preprocessed pixels -> (B, tokens_per_image, d_model)
    soft tokens for the LM.

    No CLS token: SigLIP embeds every patch symmetrically (position embeddings
    only) and lets a pooling head summarize for contrastive use; we drop that
    head and hand the LM the whole grid. The tower runs in float32 regardless
    of any surrounding autocast -- ViT activations through 12 pre-LN blocks
    accumulate enough error in half precision to visibly perturb the soft
    tokens (same policy as blocks.rms_norm).

    The pretrained tower and the fresh adapter stay cleanly separated for
    stage-1 freezing: everything the checkpoint provides lives outside
    `self.projector`, and `tower_parameters()` iterates exactly that part."""

    def __init__(self, cfg: VisionEncoderConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_encoder
        # patch embedding: a stride=patch conv, i.e. a linear map per
        # non-overlapping patch (bias included, as in the checkpoint)
        self.patch_embed = nn.Conv2d(
            3, d, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        # learned absolute position per patch -- fixed-resolution SigLIP has
        # exactly grid_size^2 positions (no interpolation machinery needed)
        self.pos_embed = nn.Embedding(cfg.grid_size**2, d)
        self.layers = nn.ModuleList(_EncoderLayer(cfg) for _ in range(cfg.n_layers))
        self.post_norm = nn.LayerNorm(d, eps=_LN_EPS)
        # freshly-initialized adapter (never in the checkpoint): pixel-shuffled
        # patch stacks -> LM embedding space, 2-layer MLP as in LLaVA-1.5
        d_merged = d * cfg.pixel_shuffle**2
        self.projector = nn.Sequential(
            nn.Linear(d_merged, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    @property
    def tokens_per_image(self) -> int:
        """Soft tokens per image -- how many IMAGE_TOKEN placeholders the
        rendering layer must emit per image so the scatter counts line up."""
        return self.cfg.tokens_per_image

    def preprocess(self, img) -> Tensor:
        """preprocess_image bound to this encoder's config, so callers hold a
        single object that both sizes the pixels and consumes them."""
        return preprocess_image(img, self.cfg)

    def tower_parameters(self):
        """Every parameter that came from the pretrained checkpoint (the ViT
        tower), i.e. all parameters except the projector's -- the set to freeze
        during stage-1 (projector-only) training."""
        projector = set(map(id, self.projector.parameters()))
        return (p for p in self.parameters() if id(p) not in projector)

    def _pixel_shuffle(self, x: Tensor) -> Tensor:
        """(B, grid^2, d) -> (B, (grid/r)^2, d*r^2): each r x r neighborhood of
        the patch grid concatenates on the channel dim. Spatially lossless --
        the projector sees every patch, just r^2 at a time."""
        r = self.cfg.pixel_shuffle
        b, n, d = x.shape
        g = self.cfg.grid_size
        x = x.view(b, g // r, r, g // r, r, d)  # rows split into (coarse, fine)
        x = x.permute(0, 1, 3, 2, 4, 5)  # group the r x r neighborhood last
        return x.reshape(b, (g // r) ** 2, d * r * r)

    def forward(self, pixels: Tensor) -> Tensor:
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = self.patch_embed(pixels.float())  # (B, d, g, g)
            x = x.flatten(2).transpose(1, 2)  # (B, g*g, d), row-major grid
            x = x + self.pos_embed.weight  # one embedding per grid position
            for layer in self.layers:
                x = layer(x)
            x = self.post_norm(x)
            x = self._pixel_shuffle(x)
            return self.projector(x)  # (B, tokens_per_image, d_model)


# -- pretrained-weight loading ------------------------------------------------


def hf_vision_key_map(cfg: VisionEncoderConfig) -> dict[str, str]:
    """HF SiglipVisionModel state-dict key -> VisionEncoder key, for every
    tower parameter. Shared by the loader and the tests so the mapping under
    test is the mapping in production."""
    m = {
        "vision_model.embeddings.patch_embedding.weight": "patch_embed.weight",
        "vision_model.embeddings.patch_embedding.bias": "patch_embed.bias",
        "vision_model.embeddings.position_embedding.weight": "pos_embed.weight",
        "vision_model.post_layernorm.weight": "post_norm.weight",
        "vision_model.post_layernorm.bias": "post_norm.bias",
    }
    for i in range(cfg.n_layers):
        hf, ours = f"vision_model.encoder.layers.{i}", f"layers.{i}"
        pairs = {
            "layer_norm1": "norm1",
            "self_attn.q_proj": "attn.q_proj",
            "self_attn.k_proj": "attn.k_proj",
            "self_attn.v_proj": "attn.v_proj",
            "self_attn.out_proj": "attn.out_proj",
            "layer_norm2": "norm2",
            "mlp.fc1": "fc1",
            "mlp.fc2": "fc2",
        }
        for hf_mod, our_mod in pairs.items():
            for suffix in ("weight", "bias"):
                m[f"{hf}.{hf_mod}.{suffix}"] = f"{ours}.{our_mod}.{suffix}"
    return m


def load_hf_vision_state(encoder: VisionEncoder, state: dict[str, Tensor]) -> None:
    """Load a HF SiglipModel/SiglipVisionModel state dict into `encoder`'s
    tower. Ignores non-vision weights (text tower, logit scale) and the
    attention-pooling head (`vision_model.head.*` -- contrastive-only); fails
    loudly if any expected tower key is absent, so checkpoint-format drift
    surfaces as an error instead of silently random weights."""
    key_map = hf_vision_key_map(encoder.cfg)
    mapped = {ours: state[hf] for hf, ours in key_map.items() if hf in state}
    missing_hf = [hf for hf in key_map if hf not in state]
    if missing_hf:
        raise KeyError(f"checkpoint is missing expected tower keys: {missing_hf[:5]}")
    # strict=False only to allow the projector to keep its fresh init; anything
    # else missing or unexpected is a bug in the mapping and must fail
    result = encoder.load_state_dict(mapped, strict=False)
    if result.unexpected_keys:
        raise KeyError(
            f"mapped keys not in VisionEncoder: {result.unexpected_keys[:5]}"
        )
    not_projector = [k for k in result.missing_keys if not k.startswith("projector.")]
    if not_projector:
        raise KeyError(f"tower keys not covered by the mapping: {not_projector[:5]}")


def load_pretrained_vision_encoder(
    d_model: int, repo_id: str = "google/siglip2-base-patch16-256"
) -> VisionEncoder:
    """Download and load the SigLIP2 vision tower from the HF Hub. The config
    is read from the checkpoint's `vision_config` (absent fields fall back to
    SigLIP-base values); the projector stays freshly initialized and must be
    trained (stage 1 of the usual VLM recipe)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    with open(hf_hub_download(repo_id, "config.json")) as f:
        vc = json.load(f).get("vision_config", {})
    cfg = VisionEncoderConfig(
        image_size=vc.get("image_size", 256),
        patch_size=vc.get("patch_size", 16),
        d_encoder=vc.get("hidden_size", 768),
        n_layers=vc.get("num_hidden_layers", 12),
        n_heads=vc.get("num_attention_heads", 12),
        d_ffn=vc.get("intermediate_size", 3072),
    )
    encoder = VisionEncoder(cfg, d_model)
    load_hf_vision_state(
        encoder, load_file(hf_hub_download(repo_id, "model.safetensors"))
    )
    return encoder
