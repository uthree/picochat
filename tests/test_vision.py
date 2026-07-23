"""Verify the vision input path: SigLIP-style preprocessing, the ported vision
tower with pixel-shuffle token merging, the tower/projector parameter split for
stage-1 freezing, and the HF checkpoint key mapping against a fabricated
safetensors file (no network, random weights)."""

import pytest
import torch
from safetensors.torch import save_file

from picochat.model.vision import (
    VisionEncoder,
    VisionEncoderConfig,
    hf_vision_key_map,
    load_hf_vision_state,
    preprocess_image,
)

# tiny config: 32px / patch8 -> 4x4 grid, shuffle 2 -> 4 soft tokens
TINY = VisionEncoderConfig(
    image_size=32, patch_size=8, d_encoder=16, n_layers=2, n_heads=2, d_ffn=32
)
D_MODEL = 24


def test_tokens_per_image_grid_arithmetic():
    assert TINY.grid_size == 4
    assert TINY.tokens_per_image == 4  # (32 / 8 / 2)^2
    # the real SigLIP2-base config: 16x16 patches -> 64 soft tokens
    assert VisionEncoderConfig().tokens_per_image == 64


def test_encoder_output_shape():
    torch.manual_seed(0)
    enc = VisionEncoder(TINY, d_model=D_MODEL)
    out = enc(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, TINY.tokens_per_image, D_MODEL)
    assert torch.isfinite(out).all()


def test_preprocess_from_tensor():
    img = torch.randint(0, 256, (3, 50, 70), dtype=torch.uint8)
    x = preprocess_image(img, TINY)
    assert x.shape == (3, 32, 32)
    # [0, 255] -> [0, 1] -> normalize with mean/std 0.5 lands in [-1, 1]
    assert x.min() >= -1.0 and x.max() <= 1.0
    # float input already in [0, 1] must not be rescaled again
    y = preprocess_image(torch.ones(3, 32, 32), TINY)
    assert torch.allclose(y, torch.ones(3, 32, 32))


def test_preprocess_from_pil():
    Image = pytest.importorskip("PIL.Image")
    import numpy as np

    arr = np.random.default_rng(0).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    x = preprocess_image(Image.fromarray(arr), TINY)
    assert x.shape == (3, 32, 32)
    assert x.min() >= -1.0 and x.max() <= 1.0
    assert x.dtype == torch.float32


def test_preprocess_rejects_bad_tensor_shape():
    with pytest.raises(ValueError):
        preprocess_image(torch.zeros(1, 32, 32), TINY)


def test_gradient_flows_to_patch_embedding():
    torch.manual_seed(0)
    enc = VisionEncoder(TINY, d_model=D_MODEL)
    enc(torch.randn(1, 3, 32, 32)).sum().backward()
    assert enc.patch_embed.weight.grad is not None
    assert enc.patch_embed.weight.grad.abs().sum() > 0


def test_tower_and_projector_partition_parameters():
    # the stage-1 freezing contract: tower_parameters() and projector cover
    # every parameter exactly once
    enc = VisionEncoder(TINY, d_model=D_MODEL)
    tower = set(map(id, enc.tower_parameters()))
    projector = set(map(id, enc.projector.parameters()))
    everything = set(map(id, enc.parameters()))
    assert tower | projector == everything
    assert tower & projector == set()
    assert len(tower) > 0 and len(projector) > 0


def _fake_hf_state(cfg: VisionEncoderConfig) -> dict[str, torch.Tensor]:
    """A fabricated SiglipVisionModel state dict with HF keys and the right
    shapes for `cfg`, built from a randomly-initialized VisionEncoder via the
    shared key map (the shapes are the module's own, so shape drift in the
    module also fails here)."""
    donor = VisionEncoder(cfg, d_model=D_MODEL)
    ours = donor.state_dict()
    state = {hf: torch.randn_like(ours[k]) for hf, k in hf_vision_key_map(cfg).items()}
    # contrastive-only extras a real checkpoint carries; the loader must skip
    state["vision_model.head.probe"] = torch.randn(1, 1, cfg.d_encoder)
    state["logit_scale"] = torch.zeros(())
    return state


def test_loader_maps_fabricated_safetensors(tmp_path):
    from safetensors.torch import load_file

    torch.manual_seed(0)
    state = _fake_hf_state(TINY)
    path = tmp_path / "model.safetensors"
    save_file(state, str(path))

    enc = VisionEncoder(TINY, d_model=D_MODEL)
    before = {k: v.clone() for k, v in enc.projector.state_dict().items()}
    load_hf_vision_state(enc, load_file(str(path)))

    # every tower weight is the checkpoint's, verified through the key map
    loaded = enc.state_dict()
    for hf_key, our_key in hf_vision_key_map(TINY).items():
        assert torch.equal(loaded[our_key], state[hf_key]), our_key
    # the projector kept its fresh init (it is never in the checkpoint)
    for k, v in enc.projector.state_dict().items():
        assert torch.equal(v, before[k])


def test_loader_fails_loudly_on_missing_tower_key():
    state = _fake_hf_state(TINY)
    del state["vision_model.post_layernorm.weight"]
    with pytest.raises(KeyError, match="missing expected tower keys"):
        load_hf_vision_state(VisionEncoder(TINY, d_model=D_MODEL), state)


def test_key_map_covers_all_tower_parameters():
    # the mapping's targets must be exactly the non-projector parameters, so a
    # new tower submodule cannot be silently left at random init
    enc = VisionEncoder(TINY, d_model=D_MODEL)
    mapped = set(hf_vision_key_map(TINY).values())
    tower_keys = {k for k in enc.state_dict() if not k.startswith("projector.")}
    assert mapped == tower_keys


def test_image_special_tokens_exist():
    # the wire format vision.py's docstring promises
    from picochat.tokenizer import IMAGE_TOKEN, VISION_END_TOKEN, VISION_START_TOKEN

    assert VISION_START_TOKEN == "<|vision_start|>"
    assert IMAGE_TOKEN == "<|image_pad|>"
    assert VISION_END_TOKEN == "<|vision_end|>"


def test_default_config_is_siglip2_base():
    cfg = VisionEncoderConfig()
    assert (cfg.image_size, cfg.patch_size) == (256, 16)
    assert (cfg.d_encoder, cfg.n_layers, cfg.n_heads, cfg.d_ffn) == (768, 12, 12, 3072)


def test_encoder_preprocess_and_token_count_are_consistent():
    # the rendering-layer contract: enc.preprocess sizes pixels for enc, and
    # enc.tokens_per_image predicts exactly how many soft tokens come out
    torch.manual_seed(0)
    enc = VisionEncoder(TINY, d_model=D_MODEL)
    pixels = enc.preprocess(torch.randint(0, 256, (3, 47, 91), dtype=torch.uint8))
    assert pixels.shape == (3, TINY.image_size, TINY.image_size)
    assert isinstance(enc.tokens_per_image, int)
    assert enc(pixels[None]).shape[1] == enc.tokens_per_image
