import math

import pytest
import torch

from picochat.gpt import Transformer, TransformerLM
from picochat.param_estimate import estimate_num_params


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
def test_transformer_output_shape():
    model = Transformer(d_model=32, n_heads=4, n_layers=3)
    x = torch.randn(2, 7, 32)
    out = model(x)
    assert out.shape == x.shape


def test_transformer_cache_per_layer():
    n_layers = 4  # lpb=4 -> 3 GDN + 1 NSA, so both cache formats appear
    model = Transformer(d_model=32, n_heads=4, n_layers=n_layers)
    out, cache, pos = model.decode(torch.randn(2, 7, 32))
    assert len(cache) == n_layers
    assert all(c is not None for c in cache)
    # GDN layers cache a (recurrent_state, conv_state) tuple; NSA a KV dict
    assert isinstance(cache[0], tuple) and model.layers[0].linear
    assert isinstance(cache[3], dict) and not model.layers[3].linear


def test_transformer_decode_pos_advances():
    model = Transformer(d_model=32, n_heads=4, n_layers=2)
    _, cache, pos = model.decode(torch.randn(2, 5, 32))
    assert pos == 5
    _, cache, pos = model.decode(torch.randn(2, 3, 32), cache, pos)
    assert pos == 8


def test_transformer_incremental_matches_full():
    torch.manual_seed(0)
    model = Transformer(
        d_model=32, n_heads=4, n_layers=4, layers_per_block=2, window_size=8,
        cmp_block=4, cmp_stride=2, sel_block=4, n_selected=4,
    ).eval()
    x = torch.randn(1, 5, 32)
    full = model(x)

    out, cache, pos = model.decode(x[:, :4])
    step, _, _ = model.decode(x[:, 4:5], cache, pos)
    assert torch.allclose(full[:, 4:5], step, atol=1e-4)


def test_transformer_grouped_query():
    model = Transformer(d_model=32, n_heads=8, n_layers=2, n_kv_heads=2)
    out, cache, pos = model.decode(torch.randn(2, 4, 32))
    assert out.shape == (2, 4, 32)


def test_transformer_backward():
    model = Transformer(d_model=32, n_heads=4, n_layers=2)
    x = torch.randn(2, 5, 32, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# Sequence packing: doc_ids confines the mixers to one document (NSA masks
# within a document; GDN resets its recurrent state at document boundaries).
# ---------------------------------------------------------------------------
def test_transformer_packed_first_document_matches_standalone():
    # The first packed document sits at absolute offset 0, so packing must not
    # change it at all: it produces exactly what it does on its own (GDN resets
    # its state / conv at the boundary; NSA masks cross-document attention). A
    # *later* document is isolated (see the next test) but is not required to
    # match its standalone forward bit-for-bit, because NSA's compression branch
    # pools RoPE'd keys and so depends on absolute position, not just the offset.
    torch.manual_seed(0)
    model = Transformer(
        d_model=32, n_heads=4, n_layers=4, layers_per_block=2, window_size=8,
        cmp_block=4, cmp_stride=2, sel_block=4, n_selected=4,
    ).eval()
    a, b = torch.randn(1, 4, 32), torch.randn(1, 6, 32)
    packed = torch.cat([a, b], dim=1)
    doc_ids = torch.tensor([[0] * 4 + [1] * 6])

    out = model(packed, doc_ids)
    assert torch.allclose(out[:, :4], model(a), atol=1e-4)


def test_transformer_doc_ids_blocks_cross_document_attention():
    torch.manual_seed(0)
    model = Transformer(
        d_model=32, n_heads=4, n_layers=4, layers_per_block=2, window_size=8,
        cmp_block=4, cmp_stride=2, sel_block=4, n_selected=4,
    ).eval()
    x = torch.randn(1, 8, 32)
    doc_ids = torch.tensor([[0] * 4 + [1] * 4])
    base = model(x, doc_ids)

    perturbed = x.clone()
    perturbed[:, 0] += 100.0  # inside document 0
    out = model(perturbed, doc_ids)
    # document 1 must not see the change; document 0 must
    assert torch.allclose(base[:, 4:], out[:, 4:], atol=1e-5)
    assert not torch.allclose(base[:, :4], out[:, :4], atol=1e-5)
    # without doc_ids the change leaks into the second half
    assert not torch.allclose(model(x)[:, 4:], model(perturbed)[:, 4:], atol=1e-5)


def test_transformer_packed_backward_with_grad_checkpoint():
    model = Transformer(d_model=32, n_heads=4, n_layers=4, grad_checkpoint=True)
    model.train()
    x = torch.randn(2, 6, 32, requires_grad=True)
    doc_ids = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 1, 1, 2, 2]])
    model(x, doc_ids).sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# TransformerLM
# ---------------------------------------------------------------------------
def test_transformer_lm_logits_shape():
    vocab_size = 40
    lm = TransformerLM(vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2)
    tokens = torch.randint(0, vocab_size, (2, 5))
    logits = lm(tokens)
    assert logits.shape == (2, 5, vocab_size)


def test_transformer_lm_incremental_matches_full():
    vocab_size = 40
    lm = TransformerLM(vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2).eval()
    tokens = torch.randint(0, vocab_size, (1, 5))
    full = lm(tokens)

    _, cache, pos = lm.decode(tokens[:, :4])
    step, _, _ = lm.decode(tokens[:, 4:5], cache, pos)
    assert torch.allclose(full[:, 4:5], step, atol=1e-4)


# ---------------------------------------------------------------------------
# Weight init: GPT-2 style. Every weight ~ normal(0, init_std), then the
# residual-path output projections (mixer proj_o, FFN proj_down) are scaled
# down by 1/sqrt(2*n_layers) so the residual variance stays constant with depth.
# ---------------------------------------------------------------------------
def test_init_non_residual_weights_use_init_std():
    torch.manual_seed(0)
    init_std = 0.02
    lm = TransformerLM(
        vocab_size=2000, d_model=64, n_heads=8, n_layers=6, init_std=init_std
    )
    # the embedding and a GDN q projection are not residual outputs -> init_std
    proj_q_0 = lm.transformer.layers[0].attn.proj_q.weight
    for w in (proj_q_0, lm.embed.weight):
        assert w.std().item() == pytest.approx(init_std, rel=0.15)


def test_init_residual_projections_are_depth_scaled():
    torch.manual_seed(0)
    init_std = 0.02
    n_layers = 6
    lm = TransformerLM(
        vocab_size=2000, d_model=64, n_heads=8, n_layers=n_layers, init_std=init_std
    )
    scaled = init_std / math.sqrt(2 * n_layers)
    # every mixer's proj_o and the FFN proj_down write into the residual stream
    # -> scaled-down std, identically at every depth (scale is 1/sqrt(2*n_layers))
    proj_o_gdn = lm.transformer.layers[0].attn.proj_o.weight  # GDN
    proj_o_nsa = lm.transformer.layers[3].attn.proj_o.weight  # NSA (block tail)
    proj_down_0 = lm.transformer.layers[0].ffn.proj_down.weight
    for w in (proj_o_gdn, proj_o_nsa, proj_down_0):
        assert w.std().item() == pytest.approx(scaled, rel=0.15)


def test_init_gdn_gate_params_reset():
    # GatedDeltaNet.reset_parameters restores the Mamba2 gate init after the
    # generic normal init: A_log ~ log(uniform(1,16)) (so exp(A_log) in [1,16]),
    # dt_bias zeroed, output-norm weight ones.
    lm = TransformerLM(vocab_size=100, d_model=32, n_heads=4, n_layers=4)
    gdn = lm.transformer.layers[0].attn
    assert torch.all(gdn.dt_bias == 0)
    assert torch.allclose(gdn.norm.weight, torch.ones_like(gdn.norm.weight))
    a = gdn.A_log.exp()
    assert torch.all(a >= 1.0) and torch.all(a <= 16.0 + 1e-4)


def test_init_default_std_is_gpt2_canonical():
    lm = TransformerLM(vocab_size=100, d_model=32, n_heads=4, n_layers=2)
    assert lm.init_std == pytest.approx(0.02)


def test_all_linears_are_bias_free():
    lm = TransformerLM(vocab_size=100, d_model=32, n_heads=4, n_layers=2)
    linears = [m for m in lm.modules() if isinstance(m, torch.nn.Linear)]
    assert linears  # sanity: there are Linear layers
    assert all(m.bias is None for m in linears)


# ---------------------------------------------------------------------------
# estimate_num_params
# ---------------------------------------------------------------------------
def _actual_params(lm) -> int:
    return sum(p.numel() for p in lm.parameters())


@pytest.mark.parametrize(
    "cfg",
    [
        dict(vocab_size=50, d_model=32, n_heads=4, n_layers=3),  # all GDN (lpb=4)
        dict(  # GDN + NSA (lpb=2)
            vocab_size=50, d_model=64, n_heads=8, n_layers=4, layers_per_block=2
        ),
        dict(  # MoE + NSA
            vocab_size=50, d_model=64, n_heads=8, n_layers=4, layers_per_block=2,
            n_experts=4, d_expert=16,
        ),
        dict(  # GQA
            vocab_size=50, d_model=48, n_heads=6, n_layers=4, n_kv_heads=2,
            layers_per_block=2,
        ),
        dict(  # LatentMoE + shared experts + MTP
            vocab_size=50, d_model=64, n_heads=8, n_layers=6, layers_per_block=3,
            n_experts=6, d_expert=16, d_latent=32, share_experts=True,
            n_mtp=2, mtp_rank=8,
        ),
    ],
)
def test_estimate_num_params_matches_actual(cfg):
    lm = TransformerLM(**cfg)
    assert estimate_num_params(**cfg) == _actual_params(lm)
