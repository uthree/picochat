import math

import pytest
import torch

from picochat.gpt import Transformer, TransformerLM, estimate_num_params


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
def test_transformer_output_shape():
    model = Transformer(d_model=32, n_heads=4, n_layers=3)
    x = torch.randn(2, 7, 32)
    out = model(x)
    assert out.shape == x.shape


def test_transformer_cache_per_layer():
    n_layers = 3
    model = Transformer(d_model=32, n_heads=4, n_layers=n_layers)
    out, cache, pos = model.decode(torch.randn(2, 7, 32))
    assert len(cache) == n_layers
    assert all(c is not None for c in cache)


def test_transformer_decode_pos_advances():
    model = Transformer(d_model=32, n_heads=4, n_layers=2)
    _, cache, pos = model.decode(torch.randn(2, 5, 32))
    assert pos == 5
    _, cache, pos = model.decode(torch.randn(2, 3, 32), cache, pos)
    assert pos == 8


def test_transformer_incremental_matches_full():
    torch.manual_seed(0)
    model = Transformer(d_model=32, n_heads=4, n_layers=2).eval()
    x = torch.randn(1, 5, 32)
    full = model(x)

    out, cache, pos = model.decode(x[:, :4])
    step, _, _ = model.decode(x[:, 4:5], cache, pos)
    assert torch.allclose(full[:, 4:5], step, atol=1e-4)


def test_transformer_grouped_query():
    model = Transformer(d_model=32, n_heads=8, n_layers=2, n_kv_heads=2)
    out, cache, pos = model.decode(torch.randn(2, 4, 32))
    assert out.shape == (2, 4, 32)
    assert cache[0].shape[2] == 2  # n_kv_heads heads cached


def test_transformer_backward():
    model = Transformer(d_model=32, n_heads=4, n_layers=2)
    x = torch.randn(2, 5, 32, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# Sequence packing: doc_ids confines attention to one document
# ---------------------------------------------------------------------------
def test_transformer_packed_forward_matches_separate_documents():
    # two documents packed into one sequence with doc_ids must produce exactly
    # what each document produces on its own: the mask blocks cross-document
    # attention, and RoPE is relative so the second document's offset position
    # doesn't matter. Mix windowed and full-attention layers.
    torch.manual_seed(0)
    model = Transformer(
        d_model=32, n_heads=4, n_layers=4, window_size=3, layers_per_block=2
    ).eval()
    a, b = torch.randn(1, 4, 32), torch.randn(1, 6, 32)
    packed = torch.cat([a, b], dim=1)
    doc_ids = torch.tensor([[0] * 4 + [1] * 6])

    out = model(packed, doc_ids)
    assert torch.allclose(out[:, :4], model(a), atol=1e-4)
    assert torch.allclose(out[:, 4:], model(b), atol=1e-4)


def test_transformer_doc_ids_blocks_cross_document_attention():
    torch.manual_seed(0)
    model = Transformer(d_model=32, n_heads=4, n_layers=2).eval()
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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flex_attention path is CUDA-only"
)
def test_transformer_packed_forward_matches_separate_documents_on_cuda():
    # on CUDA the packed mask is a flex_attention BlockMask instead of a dense
    # SDPA mask; it must agree with per-document forwards just like on CPU
    torch.manual_seed(0)
    model = (
        Transformer(
            d_model=64, n_heads=8, n_layers=4, window_size=3, layers_per_block=2
        )
        .cuda()
        .eval()
    )
    a = torch.randn(1, 4, 64, device="cuda")
    b = torch.randn(1, 6, 64, device="cuda")
    packed = torch.cat([a, b], dim=1)
    doc_ids = torch.tensor([[0] * 4 + [1] * 6], device="cuda")

    out = model(packed, doc_ids)
    assert torch.allclose(out[:, :4], model(a), atol=1e-3)
    assert torch.allclose(out[:, 4:], model(b), atol=1e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch.compile + flex_attention is CUDA-only"
)
def test_transformer_forward_compiles_fullgraph_cuda():
    # Guards the compiled forward path: with MoE + windowed attention + document
    # packing + multiple blocks, the forward must compile *without a graph break*
    # (fullgraph=True) and run. This pins the block-boundary / MLP-tail /
    # finalize refactors -- extracting them into helper methods must not split
    # the fused graph or error under inductor. (Eager numerical correctness is
    # covered by the packed-forward tests; a pure extraction leaves it
    # bit-identical.)
    torch.manual_seed(0)
    model = (
        Transformer(
            d_model=64,
            n_heads=4,  # d_head = 16 (flex_attention's minimum under compile)
            n_kv_heads=2,
            n_layers=4,
            layers_per_block=2,
            window_size=3,
            n_experts=8,
            n_active=2,
            d_expert=32,
        )
        .cuda()
        .eval()
    )
    x = torch.randn(1, 12, 64, device="cuda")
    doc_ids = torch.tensor([[0] * 6 + [1] * 6], device="cuda")
    masks = model.packed_masks(doc_ids)
    with torch.no_grad():
        out = torch.compile(model, fullgraph=True)(x, masks=masks)
    assert out.shape == (1, 12, 64)
    assert torch.isfinite(out).all()


def _dense_packed_masks(doc_ids, window_sizes):
    # Ground-truth dense causal+document+window masks (mirrors
    # _packed_attention_mask's CPU/dense branch), built on CPU then moved to the
    # doc_ids device. Passing these to the eager forward forces every layer down
    # the masked-SDPA path instead of flex_attention.
    dc = doc_ids.cpu()
    idx = torch.arange(dc.shape[-1])
    masks = {}
    for ws in window_sizes:
        m = (idx[:, None] >= idx[None, :]) & (dc[:, :, None] == dc[:, None, :])
        if ws is not None:
            m &= idx[None, :] > idx[:, None] - ws
        masks[ws] = m[:, None].to(doc_ids.device)  # (b, 1, l, l)
    return masks


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch.compile + flex_attention is CUDA-only"
)
def test_compiled_flex_forward_matches_reference_sdpa_cuda():
    # The compiled forward runs windowed/global attention through flex_attention
    # (a fused Triton kernel); this checks it computes the SAME thing as a
    # reference masked-SDPA forward. Reference = eager forward fed DENSE masks
    # (the SDPA path); candidate = torch.compiled forward fed the flex BlockMasks
    # from packed_masks(). Dense FFN only (no MoE) so the only nontrivial kernel
    # under comparison is the attention itself.
    #
    # NOTE: the sequence length must be a multiple of flex_attention's 128 block
    # size. Below that, the compiled flex kernel's partial-block handling diverges
    # from dense SDPA (~0.5 abs), an edge case real training (seq 1k-4k) never
    # hits; at block-aligned lengths the two agree to ~1e-6. Do not shrink seq
    # here to speed the test up -- it would fail spuriously, not catch a bug.
    torch.manual_seed(0)
    seq = 128
    model = (
        Transformer(
            d_model=128,
            n_heads=4,  # d_head = 32
            n_kv_heads=2,  # exercises GQA in both mask paths
            n_layers=4,
            layers_per_block=2,  # mixes windowed and global (last-of-block) layers
            window_size=64,
        )
        .cuda()
        .eval()
    )
    x = torch.randn(1, seq, 128, device="cuda")
    # Two documents packed into the one sequence, so document masking is exercised.
    doc_ids = torch.tensor([[0] * 50 + [1] * (seq - 50)], device="cuda")

    window_sizes = {layer.attn.window_size for layer in model.layers}
    with torch.no_grad():
        reference = model(x, masks=_dense_packed_masks(doc_ids, window_sizes))
        flex_masks = model.packed_masks(doc_ids)  # BlockMasks on CUDA
        compiled = torch.compile(model, fullgraph=True)(x, masks=flex_masks)
    assert torch.allclose(reference, compiled, atol=1e-4, rtol=1e-4), (
        (reference - compiled).abs().max()
    )


def test_transformer_packed_backward_with_grad_checkpoint():
    model = Transformer(d_model=32, n_heads=4, n_layers=2, grad_checkpoint=True)
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
    lm = TransformerLM(
        vocab_size=vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=2,
    )
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
# residual-path output projections (attention proj_o, FFN proj_down) are scaled
# down by 1/sqrt(2*n_layers) so the residual variance stays constant with depth.
# ---------------------------------------------------------------------------
def test_init_non_residual_weights_use_init_std():
    torch.manual_seed(0)
    init_std = 0.02
    lm = TransformerLM(
        vocab_size=2000, d_model=64, n_heads=8, n_layers=6, init_std=init_std
    )
    # the embedding and the q projection are not residual outputs -> plain init_std
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
    # proj_o / proj_down write into the residual stream -> scaled-down std,
    # identically at every depth (the scale is 1/sqrt(2*n_layers), not per-layer).
    proj_o_0 = lm.transformer.layers[0].attn.proj_o.weight
    proj_o_5 = lm.transformer.layers[5].attn.proj_o.weight
    proj_down_0 = lm.transformer.layers[0].ffn.proj_down.weight
    for w in (proj_o_0, proj_o_5, proj_down_0):
        assert w.std().item() == pytest.approx(scaled, rel=0.15)


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
        dict(vocab_size=50, d_model=32, n_heads=4, n_layers=3),  # dense
        dict(
            vocab_size=50, d_model=64, n_heads=8, n_layers=3, n_experts=4, d_expert=16
        ),  # MoE
        dict(vocab_size=50, d_model=48, n_heads=6, n_layers=2, n_kv_heads=2),  # GQA
    ],
)
def test_estimate_num_params_matches_actual(cfg):
    lm = TransformerLM(**cfg)
    assert estimate_num_params(**cfg) == _actual_params(lm)
