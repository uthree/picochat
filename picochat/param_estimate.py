"""Parameter-count estimation for the picochat model, without building it.

Pure arithmetic that mirrors the module shapes in gpt.py / linear_attn.py /
sparse_attn.py, so the scale ladder can be sized on a machine that can't hold the
larger presets in memory. Kept separate from the module definitions (which touch
torch) precisely because it touches no torch modules -- but its correctness
*depends on* those shapes, so the two must move together
(tests/test_presets.py asserts they agree exactly).
"""

from __future__ import annotations


def _gdn_params(d_model: int, n_heads: int, kv_dim: int, conv_size: int) -> int:
    """Gated DeltaNet mixer (linear_attn.GatedDeltaNet), one layer. d_head =
    d_model // n_heads; value_dim = n_heads * d_head = d_model."""
    d_head = d_model // n_heads
    value_dim = d_model
    conv_dim = 2 * kv_dim + value_dim
    return (
        d_model * kv_dim  # proj_q
        + d_model * kv_dim  # proj_k
        + d_model * value_dim  # proj_v
        + d_model * value_dim  # proj_z (output gate)
        + d_model * n_heads  # a_proj
        + d_model * n_heads  # b_proj
        + 2 * n_heads  # dt_bias, A_log
        + conv_dim * conv_size  # depthwise short conv
        + d_head  # GatedRMSNorm weight
        + value_dim * d_model  # proj_o
    )


def _nsa_params(d_model: int, n_heads: int, nsa_kv_heads: int) -> int:
    """Native Sparse Attention mixer (sparse_attn.NativeSparseAttention), one
    layer. One shared K/V for all three branches (the compressed branch derives
    its keys/values by mean pooling -- no learned compression parameters);
    partial RoPE adds no parameters (buffers only)."""
    d_head = d_model // n_heads
    kv_dim = d_head * nsa_kv_heads
    return (
        d_model * d_model  # proj_q
        + 2 * d_model * kv_dim  # shared k/v
        + d_model * d_model  # proj_o
        + d_model * (n_heads * 3)  # branch gate
    )


def estimate_num_params(
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    n_kv_heads: int | None = None,
    nsa_kv_heads: int = 1,
    layers_per_block: int = 4,
    d_ffn: int | None = None,
    conv_size: int = 4,
    n_experts: int | None = None,
    d_expert: int | None = None,
    n_active: int = 2,
    d_latent: int | None = None,
    share_experts: bool = False,
    n_mtp: int = 0,
    mtp_rank: int | None = None,
    active_only: bool = False,
    **_ignored,
) -> int:
    """Estimate a TransformerLM's parameter count from its hyperparameters,
    without building the model (which can OOM at large scale).

    active_only=False (default) counts every parameter (the total). active_only=
    True counts only the parameters a single token's forward pass touches -- the
    "active parameters" headline for a sparse model: the router still runs fully
    but only n_active of the n_experts experts fire per token. The GDN/NSA mixers
    are dense (always active); embedding and lm head are counted in full in both.
    (For a dense model the two are equal.)

    Mirrors the module shapes. RMSNorm adds no parameters, RoPE/compression
    position tables are buffers or tiny, and MoE buffers (expert_bias) are
    ignored, so this is the trainable parameter count -- exact for the current
    architecture, but treat it as an estimate since shapes may drift. Extra
    keyword arguments (max_seq_len, window_size, rope_base, sel_block, ...) are
    accepted and ignored, so a preset or a saved model_config can be splatted in:

        estimate_num_params(**MODEL_PRESETS["8b"])
        estimate_num_params(**MODEL_PRESETS["35b-moe"], active_only=True)
    """
    n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
    d_head = d_model // n_heads
    kv_dim = d_head * n_kv_heads
    ffn_hidden = 3 * d_model if d_ffn is None else d_ffn
    # TransformerLayer falls d_expert back to d_ffn, and MoE falls a None hidden
    # back to 3*d_model -- so an unset d_expert lands on ffn_hidden.
    expert_hidden = ffn_hidden if d_expert is None else d_expert

    # Layer split: the block-tail layers ((i+1) % layers_per_block == 0) are NSA
    # (global) mixers; the rest are Gated DeltaNet (linear) mixers.
    nsa_layers = n_layers // layers_per_block
    gdn_layers = n_layers - nsa_layers
    gdn = _gdn_params(d_model, n_heads, kv_dim, conv_size)
    nsa = _nsa_params(d_model, n_heads, nsa_kv_heads)
    mixers = gdn_layers * gdn + nsa_layers * nsa

    # Per-layer, mixer-independent: SwiGLU FFN + two Block-AttnRes depth queries.
    ffn = 3 * d_model * ffn_hidden
    per_layer_common = ffn + 2 * d_model
    shared_bank = 0
    if n_experts is not None:
        # router (always fully active) + per-expert up/gate/down; only n_active
        # of the experts fire for a given token, so the active count uses those.
        # With d_latent (LatentMoE) the experts' io dimension shrinks to the
        # latent size and the shared compress/expand pair (always active) is
        # added on top. With share_experts the up/gate/down weights exist once
        # for the whole stack (ExpertBank) instead of per layer.
        expert_io = d_model if d_latent is None else d_latent
        expert_size = 3 * expert_hidden * expert_io  # one expert's up/gate/down
        per_layer_common += n_experts * d_model  # router
        per_layer_common += d_model  # out_gain: expert-output RMSNorm (per layer)
        if d_latent is not None:
            per_layer_common += 2 * d_model * d_latent
        if share_experts:
            experts = min(n_layers * n_active, n_experts) if active_only else n_experts
            shared_bank = experts * expert_size
        else:
            experts = n_active if active_only else n_experts
            per_layer_common += experts * expert_size

    embed = vocab_size * d_model
    lmhead = vocab_size * d_model  # separate (untied) output projection
    # Multi-token-prediction heads (MTPHead): each is a d_model-space residual
    # transform (d_model^2, or 2*d_model*mtp_rank when low-rank) decoded by the
    # shared lm head. Counted in the total but NOT in active_only: plain
    # autoregressive decoding runs only the primary head; the MTP heads fire only
    # in the optional speculative-decoding path (see TransformerLM.decode_heads).
    mtp_per_head = d_model * d_model if mtp_rank is None else 2 * d_model * mtp_rank
    mtp = 0 if active_only else n_mtp * mtp_per_head
    layers = n_layers * per_layer_common + mixers
    mix_out = d_model  # final depth-attention query before the head
    return embed + lmhead + layers + mix_out + shared_bank + mtp
