"""Parameter-count estimation for the picochat model, without building it.

Pure arithmetic that mirrors the module shapes in gpt.py, so the scale ladder
can be sized on a machine that can't hold the larger presets in memory. Kept
separate from gpt.py (which defines the modules) precisely because it touches no
torch modules -- but its correctness *depends on* those module shapes, so the
two must move together (tests/test_presets.py asserts they agree exactly).
"""

from __future__ import annotations


def estimate_num_params(
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    n_kv_heads: int | None = None,
    d_ffn: int | None = None,
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
    True counts only the parameters that a single token's forward pass touches --
    the "active parameters" headline for a sparse model: the router still runs
    fully but only n_active of the n_experts experts fire per token. The
    embedding and the lm head are counted in full in both. (For a dense model
    the two are equal.)

    Mirrors the module shapes in gpt.py. RMSNorm/RoPE add no parameters and
    buffers (e.g. MoE expert_bias) are ignored, so this is the trainable
    parameter count -- exact for the current architecture, but treat it as an
    estimate since the shapes may drift. Extra keyword arguments (max_seq_len,
    window_size, rope_base, ...) are accepted and ignored, so a preset or a
    saved model_config can be splatted straight in:

        estimate_num_params(**MODEL_PRESETS["8b"])
        estimate_num_params(**MODEL_PRESETS["8b-moe"], active_only=True)
    """
    n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
    d_head = d_model // n_heads
    kv_dim = d_head * n_kv_heads
    ffn_hidden = 3 * d_model if d_ffn is None else d_ffn
    # TransformerLayer falls d_expert back to d_ffn, and MoE falls a None hidden
    # back to 3*d_model -- so an unset d_expert lands on ffn_hidden.
    expert_hidden = ffn_hidden if d_expert is None else d_expert

    # Attention: q/g/o are square (d_head*n_heads == d_model), k/v project to
    # kv_dim. All bias-free. g is the gated-attention output gate.
    attn = 3 * d_model * d_model + 2 * d_model * kv_dim
    # SwiGLU: up/gate/down, all bias-free.
    ffn = 3 * d_model * ffn_hidden
    # Block AttnRes: one depth-attention query per sublayer (attn + MLP).
    layer_params = attn + ffn + 2 * d_model
    shared_bank = 0
    if n_experts is not None:
        # router (always fully active) + per-expert up/gate/down; only n_active
        # of the experts fire for a given token, so the active count uses those.
        # With d_latent (LatentMoE) the experts' io dimension shrinks to the
        # latent size and the shared compress/expand pair (always active) is
        # added on top. With share_experts the up/gate/down weights exist once
        # for the whole stack (ExpertBank) instead of per layer; a token's
        # forward then touches up to n_layers * n_active distinct experts of
        # that bank, capped by the pool size.
        expert_io = d_model if d_latent is None else d_latent
        expert_size = 3 * expert_hidden * expert_io  # one expert's up/gate/down
        layer_params += n_experts * d_model  # router
        layer_params += d_model  # out_gain: expert-output RMSNorm (per layer)
        if d_latent is not None:
            layer_params += 2 * d_model * d_latent
        if share_experts:
            experts = min(n_layers * n_active, n_experts) if active_only else n_experts
            shared_bank = experts * expert_size
        else:
            experts = n_active if active_only else n_experts
            layer_params += experts * expert_size

    embed = vocab_size * d_model
    lmhead = vocab_size * d_model  # separate (untied) output projection
    # Multi-token-prediction heads (MTPHead): each is a d_model-space residual
    # transform (d_model^2, or 2*d_model*mtp_rank when low-rank) decoded by the
    # shared lm head -- so, unlike a full vocab projection, they are cheap.
    # Counted in the total but NOT in active_only: plain autoregressive decoding
    # runs only the primary head; the MTP heads fire only in the optional
    # speculative-decoding path (see TransformerLM.decode_heads).
    mtp_per_head = d_model * d_model if mtp_rank is None else 2 * d_model * mtp_rank
    mtp = 0 if active_only else n_mtp * mtp_per_head
    layers = n_layers * layer_params
    mix_out = d_model  # final depth-attention query before the head
    return embed + lmhead + layers + mix_out + shared_bank + mtp
