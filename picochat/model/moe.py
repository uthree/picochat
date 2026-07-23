"""Mixture of Experts: DeepSeek-V3-style sigmoid routing with aux-loss-free
load balancing, optional LatentMoE compression and an ExpertBank that can be
shared across layers (MoEUT-style)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from picochat.model.blocks import rms_norm


class ExpertBank(nn.Module):
    """The fused routed-expert weights of a MixtureOfExperts, factored out so
    they can be owned per layer (the default) or built once and shared by
    every layer's MoE (Transformer's share_experts, MoEUT-style): each layer
    then keeps its own router, load-balancing bias and latent projections
    while dispatching into the same expert pool.

    The weights are stored 2D, experts stacked along the rows --
    (n_experts * out_features, in_features) -- and viewed back to
    (n_experts, out, in) for the bmm in forward. torch.optim.Muon accepts
    only 2D parameters, and this flattened matrix is exactly what Muon
    orthogonalizes for a stacked expert weight anyway.

    A shared bank is registered as a submodule of every owning MoE:
    parameters()/modules() deduplicate it, torch.save stores the shared
    storage once, and load_state_dict writes the same values through each
    duplicate key -- harmless in both directions.
    """

    def __init__(self, n_experts: int, d_io: int, d_hidden: int):
        super().__init__()
        self.n_experts = n_experts
        self.d_io = d_io  # d_model, or d_latent for a LatentMoE
        self.d_hidden = d_hidden
        self.weight_up = nn.Parameter(torch.empty(n_experts * d_hidden, d_io))
        self.weight_gate = nn.Parameter(torch.empty(n_experts * d_hidden, d_io))
        self.weight_down = nn.Parameter(torch.empty(n_experts * d_io, d_hidden))
        for w in (self.weight_up, self.weight_gate, self.weight_down):
            nn.init.normal_(w, mean=0.0, std=0.02)

    def forward(self, expert_in: Tensor, p_dropout: float = 0.0) -> Tensor:
        # expert_in (n_experts, capacity, d_io): each expert's dispatched
        # tokens; runs every expert's SwiGLU as one batched bmm.
        w_up = self.weight_up.view(self.n_experts, self.d_hidden, self.d_io)
        w_gate = self.weight_gate.view(self.n_experts, self.d_hidden, self.d_io)
        w_down = self.weight_down.view(self.n_experts, self.d_io, self.d_hidden)
        up = torch.bmm(expert_in, w_up.transpose(1, 2))
        gate = torch.bmm(expert_in, w_gate.transpose(1, 2))
        h = F.dropout(up * F.silu(gate), p_dropout, training=self.training)
        out = torch.bmm(h, w_down.transpose(1, 2))
        return F.dropout(out, p_dropout, training=self.training)


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int | None = None,
        p_dropout: float = 0.1,
        n_experts: int = 8,
        n_active: int = 2,
        capacity_factor: float = 1.25,
        bias_update_rate: float = 1e-3,
        d_latent: int | None = None,
        bank: ExpertBank | None = None,
    ):
        super().__init__()
        assert n_active <= n_experts
        self.p_dropout = p_dropout
        self.n_experts = n_experts
        self.n_active = n_active
        # capacity_factor: slack over the perfectly-even per-expert token count
        # (see forward). bias_update_rate: how fast expert_bias below chases
        # load balance.
        self.capacity_factor = capacity_factor
        self.bias_update_rate = bias_update_rate
        if d_hidden is None:
            d_hidden = d_model * 3
        self.d_hidden = d_hidden
        # LatentMoE (arXiv:2601.18089): when d_latent is set, tokens are
        # compressed to that dimension by a shared down-projection before
        # dispatch, the routed experts run their SwiGLU entirely in the latent
        # space (the intermediate d_hidden stays as configured, preserving the
        # nonlinear budget), and the combined output is expanded back to
        # d_model by a shared up-projection. Expert weight loading and token
        # dispatch then cost d_latent instead of d_model each, the saving the
        # paper reinvests in more experts / higher top-k (a preset choice, via
        # n_experts / n_active). The router (below) and the dense FFN that
        # TransformerLayer runs in parallel (the paper's shared expert) stay
        # in d_model. d_latent=None keeps the standard MoE unchanged.
        self.d_latent = d_latent
        d_expert_io = d_model if d_latent is None else d_latent
        self.d_expert_io = d_expert_io
        self.weight_router = nn.Parameter(torch.empty(n_experts, d_model))
        # The routed-expert weights live in an ExpertBank (see there) so they
        # can be shared across layers: with a shared bank passed in, this MoE
        # owns only its router, load-balancing bias and latent projections.
        if bank is None:
            bank = ExpertBank(n_experts, d_expert_io, d_hidden)
        assert bank.n_experts == n_experts
        assert bank.d_io == d_expert_io and bank.d_hidden == d_hidden
        self.bank = bank
        weights = [self.weight_router]
        if d_latent is not None:
            self.weight_compress = nn.Parameter(torch.empty(d_latent, d_model))
            self.weight_expand = nn.Parameter(torch.empty(d_model, d_latent))
            weights += [self.weight_compress, self.weight_expand]
        for w in weights:
            nn.init.normal_(w, mean=0.0, std=0.02)
        # Normalize the aggregated (routed) expert output before it re-enters the
        # residual stream. The summed contribution of the selected experts has a
        # scale that drifts with routing, while fine-grained MoE outputs come out
        # systematically scaled down versus a dense FFN. A learnable RMSNorm here
        # rescales it and stabilizes training; it gives a small gain for standard
        # MoE but is crucial for fine-grained experts (Scaling Laws for
        # Fine-Grained MoE, arXiv:2402.07871), the same output normalization
        # DeepSeek-V3 / Kimi-style stacks apply. Runs on the d_model output (after
        # the latent expansion when d_latent is set).
        self.out_gain = nn.Parameter(torch.ones(d_model))
        # DeepSeek-V3 style aux-loss-free load balancing: a per-expert bias
        # that only steers *which* experts get picked (added before top-k,
        # dropped again before computing combine weights below), nudged every
        # training step toward under-loaded experts. Not a Parameter -- no
        # gradient, no loss term, just a running buffer.
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        # The per-expert load counts are *staged* here in forward and turned
        # into a bias update once by Transformer.forward, outside any
        # gradient-checkpoint boundary -- see forward() / apply_bias_update().
        # Non-persistent: derived per-step, not part of the checkpointed state.
        self.register_buffer("_pending_load", torch.zeros(n_experts), persistent=False)

    def forward(self, x: Tensor, no_drop: bool = False) -> Tensor:
        b, t, d = x.shape
        n_tokens = b * t
        tokens = rms_norm(x).reshape(n_tokens, d)

        # Route every token to its top-n_active experts (DeepSeek-V3 style). The
        # router affinity is a per-expert sigmoid; expert_bias only steers *which*
        # experts are picked (aux-loss-free load balancing), while the combine
        # weights are the selected affinities renormalized to sum to 1, so they
        # stay a differentiable function of weight_router alone.
        scores = torch.sigmoid(tokens @ self.weight_router.T)  # (n_tokens, n_experts)
        top_idx = (scores + self.expert_bias).topk(self.n_active, dim=-1).indices
        top_scores = scores.gather(-1, top_idx)  # (T, n_active)
        top_weight = top_scores / top_scores.sum(-1, keepdim=True).clamp_min(1e-9)

        # Flatten the (token, expert) assignments to one row per routed pair, in
        # slot-major order so a token's 1st-choice expert claims capacity before
        # its 2nd choice.
        pair_expert = top_idx.T.reshape(-1)  # (n_active * T,)
        pair_token = torch.arange(n_tokens, device=x.device).repeat(self.n_active)
        pair_weight = top_weight.T.reshape(-1)

        # Give each pair a slot within its expert (its rank among pairs routed
        # there). Fixed per-expert capacity keeps every tensor statically shaped
        # -- traceable under torch.compile, unlike a data-dependent gather -- and
        # pairs past capacity are dropped (Switch Transformer / GShard style).
        counts = F.one_hot(pair_expert, self.n_experts)  # (pairs, n_experts)
        slot = (counts.cumsum(0) - 1).gather(-1, pair_expert[:, None]).squeeze(-1)
        if no_drop:
            # Decode: size capacity so nothing overflows (a token routes to each
            # expert at most once, so no expert gets more than n_tokens pairs).
            # Makes the layer purely per-token, so chunked decode matches a full
            # forward exactly. Only reached from decode(), where n_tokens is small.
            capacity = n_tokens
        else:
            capacity = max(
                self.n_active,
                math.ceil(
                    n_tokens * self.n_active * self.capacity_factor / self.n_experts
                ),
            )
        keep = slot < capacity
        n_slots = self.n_experts * capacity
        # Dropped pairs are redirected to a trash-bin row (index n_slots) so the
        # scatter/gather below use a fixed-size index instead of a dynamic one.
        dest = torch.where(keep, pair_expert * capacity + slot, n_slots)

        if self.training:
            # Stage the raw load counts rather than updating the bias here.
            # Under gradient checkpointing this forward runs twice (once for
            # real, once recomputed in backward); staging into a buffer is
            # idempotent, and Transformer.forward turns it into exactly one
            # bias update outside the checkpoint (where the counts are also
            # summed across DDP ranks -- see apply_bias_update).
            with torch.no_grad():
                # tokens routed to each expert
                self._pending_load.copy_(counts.sum(0).float())

        # Dispatch pairs into per-expert buffers, run the expert SwiGLU, combine.
        # With d_latent set (LatentMoE), dispatch, expert computation and the
        # combine all happen in the latent dimension d_io; only the final
        # expansion returns to d_model. Weighting the contributions before the
        # (linear) expansion matches the paper's W_up(sum p_i * E_i) exactly.
        d_io = self.d_expert_io
        inputs = tokens if self.d_latent is None else tokens @ self.weight_compress.T
        keep_f = keep.unsqueeze(-1).to(tokens.dtype)
        buffer = inputs.new_zeros(n_slots + 1, d_io).index_add(
            0, dest, inputs[pair_token] * keep_f
        )
        expert_in = buffer[:n_slots].reshape(self.n_experts, capacity, d_io)
        expert_out = self.bank(expert_in, self.p_dropout)
        expert_out = torch.cat(
            [expert_out.reshape(n_slots, d_io), expert_out.new_zeros(1, d_io)], dim=0
        )

        contrib = expert_out[dest] * (pair_weight.unsqueeze(-1) * keep_f)
        out = inputs.new_zeros(n_tokens, d_io).index_add(0, pair_token, contrib)
        if self.d_latent is not None:
            out = out @ self.weight_expand.T
        out = rms_norm(out) * self.out_gain  # normalize the aggregated output
        return out.reshape(b, t, d)

    @torch.no_grad()
    def apply_bias_update(self) -> None:
        # Turn the load counts staged by the most recent forward into one bias
        # step toward under-loaded experts. Called by Transformer.forward once
        # per step, outside the gradient-checkpoint boundary, so the update
        # lands exactly once whether or not the layer was recomputed in
        # backward. Under DDP the counts are summed across ranks first: the
        # bias then chases the *global* batch's load (as DeepSeek-V3 does) and
        # stays identical on every rank instead of following rank 0's local
        # batch only (DDP broadcasts buffers from rank 0 each forward, which
        # would silently discard the other ranks' updates otherwise).
        load = self._pending_load
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            load = load.clone()
            torch.distributed.all_reduce(load)
        self.expert_bias += self.bias_update_rate * torch.sign(load.mean() - load)


def moe_modules(model: nn.Module) -> list[MixtureOfExperts]:
    """Every routed-expert layer in the model (empty list for a dense model)."""
    return [m for m in model.modules() if isinstance(m, MixtureOfExperts)]
