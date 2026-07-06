"""Muon optimizer (MomentUm Orthogonalized by Newton-Schulz, Keller Jordan et
al.) with an embedded AdamW for the parameters Muon can't sensibly update
(embeddings, lm heads, 1-dim params). Bundling both into one Optimizer keeps
Lightning's single-optimizer training loop and LR scheduling unchanged.

Why not torch.optim.Muon: it rejects non-2D parameters outright, so it can't
take the fused 3D MoE expert weights (n_experts, out, in) that this
implementation flattens per-step; and it is Muon-only, so the AdamW side would
need a second optimizer (= Lightning manual optimization). The math matches
torch's (same NS coefficients, scale correction = adjust_lr_fn="original")."""

import torch
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """Approximate the nearest semi-orthogonal matrix (UV^T from G's SVD) via
    a quintic Newton-Schulz iteration, run in bfloat16.

    The coefficients are tuned to maximize the convergence slope at zero
    rather than to converge exactly, so the result's singular values land in
    roughly [0.7, 1.2] instead of exactly 1 -- close enough for an optimizer
    update, and much cheaper than an SVD.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transpose = G.size(0) > G.size(1)
    if transpose:
        # Iterate on the wide orientation so X @ X.T below is the smaller
        # gram matrix.
        X = X.T
    X = X / (X.norm() + 1e-7)  # spectral norm <= frobenius norm -> <= 1
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon for matrix-shaped hidden weights, AdamW for everything else.

    Every param group must carry a `use_muon` flag:

    - `use_muon=True`: SGD-momentum (nesterov) whose update is orthogonalized
      with Newton-Schulz before being applied, with decoupled weight decay.
      Params with ndim > 2 -- the fused MoE expert weights, shaped
      (n_experts, out_features, in_features) -- are flattened to
      (n_experts * out_features, in_features) for the orthogonalization and
      reshaped back afterwards.
      Options: lr, momentum, weight_decay, ns_steps.
    - `use_muon=False`: standard decoupled AdamW.
      Options: lr, betas, eps, weight_decay.
    """

    def __init__(self, param_groups: list[dict]):
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("every Muon param group needs a `use_muon` flag")
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("ns_steps", 5)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: dict) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            assert p.ndim >= 2, "route params Muon can't handle to a use_muon=False group"
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]
            buf.lerp_(p.grad, 1 - group["momentum"])
            update = p.grad.lerp(buf, group["momentum"])  # nesterov momentum
            # Flatten leading axes (n_experts for fused MoE weights) into rows
            # so Newton-Schulz sees one 2D matrix per parameter.
            flat = update.reshape(-1, update.shape[-1])
            flat = zeropower_via_newtonschulz5(flat, group["ns_steps"])
            # Scale tall matrices back up: orthogonalization caps every
            # singular value at ~1, which shrinks the per-row RMS of updates
            # with many more rows than columns.
            flat = flat * max(1.0, flat.shape[0] / flat.shape[1]) ** 0.5
            p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(flat.reshape(p.shape).to(p.dtype), alpha=-group["lr"])

    def _adamw_step(self, group: dict) -> None:
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "exp_avg" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            step = state["step"]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.lerp_(p.grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            denom = (exp_avg_sq / (1 - beta2**step)).sqrt_().add_(group["eps"])
            p.mul_(1 - group["lr"] * group["weight_decay"])
            p.addcdiv_(exp_avg, denom, value=-group["lr"] / (1 - beta1**step))
