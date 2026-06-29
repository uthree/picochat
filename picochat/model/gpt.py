import math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def rms_norm(x: Tensor, eps: float = 1e-8) -> Tensor:
    with torch.amp.autocast(device_type="cuda", enabled=False):
        return x / (x.square().mean(-1, keepdim=True).sqrt() + eps)


def rotate_half(x: Tensor) -> Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = torch.stack([-x2, x1], dim=-1)
    x = rearrange(x, "... d r -> ... (d r)")
    return x


class SwiGLU(nn.Module):
    def __init__(
        self, d_model: int, d_hidden: int | None = None, p_dropout: float = 0.1
    ):
        super().__init__()
        self.p_dropout = p_dropout
        if d_hidden is None:
            d_hidden = d_model * 3
        self.proj_up = nn.Linear(d_model, d_hidden)
        self.proj_gate = nn.Linear(d_model, d_hidden)
        self.proj_down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = rms_norm(x)
        x = self.proj_up(x) * F.silu(self.proj_gate(x))
        x = F.dropout(x, self.p_dropout, training=self.training)
        x = self.proj_down(x)
        x = F.dropout(x, self.p_dropout, training=self.training)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        rope_base: int = 10000,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        # Specify the number of query heads; the per-head dim is derived so proj_q
        # stays square (d_head * n_heads == d_model). GQA is set by n_kv_heads.
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0  # heads tile d_model
        assert n_heads % self.n_kv_heads == 0  # GQA grouping
        assert self.d_head % 2 == 0  # RoPE rotates dimension pairs

        self.proj_q = nn.Linear(d_model, self.d_head * n_heads, bias=False)
        self.proj_k = nn.Linear(d_model, self.d_head * self.n_kv_heads, bias=False)
        self.proj_v = nn.Linear(d_model, self.d_head * self.n_kv_heads, bias=False)
        self.proj_o = nn.Linear(self.d_head * n_heads, d_model, bias=False)

        sin, cos = self._rope_tables(max_seq_len)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Shared q/k/v projection (with QK-norm) for both forward and decode.
        x = rms_norm(x)
        query = rearrange(self.proj_q(x), "b l (h d) -> b h l d", d=self.d_head)
        key = rms_norm(rearrange(self.proj_k(x), "b l (g d) -> b g l d", d=self.d_head))
        value = rms_norm(
            rearrange(self.proj_v(x), "b l (g d) -> b g l d", d=self.d_head)
        )
        return query, key, value

    def forward(self, x: Tensor) -> Tensor:
        # Training path: full causal attention over the whole sequence, no cache.
        query, key, value = self._project(x)
        query, key = self._rope(query), self._rope(key)
        attn = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, enable_gqa=True
        )
        return self.proj_o(rearrange(attn, "b h l d -> b l (h d)"))

    def decode(self, x: Tensor, cache: Tensor | None = None) -> tuple[Tensor, Tensor]:
        # Inference path: append the new keys/values to the cache and attend over
        # the full prefix. Returns the updated cache (stacked [key, value]).
        query, key, value = self._project(x)
        if cache is not None:
            key = torch.cat([cache[0], key], dim=-2)
            value = torch.cat([cache[1], value], dim=-2)
        new_cache = torch.stack([key, value])
        # query covers the last q_len positions; key covers the whole prefix.
        offset = key.shape[-2] - query.shape[-2]
        query, key = self._rope(query, offset=offset), self._rope(key)
        # Bottom-right aligned causal mask: query pos i attends to keys <= i+offset
        # (tril with diagonal=offset; reduces to plain causal when offset == 0).
        q_len, k_len = query.shape[-2], key.shape[-2]
        mask = torch.ones(q_len, k_len, dtype=torch.bool, device=query.device).tril(
            diagonal=offset
        )
        attn = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, enable_gqa=True
        )
        return self.proj_o(rearrange(attn, "b h l d -> b l (h d)")), new_cache

    def _rope_tables(self, max_seq_len: int) -> tuple[Tensor, Tensor]:
        # Build sin/cos for absolute positions 0..max_seq_len-1 (offset is handled
        # later when slicing).
        t = torch.arange(max_seq_len)[:, None].float()
        f = (
            self.rope_base
            ** (torch.linspace(0.0, 1.0, self.d_head // 2).repeat_interleave(2))
        )[None, :]
        theta = t / f
        return torch.sin(theta), torch.cos(theta)

    def _rope(self, x, offset: int = 0) -> Tensor:
        seq_len = x.shape[-2]
        assert offset + seq_len <= self.max_seq_len, (
            f"position {offset + seq_len} exceeds max_seq_len={self.max_seq_len}"
        )
        # Apply RoPE in float32 to keep positional precision under bf16 autocast.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            sin = self.sin[offset : offset + seq_len, :]
            cos = self.cos[offset : offset + seq_len, :]
            x = x * cos + rotate_half(x) * sin
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        rope_base: int = 10000,
        d_ffn: int | None = None,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.attn = SelfAttention(
            d_model,
            n_heads,
            n_kv_heads=n_kv_heads,
            rope_base=rope_base,
            max_seq_len=max_seq_len,
        )
        self.ffn = SwiGLU(d_model, d_hidden=d_ffn)

    def forward(self, x: Tensor) -> Tensor:
        # attn/ffn apply pre-norm (rms_norm) internally, so add the raw residual.
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return x

    def decode(self, x: Tensor, cache: Tensor | None = None) -> tuple[Tensor, Tensor]:
        a, cache = self.attn.decode(x, cache)
        x = a + x
        x = self.ffn(x) + x
        return x, cache


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_kv_heads: int | None = None,
        rope_base: int = 10000,
        d_ffn: int | None = None,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model,
                    n_heads,
                    n_kv_heads=n_kv_heads,
                    rope_base=rope_base,
                    d_ffn=d_ffn,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return rms_norm(x)

    def decode(
        self, x: Tensor, cache: list[Tensor | None] | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        if cache is None:
            cache = [None] * self.n_layers
        for i, layer in enumerate(self.layers):
            x, cache[i] = layer.decode(x, cache[i])
        return rms_norm(x), cache  # type: ignore


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_kv_heads: int | None = None,
        rope_base: int = 10000,
        d_ffn: int | None = None,
        max_seq_len: int = 4096,
        tie_embeddings: bool = True,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.tie_embeddings = tie_embeddings
        self.init_std = init_std

        self.embed = nn.Embedding(vocab_size, d_model)
        self.lmhead = nn.Linear(d_model, vocab_size, bias=False)
        self.transformer = Transformer(
            d_model,
            n_heads,
            n_layers,
            n_kv_heads=n_kv_heads,
            rope_base=rope_base,
            d_ffn=d_ffn,
            max_seq_len=max_seq_len,
        )
        self._init_weights()
        if tie_embeddings:
            # Share one matrix for input/output. At small scale + large vocab this
            # is the better operating point (param-efficient, and rare-token rows
            # get gradient from input occurrences too, not just when they are the
            # target). Untie at larger scale to give the output its own capacity.
            # Tie after init so the output reuses the embedding's small init std;
            # the default nn.Embedding init (std 1) would blow up the tied logits.
            self.lmhead.weight = self.embed.weight

    def _init_weights(self) -> None:
        # GPT-2 style: init every weight with normal(0, init_std) and zero biases,
        # then scale down the projections that write into the residual stream
        # (proj_o / proj_down) by 1/sqrt(2*n_layers) so the residual variance stays
        # roughly constant with depth.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=self.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=self.init_std)
        scaled_std = self.init_std / math.sqrt(2 * self.n_layers)
        for m in self.modules():
            if isinstance(m, SelfAttention):
                nn.init.normal_(m.proj_o.weight, mean=0.0, std=scaled_std)
            elif isinstance(m, SwiGLU):
                nn.init.normal_(m.proj_down.weight, mean=0.0, std=scaled_std)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        x = self.transformer(x)
        return self.lmhead(x)

    def decode(
        self, x: Tensor, cache: list[Tensor | None] | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        x = self.embed(x)
        x, cache = self.transformer.decode(x, cache)
        return self.lmhead(x), cache


# Scale ladder. Presets to switch between pico..3B with the same TransformerLM args.
# Specify n_heads (query heads); the per-head dim is derived as d_model//n_heads
# (proj_q stays square). GQA is set by n_kv_heads. Constraints: d_model % n_heads
# == 0, n_heads % n_kv_heads == 0, d_model//n_heads even. The derived d_head is
# 64 (pico/small/base) or 128 (medium/large).
# vocab_size and tie_embeddings are scale-dependent: small models use a smaller
# vocab (64k is oversized there -> many undertrained rows) and tie embeddings;
# larger models use the full 64k vocab and untie to give the output its own
# capacity. Param counts below include the (tied or untied) embeddings.
MODEL_PRESETS: dict[str, dict] = {
    "pico": dict(
        d_model=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        vocab_size=32000,
        tie_embeddings=True,
    ),  # ~40M
    "small": dict(
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        vocab_size=32000,
        tie_embeddings=True,
    ),  # ~107M
    "base": dict(
        d_model=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        vocab_size=64000,
        tie_embeddings=False,
    ),  # ~421M
    "medium": dict(
        d_model=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        vocab_size=64000,
        tie_embeddings=False,
    ),  # ~1.5B
    "large": dict(
        d_model=2560,
        n_layers=32,
        n_heads=20,
        n_kv_heads=5,
        vocab_size=64000,
        tie_embeddings=False,
    ),  # ~2.7B
}


def build_lm(
    size: str,
    vocab_size: int | None = None,
    max_seq_len: int = 4096,
    **overrides,
) -> TransformerLM:
    """Build a TransformerLM from a preset name. vocab_size defaults to the
    preset's recommended value; pass it explicitly (e.g. the tokenizer's actual
    vocab) to override. Any other field can be overridden via overrides."""
    if size not in MODEL_PRESETS:
        raise ValueError(f"unknown size '{size}'. choices: {list(MODEL_PRESETS)}")
    cfg = {**MODEL_PRESETS[size], **overrides}
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return TransformerLM(max_seq_len=max_seq_len, **cfg)


class GPT(L.LightningModule):
    def __init__(
        self,
        transformer_lm: TransformerLM,
        pad_idx: int = 0,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        warmup_steps: int = 2000,
        max_steps: int | None = None,
        min_lr_ratio: float = 0.1,
    ):
        super().__init__()
        self.model = transformer_lm
        self.pad_idx = pad_idx
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio

    def _loss(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        # next-token prediction: the output at position i predicts token i+1.
        logits = rearrange(logits[:, :-1], "b l v -> (b l) v")
        targets = rearrange(x[:, 1:], "b l -> (b l)")
        loss = F.cross_entropy(logits, targets, ignore_index=self.pad_idx)
        return loss

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._loss(batch)
        self.log("train_loss", loss)
        self.log("loss", loss, prog_bar=True, logger=False)  # for progress bar
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._loss(batch)
        self.log("valid_loss", loss)
        return loss

    def _param_groups(self) -> list[dict]:
        # Apply weight decay only to weights with 2+ dims. Exclude biases (1-dim)
        # and embeddings (rms_norm has no learnable params, so nothing to exclude there).
        embed_ids = {
            id(p)
            for m in self.model.modules()
            if isinstance(m, nn.Embedding)
            for p in m.parameters()
        }
        decay, no_decay = [], []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or id(p) in embed_ids:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _lr_lambda(self, step: int) -> float:
        # Linear warmup -> cosine decay (down to min_lr_ratio).
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        if self.max_steps is None or step >= self.max_steps:
            return self.min_lr_ratio
        progress = (step - self.warmup_steps) / max(
            1, self.max_steps - self.warmup_steps
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * coeff

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._param_groups(), lr=self.lr, betas=self.betas
        )
        if self.max_steps is None:
            # No schedule when the training horizon is unknown (optimizer only).
            return optimizer
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
