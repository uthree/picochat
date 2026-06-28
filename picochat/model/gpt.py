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
        self.proj_down = nn.Linear(d_hidden, d_model)

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
        n_groups: int | None = None,
        rope_base: int = 10000,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.n_groups = n_heads if n_groups is None else n_groups
        assert n_heads % self.n_groups == 0
        assert d_model % n_heads == 0
        self.proj_q = nn.Linear(d_model, self.d_head * n_heads, bias=False)
        self.proj_k = nn.Linear(d_model, self.d_head * self.n_groups, bias=False)
        self.proj_v = nn.Linear(d_model, self.d_head * self.n_groups, bias=False)
        self.proj_o = nn.Linear(self.d_head * n_heads, d_model, bias=False)
        # RoPE の sin/cos テーブルを最大系列長ぶん事前計算する。派生値なので
        # state_dict には保存せず（persistent=False）、max_seq_len 変更にも追従する。
        sin, cos = self._rope_tables(max_seq_len)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: Tensor, cache: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x = rms_norm(x)
        query = rearrange(self.proj_q(x), "b l (h d) -> b h l d", d=self.d_head)
        key = rms_norm(rearrange(self.proj_k(x), "b l (g d) -> b g l d", d=self.d_head))
        value = rms_norm(
            rearrange(self.proj_v(x), "b l (g d) -> b g l d", d=self.d_head)
        )
        if cache is not None:
            key, value = (
                torch.cat([cache[0], key], dim=-2),
                torch.cat([cache[1], value], dim=-2),
            )
        cache = torch.stack([key, value])
        offset = key.shape[-2] - query.shape[-2]
        query, key = self._rope(query, offset=offset), self._rope(key)
        if offset == 0:
            attn = F.scaled_dot_product_attention(
                query, key, value, is_causal=True, enable_gqa=True
            )
        else:
            # cached decoding: query is shorter than key, so is_causal would
            # align top-left and mask out the cache. Build a bottom-right
            # aligned causal mask instead (query pos i attends to keys <= i+offset).
            q_len, k_len = query.shape[-2], key.shape[-2]
            mask = torch.ones(q_len, k_len, dtype=torch.bool, device=query.device).tril(
                diagonal=offset
            )
            attn = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, enable_gqa=True
            )
        y = self.proj_o(rearrange(attn, "b h l d -> b l (h d)"))
        return y, cache

    def _rope_tables(self, max_seq_len: int) -> tuple[Tensor, Tensor]:
        # 絶対位置 0..max_seq_len-1 に対する sin/cos を作る（offset はスライス側で扱う）。
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
            f"position {offset + seq_len} が max_seq_len={self.max_seq_len} を超えた"
        )
        with torch.amp.autocast(device_type="cuda", enabled=False):
            sin = self.sin[offset : offset + seq_len, :]
            cos = self.cos[offset : offset + seq_len, :]
            x = x * cos + rotate_half(x) * sin
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_groups: int | None = None,
        rope_base: int = 10000,
        d_ffn: int | None = None,
        n_attn_layers: int | None = None,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        if n_attn_layers is None:
            n_attn_layers = n_layers
        assert n_layers % n_attn_layers == 0
        self.n_attn_layers = n_attn_layers
        self.n_layers = n_layers
        self.attn = nn.ModuleList(
            [
                SelfAttention(
                    d_model,
                    n_heads,
                    n_groups=n_groups,
                    rope_base=rope_base,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_attn_layers)
            ]
        )
        self.ffn = nn.ModuleList(
            [SwiGLU(d_model, d_hidden=d_ffn) for _ in range(n_layers)]
        )

    def forward(
        self, x: Tensor, cache: list[Tensor | None] | None
    ) -> tuple[Tensor, list[Tensor]]:
        if cache is None:
            cache = [None] * self.n_layers  # type: ignore
        for i in range(self.n_layers):
            s = x
            j = i % self.n_attn_layers
            x, cache[i] = self.attn[j](x, cache[i])  # type: ignore
            x = x + s
            s = x
            x = self.ffn[i](x)
            x = x + s
        x = rms_norm(x)
        return x, cache  # type: ignore


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_groups: int | None = None,
        rope_base: int = 10000,
        d_ffn: int | None = None,
        n_attn_layers: int | None = None,
        d_embed: int = 128,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Embedding(vocab_size, d_embed), nn.Linear(d_embed, d_model, bias=False)
        )
        self.lmhead = nn.Sequential(
            nn.Linear(d_model, d_embed, bias=False),
            nn.Linear(d_embed, vocab_size, bias=False),
        )
        self.transformer = Transformer(
            d_model,
            n_heads,
            n_layers,
            n_groups=n_groups,
            rope_base=rope_base,
            d_ffn=d_ffn,
            n_attn_layers=n_attn_layers,
            max_seq_len=max_seq_len,
        )

    def forward(
        self, x: Tensor, cache: list[Tensor | None] | None
    ) -> tuple[Tensor, list[Tensor]]:
        x = self.embed(x)
        x, cache = self.transformer(x, cache=cache)
        x = self.lmhead(x)
        return x, cache


# スケールラダー。同じ TransformerLM 引数で pico〜3B を切り替えるためのプリセット。
# 制約: d_model % n_heads == 0, n_heads % n_groups == 0。d_head は 64 か 128。
# params は vocab=64k 込みの実測値（因子化埋め込み d_embed=128 のぶん約16Mを含む）。
MODEL_PRESETS: dict[str, dict] = {
    "pico": dict(d_model=512, n_layers=8, n_heads=8, n_groups=2),  # ~41M
    "small": dict(d_model=768, n_layers=12, n_heads=12, n_groups=4),  # ~99M
    "base": dict(d_model=1024, n_layers=24, n_heads=16, n_groups=4),  # ~306M
    "medium": dict(d_model=2048, n_layers=24, n_heads=16, n_groups=8),  # ~1.2B
    "large": dict(d_model=2560, n_layers=32, n_heads=20, n_groups=5),  # ~2.4B
}


def build_lm(
    size: str,
    vocab_size: int,
    max_seq_len: int = 4096,
    **overrides,
) -> TransformerLM:
    """プリセット名から TransformerLM を作る。overrides で個別に上書き可能。"""
    if size not in MODEL_PRESETS:
        raise ValueError(f"unknown size '{size}'. choices: {list(MODEL_PRESETS)}")
    cfg = {**MODEL_PRESETS[size], **overrides}
    return TransformerLM(vocab_size=vocab_size, max_seq_len=max_seq_len, **cfg)


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
        logits, _cache = self.model(x, None)
        # next-token prediction: the output at position i predicts token i+1.
        logits = rearrange(logits[:, :-1], "b l v -> (b l) v")
        targets = rearrange(x[:, 1:], "b l -> (b l)")
        loss = F.cross_entropy(logits, targets, ignore_index=self.pad_idx)
        return loss

    def _log(self, name: str, value: Tensor, **kwargs) -> None:
        # Trainer に接続されているときだけ記録する（テストで step を直接呼ぶ場合は no-op）。
        if self._trainer is not None:
            self.log(name, value, prog_bar=True, **kwargs)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._loss(batch)
        self._log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._loss(batch)
        self._log("val_loss", loss, sync_dist=True)
        return loss

    def _param_groups(self) -> list[dict]:
        # weight decay は 2次元以上の重みにだけ掛ける。bias（1次元）と
        # embedding は除外する（rms_norm は学習パラメータを持たないので対象外）。
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
        # 線形 warmup -> cosine decay（min_lr_ratio まで下げる）。
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
            # 学習ホライズンが不明なときはスケジューラ無し（optimizer のみ）。
            return optimizer
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
