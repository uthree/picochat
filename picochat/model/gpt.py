import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.optim.optimizer import Optimizer


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
    ):
        super().__init__()
        self.rope_base = rope_base
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

    def _rope(self, x, offset: int = 0) -> Tensor:
        with torch.amp.autocast(device_type="cuda", enabled=False):
            if not hasattr(self, "sin"):
                t = (torch.arange(self.rope_base) + offset)[:, None].float()
                f = (
                    self.rope_base
                    ** (torch.linspace(0.0, 1.0, self.d_head // 2).repeat_interleave(2))
                )[None, :]
                theta = t / f
                self.register_buffer("sin", torch.sin(theta))
                self.register_buffer("cos", torch.cos(theta))
            sin, cos = (
                self.sin[offset : offset + x.shape[-2], :],
                self.cos[offset : offset + x.shape[-2], :],
            )
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
    ):
        super().__init__()
        if n_attn_layers is None:
            n_attn_layers = n_layers
        assert n_layers % n_attn_layers == 0
        self.n_attn_layers = n_attn_layers
        self.n_layers = n_layers
        self.attn = nn.ModuleList(
            [
                SelfAttention(d_model, n_heads, n_groups=n_groups, rope_base=rope_base)
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
        )

    def forward(
        self, x: Tensor, cache: list[Tensor | None] | None
    ) -> tuple[Tensor, list[Tensor]]:
        x = self.embed(x)
        x, cache = self.transformer(x, cache=cache)
        x = self.lmhead(x)
        return x, cache


class GPT(L.LightningModule):
    def __init__(
        self, transformer_lm: TransformerLM, pad_idx: int = 0, lr: float = 1e-4
    ):
        super().__init__()
        self.model = transformer_lm
        self.pad_idx = pad_idx
        self.lr = lr

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

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
