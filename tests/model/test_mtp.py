import torch
import torch.nn.functional as F

from picochat.model.gpt import GPT, TransformerLM


# ---------------------------------------------------------------------------
# Multiple token prediction (n_lmheads > 1)
# ---------------------------------------------------------------------------
def test_mtp_forward_returns_one_logits_per_head():
    vocab_size = 40
    lm = TransformerLM(
        vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2, n_lmheads=3
    )
    tokens = torch.randint(0, vocab_size, (2, 6))
    logits = lm(tokens)
    assert len(logits) == 3
    assert all(head.shape == (2, 6, vocab_size) for head in logits)


def test_mtp_heads_are_independent_params():
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2, n_lmheads=2)
    assert lm.lmheads[0].weight is not lm.lmheads[1].weight


# ---------------------------------------------------------------------------
# Weight tying (tie_embeddings)
# ---------------------------------------------------------------------------
def test_tie_embeddings_shares_the_same_parameter():
    lm = TransformerLM(
        vocab_size=40, d_model=32, n_heads=4, n_layers=2, tie_embeddings=True
    )
    assert lm.lmheads[0].weight is lm.embed.weight


def test_tie_embeddings_off_by_default_keeps_independent_params():
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    assert lm.lmheads[0].weight is not lm.embed.weight


def test_tie_embeddings_only_ties_head_zero():
    # extra MTP heads are a training-time signal separate from next-token
    # prediction, so only lmheads[0] shares the embedding.
    lm = TransformerLM(
        vocab_size=40, d_model=32, n_heads=4, n_layers=2, n_lmheads=3,
        tie_embeddings=True,
    )
    assert lm.lmheads[0].weight is lm.embed.weight
    assert lm.lmheads[1].weight is not lm.embed.weight
    assert lm.lmheads[2].weight is not lm.embed.weight


def test_tie_embeddings_backward_accumulates_gradient_on_shared_param():
    lm = TransformerLM(
        vocab_size=40, d_model=32, n_heads=4, n_layers=2, tie_embeddings=True
    )
    tokens = torch.randint(0, 40, (2, 6))
    lm(tokens)[0].sum().backward()
    assert lm.embed.weight.grad is not None
    assert lm.lmheads[0].weight.grad is lm.embed.weight.grad


def test_tie_embeddings_reduces_param_count_by_one_head():
    untied = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    tied = TransformerLM(
        vocab_size=40, d_model=32, n_heads=4, n_layers=2, tie_embeddings=True
    )
    n_untied = sum(p.numel() for p in untied.parameters())
    n_tied = sum(p.numel() for p in tied.parameters())
    assert n_untied - n_tied == 40 * 32  # one fewer vocab_size*d_model head


def test_mtp_decode_uses_next_token_head():
    # decode must emit a single logits tensor (from head 0), matching head 0 of
    # a full forward, so autoregressive generation ignores the extra heads.
    vocab_size = 40
    lm = TransformerLM(
        vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2, n_lmheads=3
    ).eval()
    tokens = torch.randint(0, vocab_size, (1, 5))
    full = lm(tokens)[0]
    step, _, _ = lm.decode(tokens)
    assert torch.allclose(full, step, atol=1e-4)


def test_mtp_head_losses_shift_targets():
    # head k's output at position i is scored against token i+1+k: feed an
    # oracle that emits one-hot logits for exactly that token and the loss of
    # every head must be ~0 (any off-by-one in the shifts would blow it up).
    vocab_size = 10
    n_lmheads = 3
    lm = TransformerLM(
        vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=1, n_lmheads=n_lmheads
    )
    gpt = GPT(lm, pad_idx=0, compile=False)
    x = torch.randint(1, vocab_size, (2, 7))

    # Replace each lm head with an oracle that ignores the hidden state and emits
    # one-hot logits for the token k+1 steps ahead; the trailing positions (no
    # such target) stay zeros, which _head_loss must slice off.
    class _OracleHead(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.register_buffer("logits", logits)

        def forward(self, h):
            return self.logits

    heads = []
    for k in range(n_lmheads):
        shift = k + 1
        logits = torch.zeros(*x.shape, vocab_size)
        logits[:, : x.shape[1] - shift] = (
            F.one_hot(x[:, shift:], vocab_size).float() * 200.0
        )
        heads.append(_OracleHead(logits))
    gpt.model.lmheads = torch.nn.ModuleList(heads)

    losses = gpt._head_losses(x)
    assert losses.shape == (n_lmheads,)
    assert torch.allclose(losses, torch.zeros(n_lmheads), atol=1e-3)


def test_mtp_loss_backward_reaches_all_heads():
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2, n_lmheads=3)
    gpt = GPT(lm, pad_idx=0, compile=False)
    batch = torch.randint(1, 40, (2, 8))
    losses = gpt._head_losses(batch)
    assert losses.shape == (3,)
    assert torch.isfinite(losses).all()
    losses.mean().backward()
    for head in lm.lmheads:
        assert head.weight.grad is not None
        assert head.weight.grad.abs().sum() > 0


def test_mtp_training_step_scalar_loss():
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2, n_lmheads=2)
    gpt = GPT(lm, pad_idx=0, compile=False)
    batch = torch.randint(1, 40, (2, 8))
    loss = gpt.training_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_mtp_backward_matches_naive_summed_loss():
    # the memory-optimized two-stage backward (detach trunk, backprop each head,
    # then one trunk backward) must yield exactly the same gradients as a plain
    # backward of the mean-over-heads loss.
    torch.manual_seed(0)
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2, n_lmheads=3)
    gpt = GPT(lm, pad_idx=0, compile=False).eval()  # eval: fixed (no) dropout masks
    batch = torch.randint(1, 40, (2, 8))

    gpt.zero_grad(set_to_none=True)
    gpt._loss(batch).backward()  # reference: mean over heads, single backward
    ref = {n: p.grad.clone() for n, p in lm.named_parameters()}

    gpt.zero_grad(set_to_none=True)
    gpt._mtp_backward(batch, scale=1.0)  # memory-optimized path
    for n, p in lm.named_parameters():
        assert torch.allclose(p.grad, ref[n], atol=1e-5), n


def test_mtp_backward_scale_scales_grads():
    # `scale` (gradient accumulation) linearly scales every gradient.
    torch.manual_seed(0)
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2, n_lmheads=2)
    gpt = GPT(lm, pad_idx=0, compile=False).eval()  # eval: fixed (no) dropout masks
    batch = torch.randint(1, 40, (2, 8))

    gpt.zero_grad(set_to_none=True)
    gpt._mtp_backward(batch, scale=1.0)
    full = lm.embed.weight.grad.clone()

    gpt.zero_grad(set_to_none=True)
    gpt._mtp_backward(batch, scale=0.25)
    assert torch.allclose(lm.embed.weight.grad, full * 0.25, atol=1e-6)


def test_mtp_generate_works():
    lm = TransformerLM(
        vocab_size=40, d_model=32, n_heads=4, n_layers=2, n_lmheads=2, max_seq_len=64
    )
    gpt = GPT(lm, pad_idx=0, compile=False).eval()
    prompt = torch.randint(1, 40, (1, 8))
    generated = gpt._generate(prompt, max_new_tokens=6)
    assert generated.shape == (1, 6)
