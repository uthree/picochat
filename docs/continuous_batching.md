# Continuous batching for the API server — design notes

Status: **design only, not implemented.** The API server
(`picochat/inference/api.py`) currently serves one request at a time behind a
`generation_lock`. This note scopes what continuous (in-flight) batching would
take, because the picochat architecture makes it less standard than for a
plain transformer.

## Why it's wanted

Single-request serving leaves the accelerator idle between a request's decode
steps and can't overlap the prefill of a new request with the decode of
another. Continuous batching (vLLM-style) keeps a running batch of active
sequences, admits new ones as slots free up, and evicts finished ones — the
standard way to raise serving throughput for many concurrent users.

## What makes picochat different

The hybrid GDN-2 / NSA stack does not have a single uniform KV cache, so the
usual "concatenate everyone's KV and mask" batching needs per-mixer handling:

1. **NSA (sparse attention) layers** keep a growing per-sequence K/V dict
   (`sparse_attn.NativeSparseAttention.decode`). Batching these means padding
   K/V to the batch's max length and carrying a per-row length — ordinary
   ragged-KV batching, the same problem vLLM solves with paged attention.

2. **GDN-2 (linear attention) layers** keep a **fixed-size recurrent state**
   per sequence (`(recurrent_state, conv_state)`), *not* a growing cache.
   This is actually the easy part: the state is the same shape for every
   sequence regardless of its length, so a batch is a clean leading batch
   dimension — no padding, no length bookkeeping. `fused_recurrent_gdn2`
   already takes a batched state.

3. **The Block-Attention-Residual (`DepthAttention`) stream** is per-token and
   mask-independent (see `gpt`/`transformer.py`), so it batches trivially.

The upshot: the linear-attention half is *easier* to batch than a normal
transformer (constant-size state), and the sparse half is the standard
ragged-KV problem. There is no fundamental blocker, only engineering.

## Sketch of an implementation

- A scheduler holding a set of active sequences, each with its own
  `(cache, pos, sampling, remaining_budget)`.
- Each decode step: gather the active sequences' next-token inputs into one
  batch, run one batched `model.decode`, scatter the sampled tokens back.
  - GDN-2 layers: stack the per-sequence recurrent/conv states on dim 0 (they
    are equal-shaped) — no padding.
  - NSA layers: pad each sequence's K/V to the batch max length and pass a
    per-row valid-length so masking ignores the padding; free the row's K/V
    when it finishes. (Paged/block allocation avoids repeated re-padding.)
- Admission: when a slot frees, prefill a waiting request (its prompt in one
  forward) and add it to the active set. Prefill and decode can run in
  separate micro-steps or be fused with a "chunked prefill" policy.
- Per-sequence sampling: `sample()` already takes per-row `(B, V)` logits and
  a per-row history, so temperature/top-k/penalties differ per sequence for
  free.

## Why it's deferred

Correctness-critical and invasive: it rewrites the decode loop, the cache
representation for NSA, and the server's request lifecycle, and it needs real
multi-GPU / high-QPS load to validate. The current lock-serialized server is
correct and adequate for single-user and light multi-user use (chat, eval,
agent loops). Continuous batching is the right next step **when** picochat is
deployed for concurrent production traffic, and the design above is the
starting point.
