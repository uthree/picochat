# picochat
![logo](assets/logo.png)

A project inspired by [nanochat](https://github.com/karpathy/nanochat): a
minimal chat-LLM stack you can train from scratch for roughly $100 at the
smallest rung, built so the same code scales up unchanged to much larger models.
The sequence mixer is a hybrid of [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)
linear-attention layers and [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
layers (3:1) for long context at a fraction of the KV cost.

## Requirements
- Python 3.11
- Training: a CUDA GPU. PyTorch resolves from PyPI as CUDA 13 (cu130) wheels,
  which need an r580+ NVIDIA driver. On a data-center GPU (e.g. L4) with an
  older, still-supported driver branch (e.g. r570), install NVIDIA's
  [forward-compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
  package instead of upgrading the driver: `apt install cuda-compat-13-0`, then
  put `/usr/local/cuda-13.0/compat` on the loader path (e.g.
  `echo /usr/local/cuda-13.0/compat > /etc/ld.so.conf.d/00-cuda-compat.conf && ldconfig`)
- Tests and evaluation also run on CPU
- RL post-training (`scripts/grpo_train.py`) executes untrusted, model-generated
  code to score it. Install [bubblewrap](https://github.com/containers/bubblewrap)
  (`apt install bubblewrap`) so it runs sandboxed (isolated filesystem/network);
  without it, or with `PICOCHAT_SANDBOX=none`, code runs in a hardened
  subprocess (rlimits + scrubbed env) but **without** fs/network isolation. Set
  `PICOCHAT_SANDBOX=bwrap` (or `sandbox: bwrap` in the config) to require it.

## Usage

### 1. Set up the environment
We use [uv](https://docs.astral.sh/uv/) for the virtual environment:
```bash
uv venv --python 3.11  # initialize venv
uv pip install -e .    # install dependencies

# optional: Hub-loaded Triton kernels (Liger fused cross-entropy) for
# lower-memory training on CUDA; everything runs without it
uv pip install -e ".[kernels]"
```
On Linux the base install pulls in `fla-core` (flash-linear-attention's Triton
kernels) for the Gated DeltaNet-2 layers; on macOS/Windows it is skipped and
the model uses the built-in pure-PyTorch implementation (also the CPU path
everywhere). Both compute the same result.

### 2. Train the tokenizer
Train a BPE tokenizer (64k vocab, ChatML special tokens; CJK split
per-character so tokens stay morpheme-level) from a YAML recipe:
```bash
uv run scripts/tok_train.py --config configs/tok/default.yml

# evaluate compression (bytes/token, higher = denser); 4-6 is fine
uv run scripts/tok_eval.py
```

### 3. Preprocess the pretraining data
The datasets are tokenized, packed into fixed-length rows (MosaicBERT-style
sequence packing; the config's `block_size` must match the training config)
and written as sharded token binaries under `data/`:
```bash
uv run scripts/base_setup.py --config configs/base_setup/default.yml
```

### 4. Pretrain
A single merged run (STEM / educational + computing & coding, 16k-token rows
for long context, plus conversational-level multilingual coverage -- see
`configs/base_setup/default.yml` for the corpus rationale):
```bash
uv run scripts/base_train.py --config configs/base_train/default.yml
```
Interrupted runs resume automatically from `output_dir/last.ckpt`. Training
curves and generation samples are logged to TensorBoard (`lightning_logs/`).
If you hit out-of-memory errors, reduce `block_size` (repack) or raise
`accumulate` in the config.

To **grow a bigger model** from a smaller trained one instead of pretraining it
from scratch, set `grow_from: <smaller-checkpoint>` in the target preset's config
(see the growth chain under Model Architecture). The larger model starts as
(nearly) the same function the small one learned, then continues training.

### 5. Supervised fine-tuning (SFT)
Preprocess the chat corpus, then fine-tune the final pretraining checkpoint:
```bash
uv run scripts/sft_setup.py --config configs/sft_setup/setup.yml
uv run scripts/sft_train.py --config configs/sft_train/stage1.yml
```

### 5b. Multimodal SFT (audio & image input, optional)
Attach pretrained media encoders and fine-tune on part-structured
conversations (JSONL with `{"type": "audio"|"image", "path": ...}` parts --
see `configs/sft_train/multimodal.yml` for the format and the two-stage
frozen-tower recipe):
```bash
uv run scripts/sft_train.py --config configs/sft_train/multimodal.yml
```
The encoders follow the current de-facto standard and are both Apache-2.0:
[Whisper](https://huggingface.co/openai/whisper-small)'s encoder for audio
(Qwen2-Audio-style: pooled to ~10 soft tokens/sec, MLP-projected into the
token stream) and [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-256)'s
vision tower for images (LLaVA-style: 2x2 pixel-shuffled to 64 soft tokens,
MLP-projected). Both are implemented in-repo (no `transformers` dependency;
weights load from the Hub via `safetensors`, numerically verified against the
reference implementations) and ride along inside the SFT checkpoint, so
inference needs no Hub access. The resulting checkpoint serves multimodal
requests through the API server (OpenAI `input_audio` / `image_url` data-URI
content parts, see step 7).

### 6. Chat
An interactive chat TUI (built on [textual](https://textual.textualize.io/)):
streaming replies, multi-turn history, Tab-completed slash commands --
`/reset` (clear the conversation), `/system <text>`,
`/set temperature|top_k|top_p|max_new_tokens <value>`, `/theme <name>`,
`/help`, `/quit`; Esc stops a running generation. The status bar shows the
sampling settings and context-window usage.
```bash
uv run scripts/chat.py --checkpoint weights/sft-stage1/last.ckpt \
    --system "You are a helpful assistant."
```
By default the UI renders with the terminal's own 16-color ANSI palette
(`ansi-dark`); pass `--theme <name>` or switch live with `/theme` for a
true-color theme (nord, gruvbox, tokyo-night, ...).

### 7. Serve an OpenAI-compatible API
`GET /v1/models` and `POST /v1/chat/completions` (streaming or not), for
tools that speak the OpenAI Chat Completions format (e.g. OpenCode's
`@ai-sdk/openai-compatible` provider):
```bash
uv run scripts/api.py --checkpoint weights/sft-stage1/last.ckpt --port 8000
```
Requests are served one at a time (see `picochat/inference/api.py`); `--temperature`/
`--top-k`/`--top-p`/`--max-new-tokens` set the defaults, and a request may
override any of them.

### 8. Evaluate
Multiple-choice benchmarks scored by completion log-likelihood -- English
(hellaswag, arc_easy, arc_challenge, openbookqa, winogrande, boolq) and,
matching the CJK-tuned tokenizer, Japanese (jcommonsenseqa, belebele_ja,
xwinograd_ja):
```bash
# base checkpoints: plain text-continuation scoring
uv run scripts/base_eval.py --checkpoint weights/base/last.ckpt

# SFT checkpoints: items rendered as ChatML user turns (comparable numbers)
uv run scripts/base_eval.py --checkpoint weights/sft-stage1/last.ckpt --chat
```
`--limit N` caps the examples per task for a quick smoke run and
`--tasks a,b` selects a subset.

Generative pass@1 on verifiable code tasks -- the model writes code and the
reply is executed against each task's unit tests in the isolation sandbox.
This is what GRPO post-training optimizes, so run it on the checkpoints
before and after `grpo_train.py` (same JSONL task format) to measure what the
RL stage bought:
```bash
uv run scripts/code_eval.py --checkpoint weights/grpo/last.ckpt \
    --tasks configs/grpo/sample_tasks.jsonl
```
Decoding is greedy by default (deterministic; add `--temperature` etc. to
sample instead), `--limit N` caps the task count, and `--output results.json`
writes the per-task records.

### Tests
```bash
uv run pytest
```

## Project layout
The package is organized by domain (the project started as a nanochat-style
flat package, but has outgrown it), with `scripts/` holding one CLI per
pipeline step:

| Package / module | Responsibility |
|---|---|
| `picochat/model/` | the model. `blocks.py` (RMSNorm, SwiGLU, depth-attention residuals), `moe.py` (MoE + shareable ExpertBank), `linear_attn.py` (Gated DeltaNet-2), `sparse_attn.py` (Native Sparse Attention), `transformer.py` (`TransformerLayer` up through `TransformerLM`), `presets.py` (the scale ladder + `build_lm`), `estimate.py` (`estimate_num_params`), `grow.py` (width/depth/MoE-upcycle growth), `audio.py` (log-mel front end + audio encoders: pretrained Whisper or from-scratch), `vision.py` (SigLIP2 vision tower), `multimodal.py` (part-structured ChatML rendering + soft-token splicing) |
| `picochat/training/` | training. `modules.py` (the LightningModules: `GPT` for pretraining, `SFTModule` for SFT, and the shared `LMTrainerMixin`), `optim.py` (Muon/AdamW param split + LR schedule), `checkpoint.py` (loading checkpoints back into models), `kernels.py` (optional [HF `kernels`](https://github.com/huggingface/kernels) fused-loss integration, see below) |
| `picochat/data/` | data. `sources.py` (HF Hub streaming sources for pretraining text and SFT conversations), `dataloader.py` (sequence packing, the sharded on-disk token format, Datasets/samplers/DataModule), `multimodal.py` (part-structured JSONL SFT conversations with audio/image files) |
| `picochat/rl/` | RL post-training. `grpo.py` (rollouts, group advantages, the clipped-surrogate + KL loss), `reward.py` (verifiable rewards: test runner, LLM judge, multi-turn code-fixing env), `sandbox.py` (bubblewrap / hardened-subprocess isolation for the untrusted code rewards execute) |
| `picochat/evals/` | evaluation. `tasks.py` (likelihood-based multiple-choice benchmarks: hellaswag, arc, ...) |
| `picochat/inference/` | inference. `engine.py` (sampling, KV-cached streaming generation, speculative decoding, device/sampling CLI helpers), `api.py` (OpenAI-compatible Chat Completions endpoints) |
| `picochat/tokenizer.py` | BPE tokenizer (rustbpe training / tiktoken inference), special tokens, and the ChatML rendering built on them |
| `picochat/config.py` | config loading and the multi-device (linear-scaling) launch helpers shared across the training CLIs |
| `scripts/` | one CLI per pipeline step: `tok_train` → `base_setup` → `base_train` → `sft_setup` → `sft_train` → `grpo_train` → `base_eval`/`code_eval`/`chat`/`api` |

## Performance
Multi-GPU: pass `--devices N` to a training script to run DDP (add
`--num-nodes M` under a multi-node launcher to keep the scaling right).
Configs stay written for one GPU -- lr/max_steps/warmup_steps are
linear-scaled by the world size automatically -- while each rank draws its
own seeded IID sample stream (`seed` in the config, default 42), gradient
accumulation syncs gradients once per cycle instead of per microbatch, and
the MoE load-balancing bias follows the global batch's expert load. Sharded
strategies (`fsdp`, `deepspeed`) are rejected at launch: the trainers assume
replicated parameters (grad clipping, DDP `no_sync`, the MoE bias
all-reduce), and the default Muon optimizer needs whole 2D weight matrices,
which flat sharding breaks -- sharded training of the 8b+ presets remains
future work.

Training compiles the model with `torch.compile`. On Linux the base install
includes [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)'s
Triton kernels (`fla-core`), used on CUDA by *both* mixers: the Gated DeltaNet-2
layers (chunked GDN-2 rule, `fla.ops.gdn2`) and the Native Sparse Attention layers
(`parallel_nsa` for the fused compression/selection branches plus
`parallel_attn` for the sliding window) -- pure-PyTorch reference
implementations are the fallback and the CPU/test path. On top of that,
`trainer.fused_loss: true` in a stage config folds the lm-head matmul into
[Liger's](https://github.com/linkedin/Liger-Kernel) fused cross-entropy
kernel, loaded from the Hub via the optional
[HF `kernels`](https://github.com/huggingface/kernels) extra
([kernels-community/liger-kernels](https://huggingface.co/kernels-community/liger-kernels)).
At a 64k vocab the logits tensor is the largest activation of a training
step; never materializing it roughly halves peak memory (measured on the
a ≈0.5B model, batch 8 x 1024 tokens, bf16, L4 24GB: 13.1 → 6.8 GiB) at the
cost of some step time on smaller GPUs (+20% on that L4; the chunked kernel
re-reads the lm-head weight per chunk). Turn it on when memory-bound -- a
bigger model, longer context, or a batch that otherwise OOMs -- and leave it
off when raw throughput matters more. It is exactly loss-equivalent (same
values and gradients as the plain loss; verified in `tests/test_kernels.py`).

## Model Architecture
A decoder-only Transformer (pre-RMSNorm, no biases, untied embeddings) whose
sequence mixer is a **hybrid** stack: within each block of `layers_per_block`
layers, the first layers are [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)
linear-attention mixers and the block-tail layer is
[Native Sparse Attention](https://arxiv.org/abs/2502.11089) -- a 3:1 GDN:NSA
ratio at the default `layers_per_block: 4` (Qwen3-Next-style). Plus:
- [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791): a recurrent linear-
  attention mixer with a fixed-size state instead of a growing KV cache. It
  refines Gated DeltaNet / KDA by decoupling the single scalar gate into two
  independent channel-wise gates -- an erase gate on the key axis and a write
  gate on the value axis -- on top of KDA's channel-wise decay. It learns
  positions implicitly from its recurrence, so these layers use **no RoPE**.
  Chunkwise-parallel training and O(1)-per-token decode, with exact
  pure-PyTorch kernels and optional
  [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
  Triton kernels on CUDA.
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089): sparse softmax
  over three branches -- a compressed (mean-pooled blocks) branch, a top-n
  block-*selected* branch (its scores reuse the compressed branch, so selection
  is trained end-to-end; the sink/current/previous blocks are always kept), and
  a sliding window -- combined by a learned per-head gate. Runs on fla's Triton
  NSA kernels on CUDA (no O(T^2) score materialization; the GQA group per
  selection is 16, hence the presets' 16-query-head MQA / 32-head GQA-2 NSA
  configs). These layers keep **partial** [RoPE](https://arxiv.org/abs/2104.09864)
  (a fraction of each head's dims rotated) as their positional signal.
- [RMS Normalization](https://arxiv.org/abs/1910.07467)
- [QK Normalization](https://arxiv.org/abs/2010.04245) (L2-normalized q/k in GDN)
- [SwiGLU](https://arxiv.org/abs/2002.05202)
- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) (both mixers)
- [Mixture of Experts](https://arxiv.org/abs/2101.03961) with a
  [shared expert](https://arxiv.org/abs/2401.06066) and
  [DeepSeek-V3-style](https://arxiv.org/abs/2412.19437) sigmoid gating
  (the `-moe` presets)
- [MosaicBERT-style sequence packing](https://arxiv.org/abs/2312.17482):
  documents are greedy best-fit packed into fixed-length sequences, and
  attention never crosses document boundaries
- [Multi-token prediction](https://arxiv.org/abs/2404.19737) heads (`n_mtp`,
  the 8b+ presets): lightweight extra heads sharing the lm head
  (Medusa-style), trained as an auxiliary loss and reused for self-speculative
  decoding -- greedy generation (`generate()` at temperature 0) automatically
  drafts and verifies several tokens per forward, with a token stream
  identical to plain greedy decoding
- [Muon](https://kellerjordan.github.io/posts/muon/) optimizer for the hidden
  matrices, AdamW for embeddings/heads
- [Checkpoint averaging](https://arxiv.org/abs/2203.05482) ("model soup"):
  `scripts/avg_ckpts.py` uniformly averages several late-run checkpoints into
  one for a cheap generalization bump (also subsumes EMA)
- Long-context continual pretraining (`configs/base_train/longctx.yml`):
  warm-start from the base run and continue on longer rows -- the GDN-2 /
  NSA hybrid is built for it (fixed-size recurrent state + sparse attention)
- [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md)
  chat format
- Tool calling (Hermes / Qwen2.5-style function calling): tools declared to
  the model as JSON schema in the system prompt, calls emitted between
  `<|tool_call|>`/`<|/tool_call|>` special tokens and parsed back to OpenAI
  `tool_calls`; the API server accepts `tools` and `tool` result turns (see
  picochat/model/tools.py)
- Multimodal input (optional, via SFT): audio and image soft tokens spliced
  at placeholder positions (LLaVA / Qwen2-Audio / Qwen2-VL style), produced
  by a pretrained [Whisper](https://arxiv.org/abs/2212.04356) encoder
  (pooled + MLP-projected) and a
  [SigLIP2](https://arxiv.org/abs/2502.14786) vision tower (pixel-shuffled +
  MLP-projected), both Apache-2.0 and implemented in-repo

Presets (`configs/presets.yml`), named by nominal parameter count: `200m`
≈0.22B, `1b` ≈1.1B and `8b` ≈8.7B (dense; the GDN-2 channel-wise gates add a
few percent over the original name targets); at 35B/120B the ladder is
MoE-only, each in two variants matched on total params -- `35b-moe` ≈36.0B
total / 2.4B active and `120b-moe` ≈119B total / 6.2B active (fine-grained
LatentMoE, per-layer expert pools), `35b-moe-shared` ≈35.8B / 3.3B active and
`120b-moe-shared` ≈119B / 8.1B active (coarse-grained, one expert pool shared
across layers).

The dense rungs form a **growth chain**: they share a constant `d_head` of 64
and step `d_model` ×2 (1024 → 2048 → 4096), so a trained rung can *grow* into
the next instead of pretraining from scratch (`picochat.model.grow`) -- HyperCloning
widens it (replicating heads at a fixed head size, exactly function-preserving),
then whole blocks are stacked for depth. A config's `grow_from` points at the
smaller checkpoint; dense→MoE upcycling adds a zero-initialized routed branch to
each layer (also function-preserving) for warm-starting an MoE model.
