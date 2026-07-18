# picochat
![logo](assets/logo.png)

A project inspired by [nanochat](https://github.com/karpathy/nanochat):
training a small chat LLM from scratch on a budget of roughly $100.

## Requirements
- Python 3.11
- Training: a CUDA GPU (CUDA 12.6 or later in the 12.x series), PyTorch 2.12.1 or later (cu126 wheels)
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

### 2. Train the tokenizer
Train a BPE tokenizer (128k vocab, ChatML special tokens) from a YAML recipe:
```bash
uv run scripts/tok_train.py --config configs/tok/default.yml

# evaluate compression (bytes/token, higher = denser); 4-6 is fine
uv run scripts/tok_eval.py
```

### 3. Preprocess the pretraining data
Each stage's datasets are tokenized, packed into fixed-length rows
(MosaicBERT-style sequence packing; each config's `block_size` must match its
training config) and written as sharded token binaries under `data/`:
```bash
uv run scripts/base_setup.py --config configs/base_setup/stage1.yml
uv run scripts/base_setup.py --config configs/base_setup/stage2.yml
```

### 4. Pretrain (2-stage)
Stage 1 is the single knowledge-oriented base corpus (code + textbooks +
wikipedia); stage 2 broadens to multilingual/web text and extends the context
length, warm-starting from stage 1's checkpoint (`init_from` in the config):
```bash
uv run scripts/base_train.py --config configs/base_train/stage1.yml  # knowledge base corpus
uv run scripts/base_train.py --config configs/base_train/stage2.yml  # multilingual, longer context
```
Interrupted stages resume automatically from `output_dir/last.ckpt`. Training
curves and generation samples are logged to TensorBoard (`lightning_logs/`).
If you hit out-of-memory errors, reduce `batch_size` (or raise `accumulate`)
in the stage config.

### 5. Supervised fine-tuning (SFT)
Preprocess the chat corpus, then fine-tune the final pretraining checkpoint:
```bash
uv run scripts/sft_setup.py --config configs/sft_setup/setup.yml
uv run scripts/sft_train.py --config configs/sft_train/stage1.yml
```

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
Requests are served one at a time (see `picochat/api.py`); `--temperature`/
`--top-k`/`--top-p`/`--max-new-tokens` set the defaults, and a request may
override any of them.

### 8. Evaluate
Multiple-choice benchmarks (hellaswag, arc_easy, arc_challenge, openbookqa,
winogrande, boolq) scored by completion log-likelihood:
```bash
# base checkpoints: plain text-continuation scoring
uv run scripts/base_eval.py --checkpoint weights/stage2/last.ckpt

# SFT checkpoints: items rendered as ChatML user turns (comparable numbers)
uv run scripts/base_eval.py --checkpoint weights/sft-stage1/last.ckpt --chat
```
`--limit N` caps the examples per task for a quick smoke run and
`--tasks a,b` selects a subset.

### Tests
```bash
uv run pytest
```

## Project layout
A flat package, one file per concern (following
[nanochat](https://github.com/karpathy/nanochat)):

| Module | Responsibility |
|---|---|
| `picochat/gpt.py` | the model: blocks (RMSNorm, RoPE, SwiGLU, MoE, attention) up through `TransformerLM` |
| `picochat/presets.py` | the scale-ladder presets (`configs/presets.yml`) and the `build_lm` factory |
| `picochat/param_estimate.py` | `estimate_num_params`: size a config without building the model |
| `picochat/trainer.py` | the LightningModules that train it (`GPT` for pretraining, `SFTModule` for SFT) and the shared Muon/AdamW + LR-schedule scaffolding |
| `picochat/grpo.py` | GRPO RL post-training: rollouts (single- and multi-turn/agentic), group advantages, the clipped-surrogate + KL loss |
| `picochat/reward.py` | verifiable rewards for GRPO: a test-runner backbone, an LLM judge, and the multi-turn code-fixing environment |
| `picochat/sandbox.py` | isolated (bubblewrap / hardened-subprocess) execution of the untrusted code GRPO rewards |
| `picochat/engine.py` | sampling, KV-cached streaming generation, and the shared device/sampling CLI helpers |
| `picochat/config.py` | config loading and the multi-device (linear-scaling) launch helpers shared across the training CLIs |
| `picochat/tokenizer.py` | BPE tokenizer (rustbpe training / tiktoken inference), special tokens, and the ChatML rendering built on them |
| `picochat/dataset.py` | where raw data comes from: HF Hub sources for pretraining text and SFT conversations |
| `picochat/dataloader.py` | sequence packing, the sharded on-disk token format, Datasets/samplers/DataModule |
| `picochat/tasks.py` | likelihood-based multiple-choice benchmarks (hellaswag, arc, ...) |
| `picochat/audio.py` | soft-token audio input path (Qwen-style, for multimodal experiments) |
| `picochat/kernels.py` | optional [HF `kernels`](https://github.com/huggingface/kernels) integration with plain-PyTorch fallback (see below) |
| `picochat/api.py` | OpenAI-compatible Chat Completions endpoints |
| `scripts/` | one CLI per pipeline step: `tok_train` → `base_setup` → `base_train` → `sft_setup` → `sft_train` → `grpo_train` → `base_eval`/`chat`/`api` |

## Performance
Training compiles the model with `torch.compile` (FlexAttention fuses the
sliding-window/packed-document attention). On top of that,
`trainer.fused_loss: true` in a stage config folds the lm-head matmul into
[Liger's](https://github.com/linkedin/Liger-Kernel) fused cross-entropy
kernel, loaded from the Hub via the optional
[HF `kernels`](https://github.com/huggingface/kernels) extra
([kernels-community/liger-kernels](https://huggingface.co/kernels-community/liger-kernels)).
At a 128k vocab the logits tensor is the largest activation of a training
step; never materializing it roughly halves peak memory (measured on the
`pico` preset, batch 8 x 1024 tokens, bf16, L4 24GB: 13.1 → 6.8 GiB) at the
cost of some step time on smaller GPUs (+20% on that L4; the chunked kernel
re-reads the lm-head weight per chunk). Turn it on when memory-bound -- a
bigger model, longer context, or a batch that otherwise OOMs -- and leave it
off when raw throughput matters more. It is exactly loss-equivalent (same
values and gradients as the plain loss; verified in `tests/test_kernels.py`).

## Model Architecture
A decoder-only Transformer (pre-RMSNorm, no biases, untied embeddings) with
several improvements:
- [RoPE](https://arxiv.org/abs/2104.09864)
- [RMS Normalization](https://arxiv.org/abs/1910.07467)
- [QK Normalization](https://arxiv.org/abs/2010.04245)
- [SwiGLU](https://arxiv.org/abs/2002.05202)
- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245)
- [Sliding Window Attention](https://arxiv.org/abs/2502.18845) with periodic
  global layers, via
  [FlexAttention](https://pytorch.org/blog/flexattention/) on CUDA
- [Mixture of Experts](https://arxiv.org/abs/2101.03961) with a
  [shared expert](https://arxiv.org/abs/2401.06066) and
  [DeepSeek-V3-style](https://arxiv.org/abs/2412.19437) sigmoid gating
  (the `-moe` presets)
- [MosaicBERT-style sequence packing](https://arxiv.org/abs/2312.17482):
  documents are greedy best-fit packed into fixed-length sequences, and
  attention never crosses document boundaries
- [Muon](https://kellerjordan.github.io/posts/muon/) optimizer for the hidden
  matrices, AdamW for embeddings/heads
- [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md)
  chat format

Presets (`configs/presets.yml`): `pico` ≈0.5B, `small` ≈1.0B and `base`
≈1.9B parameters (dense); `medium` ≈7.5B total / 2.6B active and `large`
≈23B total / 4.9B active (MoE).
