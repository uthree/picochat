# picochat
![logo](assets/logo.png)

A project inspired by [nanochat](https://github.com/karpathy/nanochat):
training a small chat LLM from scratch on a budget of roughly $100.

## Requirements
- Training: a CUDA GPU (CUDA 13.0 or later), PyTorch 2.12.1 or later
- Tests and evaluation also run on CPU

## Usage

### 1. Set up the environment
We use [uv](https://docs.astral.sh/uv/) for the virtual environment:
```bash
uv venv --python 3.14  # initialize venv
uv pip install -e .    # install dependencies
```

### 2. Train the tokenizer
Train a BPE tokenizer (128k vocab, ChatML special tokens) from a YAML recipe:
```bash
uv run scripts/tok_train.py --config configs/tok/major_langs.yml

# evaluate compression (bytes/token, higher = denser); 4-6 is fine
uv run scripts/tok_eval.py
```

### 3. Preprocess the pretraining data
Each stage's datasets are tokenized into sharded token binaries under `data/`:
```bash
uv run scripts/base_setup.py --config configs/base_setup/stage1.yml
uv run scripts/base_setup.py --config configs/base_setup/stage2.yml
uv run scripts/base_setup.py --config configs/base_setup/stage3.yml
```

### 4. Pretrain (3-stage curriculum)
The base model trains through a curriculum; stages 2 and 3 warm-start from
the previous stage's checkpoint (`init_from` in the config):
```bash
uv run scripts/base_train.py --config configs/base_train/stage1.yml  # simple stories
uv run scripts/base_train.py --config configs/base_train/stage2.yml  # textbooks + wikipedia
uv run scripts/base_train.py --config configs/base_train/stage3.yml  # multilingual web text
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
streaming replies, multi-turn history, and slash commands --
`/reset` (clear the conversation), `/system <text>`,
`/set temperature|top_k|top_p|max_new_tokens <value>`, `/theme <name>`,
`/help`, `/quit`; Esc stops a running generation.
```bash
uv run scripts/base_chat.py --checkpoint weights/sft-stage1/last.ckpt \
    --system "You are a helpful assistant."
```
The UI renders in true color by default; pass `--theme ansi-dark` (or
`ansi-light`, or switch live with `/theme`) to use the terminal's own
16-color ANSI palette instead.

### 7. Evaluate
Multiple-choice benchmarks (hellaswag, arc_easy, arc_challenge, openbookqa,
winogrande, boolq) scored by completion log-likelihood:
```bash
# base checkpoints: plain text-continuation scoring
uv run scripts/base_eval.py --checkpoint weights/stage3/last.ckpt

# SFT checkpoints: items rendered as ChatML user turns (comparable numbers)
uv run scripts/base_eval.py --checkpoint weights/sft-stage1/last.ckpt --chat
```
`--limit N` caps the examples per task for a quick smoke run and
`--tasks a,b` selects a subset.

### Tests
```bash
uv run pytest
```

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
  (`medium`/`large` presets)
- [MosaicBERT-style sequence packing](https://arxiv.org/abs/2312.17482):
  attention never crosses document boundaries
- [Muon](https://kellerjordan.github.io/posts/muon/) optimizer for the hidden
  matrices, AdamW for embeddings/heads
- [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md)
  chat format

Presets (`picochat/model/gpt.py`): `pico` ≈0.5B, `small` ≈1.0B and `base`
≈1.9B parameters (dense); `medium` ≈7.5B total / 2.6B active and `large`
≈23B total / 4.9B active (MoE).
