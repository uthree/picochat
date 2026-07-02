# picochat
A project inspired by nanochat.
It involves training a sub-billion-parameter language model from scratch for $100.

## Model Architecture
We applied several improvements to a decoder-only Transformer, such as GPT-2.
- RoPE
- RMS Normalization
- QK Normalization
- SwiGLU
- Flash Attention
- Sliding Widnow Attention
- Looped Transformer
