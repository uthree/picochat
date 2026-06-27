# picochat
Train and run inference on modern LLMs on a low budget.

## Model Architecture
We applied several improvements to a decoder-only Transformer, such as GPT.
- RoPE
- RMS Normalization
- QK Normalization
- SwiGLU
- Share attention parameters across layers
- Flash Attention
- Factorized Embedding
