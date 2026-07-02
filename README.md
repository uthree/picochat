# picochat
!["logo"]("assets/logo.png")  

A project inspired by [nanochat](https://github.com/karpathy/nanochat).
It involves training a sub-billion-parameter language model from scratch for $100.

## Requirements
- CUDA 13.0 or later, pytorch 2.12.1 or later

## Usage

### full-scratch training
1. setup environment.
We will set up a virtual environment using uv.
```bash
uv venv --python 3.14 # initialize venv.
uv pip install -e . # install dependencies.
```
2. train tokenizer.
Train a BPE tokenizer to split text into tokens.
```bash
# Currently, only mixed Japanese and English data is supported.
uv run scripts/tok_train.py --config configs/tok/en_ja.yml
```

3. preprocess dataset.
Preprocessing begins.
The dataset is tokenized using a BPE tokenizer, and the token IDs are saved to a binary file.
You can change the dataset or model size used for training by modifying the configuration file.  
```bash
uv run scripts/base_setup.py --config configs/base_setup/stage1_basic.yml
```

4. train language model (pretraining phase)
Training of the language model will now begin. If an "out of memory" error occurs, please edit the files in `configs/base_train/*.yml` to reduce the batch size or make other adjustments.
```bash
uv run scripts/base_train.py --config configs/base_train/base_setup.yml
```

## Model Architecture
We applied several improvements to a decoder-only Transformer, such as GPT-2.
- RoPE
- RMS Normalization
- QK Normalization
- SwiGLU
- Flash Attention
- Sliding Widnow Attention
- Looped Transformer
