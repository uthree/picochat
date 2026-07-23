"""The picochat model: building blocks, the two sequence mixers, MoE, the
assembled TransformerLM, the scale-ladder presets, parameter estimation and
growth/upcycling. The names re-exported here are the package's public model
API; the submodules hold the implementations:

- blocks.py       rms_norm / SwiGLU / DepthAttention / packing boundaries
- linear_attn.py  Gated DeltaNet-2 (the linear-attention mixer)
- sparse_attn.py  Native Sparse Attention (the softmax mixer)
- moe.py          Mixture of Experts + the shareable ExpertBank
- transformer.py  TransformerLayer / Transformer / MTPHead / TransformerLM
- presets.py      the scale ladder (configs/presets.yml) + build_lm
- estimate.py     estimate_num_params (no model build required)
- grow.py         width/depth/MoE-upcycle growth transforms
- audio.py        log-mel front end + audio encoders (pretrained Whisper /
                  from-scratch)
- vision.py       SigLIP2 vision tower + projector
- multimodal.py   part-structured ChatML rendering + soft-token splicing
"""

from picochat.model.blocks import (
    DepthAttention,
    SwiGLU,
    doc_ids_to_cu_seqlens,
    rms_norm,
)
from picochat.model.estimate import estimate_num_params
from picochat.model.linear_attn import GatedDeltaNet2
from picochat.model.moe import ExpertBank, MixtureOfExperts, moe_modules
from picochat.model.presets import MODEL_PRESETS, build_lm, estimate_preset_params
from picochat.model.sparse_attn import NativeSparseAttention
from picochat.model.transformer import (
    MTPHead,
    Transformer,
    TransformerLayer,
    TransformerLM,
)

__all__ = [
    "DepthAttention",
    "ExpertBank",
    "GatedDeltaNet2",
    "MODEL_PRESETS",
    "MTPHead",
    "MixtureOfExperts",
    "NativeSparseAttention",
    "SwiGLU",
    "Transformer",
    "TransformerLayer",
    "TransformerLM",
    "build_lm",
    "doc_ids_to_cu_seqlens",
    "estimate_num_params",
    "estimate_preset_params",
    "moe_modules",
    "rms_norm",
]
