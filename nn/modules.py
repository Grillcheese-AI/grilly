"""
nn/modules.py — backward-compatibility re-exports.

All class implementations have been split into focused files:
  nn/linear.py            — Linear
  nn/normalization_modules.py — LayerNorm, RMSNorm
  nn/activations.py       — ReLU, GELU, SiLU, GCU, RoSwish, SwiGLU, Softmax, Softplus
  nn/dropout.py           — Dropout
  nn/embedding.py         — Embedding
  nn/containers.py        — Sequential, Residual
  nn/attention.py         — MultiheadAttention, FlashAttention2

Existing code that does `from nn.modules import Linear` (or any other name) still works.
"""

# Also re-export shared helpers so any code doing
# `from nn.modules import _get_param_array` keeps working.
from ._helpers import _bridge_to_numpy, _create_param_wrapper, _get_param_array
from .activations import GCU, GELU, ReLU, RoSwish, SiLU, Softmax, Softplus, SwiGLU
from .attention import (
    FlashAttention2,
    FNetMixing,
    HYLAAttention,
    MultiheadAttention,
    SympFormerBlock,
)

# _FUSED_ACTIVATION_MAP was previously a module-level dict in modules.py
from .containers import Residual, Sequential
from .dropout import Dropout
from .embedding import Embedding
from .linear import Linear
from .normalization_modules import LayerNorm, RMSNorm

__all__ = [
    "Linear",
    "LayerNorm",
    "RMSNorm",
    "ReLU",
    "GELU",
    "SiLU",
    "GCU",
    "RoSwish",
    "SwiGLU",
    "Softmax",
    "Softplus",
    "Dropout",
    "Embedding",
    "Sequential",
    "Residual",
    "MultiheadAttention",
    "FlashAttention2",
    "FNetMixing",
    "HYLAAttention",
    "SympFormerBlock",
    "_bridge_to_numpy", "_create_param_wrapper", "_get_param_array"
]
