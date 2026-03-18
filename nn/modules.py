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

from .linear import Linear
from .normalization_modules import LayerNorm, RMSNorm
from .activations import ReLU, GELU, SiLU, GCU, RoSwish, SwiGLU, Softmax, Softplus
from .dropout import Dropout
from .embedding import Embedding
from .containers import Sequential, Residual
from .attention import MultiheadAttention, FlashAttention2, FNetMixing, HYLAAttention, SympFormerBlock

# Also re-export shared helpers so any code doing
# `from nn.modules import _get_param_array` keeps working.
from ._helpers import (
    _PARAMETER_AVAILABLE,
    ParameterClass,
    _USE_CPP_BRIDGE,
    _bridge,
    _bridge_to_numpy,
    _get_param_array,
    _create_param_wrapper,
)

# _FUSED_ACTIVATION_MAP was previously a module-level dict in modules.py
from .containers import _FUSED_ACTIVATION_MAP

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
]
