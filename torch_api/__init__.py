"""
Torch-compatible facade for Grilly: ``import grilly.torch_api as torch`` or use
re-exports on :mod:`grilly` (``device``, ``tensor``, ``save``, ``load``, …).

No PyTorch runtime; GPU path is Vulkan via ``grilly_core`` + SPIR-V. Checkpoints
use ``.grl`` (see :mod:`grilly.utils.grl_checkpoint`).
"""

from __future__ import annotations

import types

import grilly.nn as nn
import grilly.optim as optim
from grilly.functional.mf_activations import (
    mf_relu,
    mf_sigmoid,
    mf_sigmoid_01,
    mf_softmax,
    mf_softplus,
)

from .amp_mod import GradScaler, autocast
from .dtypes_and_device import (
    cpu,
    device,
    float16,
    float32,
    float64,
    int32,
    int64,
    long,
)
from .ops import (
    cdist,
    clamp,
    empty,
    no_grad,
    ones,
    randn,
    randperm,
    sign,
    tanh,
    tensor,
    zeros,
)
from .serialization import load, save
from .tensor import LongTensor, Tensor
from .vulkan_mod import is_available as _vulkan_is_available

amp = types.SimpleNamespace(
    autocast=autocast,
    GradScaler=GradScaler,
    __doc__="Automatic mixed precision (Vulkan-oriented; no CUDA).",
)


class _Vulkan:
    """``torch.cuda``-free replacement: ``torch.vulkan.is_available()``."""

    @staticmethod
    def is_available() -> bool:
        return _vulkan_is_available()


vulkan = _Vulkan()

__all__ = [
    "Tensor",
    "LongTensor",
    "device",
    "cpu",
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "long",
    "tensor",
    "empty",
    "zeros",
    "ones",
    "randn",
    "randperm",
    "cdist",
    "tanh",
    "clamp",
    "sign",
    "no_grad",
    "save",
    "load",
    "nn",
    "optim",
    "amp",
    "vulkan",
    "mf_softmax",
    "mf_softplus",
    "mf_sigmoid",
    "mf_sigmoid_01",
    "mf_relu",
]
