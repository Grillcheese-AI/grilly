"""
Grilly - GPU-accelerated neural network operations using Vulkan.

PyTorch-like API for GPU-accelerated neural networks:
- nn: Neural network layers (Module, Linear, LayerNorm, Attention, etc.)
- functional: Functional API (activations, linear, attention, memory, etc.)
- optim: Optimizers (Adam, SGD, NLMS, NaturalGradient)
- utils: Utilities (data loading, checkpointing, device management)

This package provides GPU acceleration for:
- Spiking Neural Networks (SNN)
- Feedforward Neural Networks (FNN)
- Attention mechanisms
- Memory operations
- FAISS similarity search
- Place and time cells
- Learning operations (STDP, Hebbian, EWC, NLMS, Whitening)
- Bridge operations (continuous ↔ spike)
- Hippocampal transformer with capsule memory
"""

# grilly_core.<plat>.pyd lives in this directory as a sibling .pyd. Editable
# (PEP 660) installs route `import grilly` through a path hook that does NOT
# add this directory to sys.path, so `import grilly_core` would fail and the
# Vulkan probe in backend/base.py would silently report VULKAN_AVAILABLE=False
# even on machines with a perfectly working Vulkan device. Insert the package
# directory once, before any submodule import triggers the probe.
import os as _os
import sys as _sys

_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)

# Load the compiled core as a package submodule and register it under its
# top-level name, so `import grilly_core` works for any consumer that has
# imported grilly first — without copying the .pyd into site-packages.
# PyTorch ships torch._C exactly this way. This resolves for both editable
# installs (the .pyd sits in this source dir, on grilly.__path__) and real
# wheels (package-data bundles grilly_core.*.pyd inside the grilly package).
try:
    from . import grilly_core as _grilly_core
    _sys.modules.setdefault("grilly_core", _grilly_core)
except ImportError:
    pass  # core not built (CPU-only); the backend probe reports unavailable

import grilly.functional as functional
import grilly.nn as nn
import grilly.optim as optim
import grilly.utils as utils
from grilly.backend.base import (
    VULKAN_AVAILABLE,
    VULKAN_PYTHON_BINDINGS_AVAILABLE,
    VULKAN_PYTHON_LEGACY_BACKEND_AVAILABLE,
)
from grilly.backend.capsule_transformer import (
    CapsuleMemory,
    CapsuleTransformerConfig,
    CognitiveFeatures,
    MemoryType,
)
from grilly.backend.compute import VulkanCompute
from grilly.backend.learning import VulkanLearning
from grilly.backend.snn_compute import SNNCompute

from . import torch_api

# Main API exports
Compute = VulkanCompute
Learning = VulkanLearning

# Import utilities for HuggingFace and PyTorch compatibility
try:
    from grilly.utils.device_manager import get_cuda_backend, get_device_manager, get_vulkan_backend
    from grilly.utils.huggingface_bridge import HuggingFaceBridge, get_huggingface_bridge
    from grilly.utils.pytorch_compat import Tensor, ones, randn, tensor, zeros
    from grilly.utils.pytorch_ops import (
        add,
        avg_pool2d,
        conv2d,
        cross_entropy_loss,
        dropout,
        gelu,
        layer_norm,
        matmul,
        max_pool2d,
        mse_loss,
        mul,
        relu,
        softmax,
    )

    COMPAT_AVAILABLE = True
except Exception:
    COMPAT_AVAILABLE = False

__all__ = [
    "VULKAN_AVAILABLE",
    "VULKAN_PYTHON_BINDINGS_AVAILABLE",
    "VULKAN_PYTHON_LEGACY_BACKEND_AVAILABLE",
    "VulkanCompute",
    "Compute",
    "SNNCompute",
    "VulkanLearning",
    "Learning",
    "CapsuleMemory",
    "CapsuleTransformerConfig",
    "CognitiveFeatures",
    "MemoryType",
    # Submodules
    "nn",
    "functional",
    "optim",
    "utils",
    "torch_api",
]

# Conditionally add compatibility exports
if COMPAT_AVAILABLE:
    __all__.extend(
        [
            "get_device_manager",
            "get_vulkan_backend",
            "get_cuda_backend",
            "HuggingFaceBridge",
            "get_huggingface_bridge",
            "Tensor",
            "tensor",
            "zeros",
            "ones",
            "randn",
            "add",
            "mul",
            "matmul",
            "relu",
            "gelu",
            "softmax",
            "layer_norm",
            "dropout",
            "conv2d",
            "max_pool2d",
            "avg_pool2d",
            "mse_loss",
            "cross_entropy_loss",
        ]
    )

# Tensor conversion utilities
try:
    from grilly.utils.tensor_conversion import (
        VulkanTensor,
        ensure_vulkan_compatible,
        from_vulkan,
        to_vulkan,
        to_vulkan_batch,
        to_vulkan_gpu,
    )

    if "to_vulkan" not in __all__:
        __all__.extend(
            [
                "to_vulkan",
                "to_vulkan_batch",
                "to_vulkan_gpu",
                "from_vulkan",
                "ensure_vulkan_compatible",
                "VulkanTensor",
            ]
        )
except Exception:
    pass

# ONNX utilities
try:
    from grilly.utils.onnx_exporter import OnnxExporter
    from grilly.utils.onnx_finetune import OnnxFineTuner
    from grilly.utils.onnx_loader import GrillyOnnxModel, OnnxModelLoader

    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False

if ONNX_AVAILABLE:
    __all__.extend(
        [
            "OnnxModelLoader",
            "OnnxExporter",
            "GrillyOnnxModel",
            "OnnxFineTuner",
        ]
    )

try:
    from grilly._version import version as __version__
except ImportError:
    __version__ = "0.0.0.dev0"  # fallback when _version.py not yet generated
