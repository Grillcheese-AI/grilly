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

from grilly.backend.base import VULKAN_AVAILABLE
from grilly.backend.compute import VulkanCompute
from grilly.backend.snn_compute import SNNCompute
from grilly.backend.learning import VulkanLearning
from grilly.backend.capsule_transformer import (
    CapsuleMemory,
    CapsuleTransformerConfig,
    CognitiveFeatures,
    MemoryType,
)

# Main API exports
Compute = VulkanCompute
Learning = VulkanLearning

# Import submodules for easy access
import grilly.nn as nn
import grilly.functional as functional
import grilly.optim as optim
import grilly.utils as utils

# Import utilities for HuggingFace and PyTorch compatibility
try:
    from grilly.utils.device_manager import get_device_manager, get_vulkan_backend, get_cuda_backend
    from grilly.utils.huggingface_bridge import HuggingFaceBridge, get_huggingface_bridge
    from grilly.utils.pytorch_compat import Tensor, tensor, zeros, ones, randn
    from grilly.utils.pytorch_ops import (
        add, mul, matmul, relu, gelu, softmax, layer_norm, dropout,
        conv2d, max_pool2d, avg_pool2d, mse_loss, cross_entropy_loss
    )
    COMPAT_AVAILABLE = True
except Exception:
    COMPAT_AVAILABLE = False

__all__ = [
    'VULKAN_AVAILABLE',
    'VulkanCompute',
    'Compute',
    'SNNCompute',
    'VulkanLearning',
    'Learning',
    'CapsuleMemory',
    'CapsuleTransformerConfig',
    'CognitiveFeatures',
    'MemoryType',
    # Submodules
    'nn',
    'functional',
    'optim',
    'utils',
]

# Conditionally add compatibility exports
if COMPAT_AVAILABLE:
    __all__.extend([
        'get_device_manager',
        'get_vulkan_backend',
        'get_cuda_backend',
        'HuggingFaceBridge',
        'get_huggingface_bridge',
        'Tensor',
        'tensor',
        'zeros',
        'ones',
        'randn',
        'add',
        'mul',
        'matmul',
        'relu',
        'gelu',
        'softmax',
        'layer_norm',
        'dropout',
        'conv2d',
        'max_pool2d',
        'avg_pool2d',
        'mse_loss',
        'cross_entropy_loss',
    ])

# Tensor conversion utilities
try:
    from grilly.utils.tensor_conversion import (
        to_vulkan, to_vulkan_batch, to_vulkan_gpu, from_vulkan,
        ensure_vulkan_compatible, VulkanTensor
    )
    if 'to_vulkan' not in __all__:
        __all__.extend([
            'to_vulkan',
            'to_vulkan_batch',
            'to_vulkan_gpu',
            'from_vulkan',
            'ensure_vulkan_compatible',
            'VulkanTensor',
        ])
except Exception:
    pass

__version__ = '0.1.0'
