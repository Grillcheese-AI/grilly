"""
Vulkan compute backend module for GPU-accelerated neural network operations.

This module provides GPU acceleration for:
- Spiking Neural Networks (SNN)
- Feedforward Neural Networks (FNN)
- Attention mechanisms
- Memory operations
- FAISS similarity search
- Place and time cells
- Learning operations (STDP, Hebbian, EWC, NLMS, Whitening)
- Bridge operations (continuous ↔ spike)
"""

from .base import VULKAN_AVAILABLE
from .compute import VulkanCompute
from .snn_compute import SNNCompute
from .learning import VulkanLearning
from .capsule_transformer import (
    CapsuleMemory,
    CapsuleTransformerConfig,
    CognitiveFeatures,
    MemoryType,
)

__all__ = [
    'VULKAN_AVAILABLE',
    'VulkanCompute',
    'SNNCompute',
    'VulkanLearning',
    'CapsuleMemory',
    'CapsuleTransformerConfig',
    'CognitiveFeatures',
    'MemoryType',
]

