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
from .capsule_transformer import (
    CapsuleMemory,
    CapsuleTransformerConfig,
    CognitiveFeatures,
    MemoryType,
)
from .compute import VulkanCompute
from .learning import VulkanLearning
from .lora import VulkanLoRA
from .snn_compute import SNNCompute

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

