"""
Grilly Quick Start Tutorial
============================

Basic usage of GPU-accelerated neural network operations on AMD/NVIDIA/Intel GPUs.
"""

import _path_setup  # noqa: F401 - must be first to ensure grilly is importable

import numpy as np
from grilly import Compute

# Initialize Vulkan compute backend
compute = Compute()
print(f"GPU: {compute.core.device_properties.deviceName}")

# 1. Activations
x = np.random.randn(4, 8).astype(np.float32)

relu_out = compute.fnn.activation_relu(x)
gelu_out = compute.fnn.activation_gelu(x)
softmax_out = compute.fnn.activation_softmax(x)

print(f"ReLU output shape: {relu_out.shape}")
print(f"GELU output shape: {gelu_out.shape}")
print(f"Softmax sum (should be ~1): {softmax_out.sum(axis=-1)}")

# 2. Linear layer
x = np.random.randn(2, 64).astype(np.float32)
weights = np.random.randn(128, 64).astype(np.float32)
bias = np.random.randn(128).astype(np.float32)

linear_out = compute.fnn.linear(x, weights, bias)
print(f"Linear output shape: {linear_out.shape}")

# 3. Layer normalization
x = np.random.randn(2, 64).astype(np.float32)
gamma = np.ones(64, dtype=np.float32)
beta = np.zeros(64, dtype=np.float32)

norm_out = compute.fnn.layernorm(x, gamma, beta)
print(f"LayerNorm output shape: {norm_out.shape}")
print(f"LayerNorm mean (should be ~0): {norm_out.mean():.4f}")
print(f"LayerNorm std (should be ~1): {norm_out.std():.4f}")

print("\nAll operations completed successfully!")
