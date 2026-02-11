"""
Example: GPU-Accelerated VSA Operations

Demonstrates Vulkan-accelerated VSA operations with CPU fallback.
"""

import numpy as np

try:
    from grilly.backend.base import VULKAN_AVAILABLE
    from grilly.backend.core import VulkanCore
    from grilly.backend.experimental.vsa import VulkanVSA

    if not VULKAN_AVAILABLE:
        raise ImportError("Vulkan not available")
except ImportError:
    print("Vulkan backend not available. Skipping GPU examples.")
    print("Install Vulkan SDK and ensure shaders are compiled.")
    exit(0)

from grilly.experimental.vsa import BinaryOps, HolographicOps

print("=" * 60)
print("GPU-Accelerated VSA Operations")
print("=" * 60)

# Initialize GPU backend
print("\n1. Initialization")
print("-" * 60)

core = VulkanCore()
vsa = VulkanVSA(core)

print("VulkanCore initialized")
print("VulkanVSA initialized")

available_shaders = [k for k in core.shaders.keys() if k.startswith("vsa-")]
print(f"Available VSA shaders: {available_shaders}")

# Bipolar Binding
print("\n2. Bipolar Binding (GPU)")
print("-" * 60)

dim = 4096
a = BinaryOps.random_bipolar(dim).astype(np.float32)
b = BinaryOps.random_bipolar(dim).astype(np.float32)

print(f"Input vectors: shape={a.shape}, dtype={a.dtype}")

gpu_result = vsa.bind_bipolar(a, b)
print(f"GPU result: shape={gpu_result.shape}")

cpu_result = BinaryOps.bind(a, b)
print(f"CPU result: shape={cpu_result.shape}")

match = np.allclose(gpu_result, cpu_result, atol=1e-5)
print(f"GPU matches CPU: {match}")

# Bundling
print("\n3. Bundling (GPU)")
print("-" * 60)

vectors = [BinaryOps.random_bipolar(dim).astype(np.float32) for _ in range(5)]

gpu_bundled = vsa.bundle(vectors)
cpu_bundled = BinaryOps.bundle(vectors)

print(f"GPU bundled: shape={gpu_bundled.shape}")
print(f"CPU bundled: shape={cpu_bundled.shape}")

match_bundle = np.allclose(gpu_bundled, cpu_bundled, atol=1e-4)
print(f"GPU matches CPU: {match_bundle}")

# Similarity Batch
print("\n4. Batch Similarity (GPU)")
print("-" * 60)

query = HolographicOps.random_vector(dim).astype(np.float32)
codebook = [HolographicOps.random_vector(dim).astype(np.float32) for _ in range(100)]

gpu_similarities = vsa.similarity_batch(query, codebook)
print(f"GPU similarities: shape={gpu_similarities.shape}")

cpu_similarities = np.array([HolographicOps.similarity(query, vec) for vec in codebook])

print(f"CPU similarities: shape={cpu_similarities.shape}")

match_sim = np.allclose(gpu_similarities, cpu_similarities, atol=1e-4)
print(f"GPU matches CPU: {match_sim}")

if match_sim:
    top_k = 3
    top_indices = np.argsort(gpu_similarities)[-top_k:][::-1]
    print(f"Top-{top_k} most similar vectors: {top_indices}")

# Circular Convolution
print("\n5. Circular Convolution (GPU)")
print("-" * 60)

vec1 = HolographicOps.random_vector(dim).astype(np.float32)
vec2 = HolographicOps.random_vector(dim).astype(np.float32)

gpu_conv = vsa.circular_convolve(vec1, vec2)
cpu_conv = HolographicOps.convolve(vec1, vec2)

print(f"GPU convolution: shape={gpu_conv.shape}")
print(f"CPU convolution: shape={cpu_conv.shape}")

match_conv = np.allclose(gpu_conv, cpu_conv, atol=1e-4)
print(f"GPU matches CPU: {match_conv}")

# Performance comparison
print("\n6. Performance Comparison")
print("-" * 60)

import time

large_dim = 8192
large_a = BinaryOps.random_bipolar(large_dim).astype(np.float32)
large_b = BinaryOps.random_bipolar(large_dim).astype(np.float32)

# GPU timing
start = time.time()
for _ in range(10):
    _ = vsa.bind_bipolar(large_a, large_b)
gpu_time = (time.time() - start) / 10

# CPU timing
start = time.time()
for _ in range(10):
    _ = BinaryOps.bind(large_a, large_b)
cpu_time = (time.time() - start) / 10

print(f"Dimension: {large_dim}")
print(f"GPU time (avg): {gpu_time * 1000:.2f} ms")
print(f"CPU time (avg): {cpu_time * 1000:.2f} ms")
print(f"Speedup: {cpu_time / gpu_time:.2f}x")
