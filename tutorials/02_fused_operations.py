"""
Fused Operations Tutorial
=========================

Fused shaders combine multiple operations into single GPU dispatch for better performance.
"""

import _path_setup  # noqa: F401 - must be first to ensure grilly is importable

import numpy as np
import time
from grilly import Compute

compute = Compute()
print(f"GPU: {compute.core.device_properties.deviceName}")

# Test data
x = np.random.randn(32, 256).astype(np.float32)
w = np.random.randn(512, 256).astype(np.float32)
b = np.random.randn(512).astype(np.float32)

# Warmup
for _ in range(5):
    compute.fnn.linear(x, w, b)

iterations = 100

# Separate operations: Linear -> GELU
start = time.perf_counter()
for _ in range(iterations):
    out = compute.fnn.linear(x, w, b)
    out = compute.fnn.activation_gelu(out)
separate_time = (time.perf_counter() - start) / iterations * 1000

# Fused operation: Linear+GELU in one dispatch
start = time.perf_counter()
for _ in range(iterations):
    out = compute.fnn.fused_linear_gelu(x, w, b)
fused_time = (time.perf_counter() - start) / iterations * 1000

speedup = separate_time / fused_time
print(f"\nLinear+GELU Benchmark (32x256 -> 512):")
print(f"  Separate: {separate_time:.2f}ms")
print(f"  Fused:    {fused_time:.2f}ms")
print(f"  Speedup:  {speedup:.2f}x")

# Other fused operations
print("\nAvailable fused operations:")
print("  - fused_linear_gelu(x, w, b)  # For BERT, GPT")
print("  - fused_linear_relu(x, w, b)  # Classic FFN")
print("  - fused_linear_silu(x, w, b)  # For LLaMA, Mistral")
