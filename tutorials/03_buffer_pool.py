"""
Buffer Pool Tutorial
====================

GPU buffer pooling for efficient memory reuse across operations.
The buffer pool reduces allocation overhead by reusing buffers.

Note: VMA (Vulkan Memory Allocator) integration is available but disabled
by default for stability. The legacy BufferPool works reliably on all GPUs.
"""

import _path_setup  # noqa: F401 - must be first to ensure grilly is importable

import numpy as np
from grilly import Compute
from grilly.backend.buffer_pool import is_vma_available

compute = Compute()
print(f"GPU: {compute.core.device_properties.deviceName}")
print(f"VMA Available: {is_vma_available()} (disabled by default for stability)")

# Run some operations to initialize the buffer pool
x = np.random.randn(64, 256).astype(np.float32)
compute.fnn.activation_relu(x)  # This initializes the FNN's buffer pool

# Get the buffer pool from the FNN instance
pool = compute.fnn.buffer_pool
print(f"Pool type: {type(pool).__name__}")
print(f"Initial stats: {pool}")

# Run operations - buffers are automatically pooled
for i in range(20):
    compute.fnn.activation_relu(x)
    compute.fnn.activation_gelu(x)

print(f"\nAfter 40 operations: {pool}")
stats = pool.get_stats()
print(f"  Hit rate: {stats['hit_rate']:.1%}")
print(f"  Allocations: {stats['allocations']}")

# Larger operations
x = np.random.randn(128, 512).astype(np.float32)
for i in range(20):
    compute.fnn.activation_softmax(x)

print(f"\nAfter mixed sizes: {pool}")
stats = pool.get_stats()
print(f"  Hit rate: {stats['hit_rate']:.1%}")
print(f"  Total acquired: {stats['total_acquired']}")

# Buffer pool benefits:
print("\nBuffer Pool Benefits:")
print("  - Reduces GPU memory allocation overhead")
print("  - High hit rate after warmup (buffers reused)")
print("  - Size bucketing for efficient reuse")
print("  - Per-instance pools avoid cross-session issues")
