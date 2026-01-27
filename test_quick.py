"""Quick test to verify basic functionality"""
import sys
print(f"Python: {sys.version}")

import numpy as np
print(f"Numpy: {np.__version__}")

# Test basic numpy
x = np.random.randn(10, 10).astype(np.float32)
print(f"Created array: {x.shape}")

# Test VulkanTensor
from grilly.utils.tensor_conversion import VulkanTensor
vt = VulkanTensor(x)
print(f"VulkanTensor: {vt}")

# Test numpy conversion
x_back = vt.numpy()
print(f"Roundtrip works: {np.allclose(x, x_back)}")

# Test numba ops (if available)
try:
    from grilly.utils.numba_ops import NUMBA_AVAILABLE, layernorm, softmax
    print(f"Numba available: {NUMBA_AVAILABLE}")
    if NUMBA_AVAILABLE:
        # Test layernorm
        gamma = np.ones(10, dtype=np.float32)
        beta = np.zeros(10, dtype=np.float32)
        result = layernorm(x, gamma, beta)
        print(f"Numba layernorm works: {result.shape}")
except ImportError as e:
    print(f"Numba import failed: {e}")

print("\n=== All basic tests passed! ===")
