# GPU-First Design

Version 0.5.0 introduced the "GPU-First" architecture: a C++ Tensor with dual-validity tracking that keeps data GPU-resident between operations.

---

## The Problem: CPU-GPU Ping-Pong

In early versions of grilly, every operation followed this pattern:

1. Upload numpy array to GPU (vkMapMemory, memcpy, vkUnmapMemory)
2. Run compute shader
3. Download result to CPU (vkMapMemory, memcpy, vkUnmapMemory)
4. Repeat for next operation

For a model with 20 operations per forward pass, that is 40 GPU memory map/unmap cycles. The map/unmap overhead dominated actual compute time.

---

## The Fix: C++ Tensor with Dual Validity

The C++ backend introduces a `Tensor` type that tracks where valid data lives:

- **CPU-valid**: data is current in the numpy array
- **GPU-valid**: data is current in the Vulkan buffer
- **Both-valid**: data is synchronized

When you chain operations, data stays GPU-resident:

```python
# Only one upload, one download -- data stays on GPU between ops
h = F.linear(x, w1, b1)     # Upload x, w1, b1. Result stays on GPU.
h = F.relu(h)                # No transfer. GPU reads h, writes h.
h = F.linear(h, w2, b2)     # Upload w2, b2. h stays on GPU.
out = F.softmax(h, dim=-1)   # No transfer. Download only when accessed.
```

---

## VMA Persistent Mapping

The C++ backend uses Vulkan Memory Allocator (VMA) with persistent mapping. GPU buffers are mapped once at allocation and stay mapped for the lifetime of the buffer. This eliminates per-operation `vkMapMemory`/`vkUnmapMemory` calls.

Combined with `BufferPool` bucketed allocation, buffer reuse minimizes allocation overhead.

---

## Bridge Architecture

`backend/_bridge.py` is the routing layer. It lazily initializes a `grilly_core.Device` singleton and routes ops through C++:

```python
# _bridge.py routes to C++ when available
def linear(x, weight, bias=None):
    dev = _get_device()
    if dev is None:
        return None  # Caller falls back to Python path
    x = _ensure_f32_contiguous(x)
    weight = _ensure_f32_contiguous(weight)
    return dev.linear(x, weight, bias)
```

Higher-level modules call bridge functions with a try/fallback pattern. If the C++ backend returns a result, use it. Otherwise, fall through to the pure Python Vulkan path.

---

## VulkanTensor

`VulkanTensor` wraps a GPU buffer for explicit zero-copy access:

```python
from grilly.utils.tensor_conversion import VulkanTensor, to_vulkan, from_vulkan

# Upload to GPU
vt = to_vulkan(numpy_array)

# Use in operations (stays on GPU)
result_vt = F.linear(vt, weight_vt)

# Download when needed
result_np = from_vulkan(result_vt)
```

!!! note
    Conv2d's GEMM path currently downloads for CPU transpose. A GPU transpose kernel is planned.

---

## What Changed in 0.5.0

| Before (0.4.x) | After (0.5.0+) |
|---|---|
| Python ctypes Vulkan dispatch | C++ pybind11 dispatch |
| Map/unmap per operation | VMA persistent mapping |
| Allocate per operation | BufferPool bucketed reuse |
| CPU-GPU round-trip per op | Data stays GPU-resident |
| Single `bindings.cpp` | 11 focused binding files |
| No JIT | `@grilly.jit` trace compilation |
| No AMP | `autocast()` + `GradScaler` |
