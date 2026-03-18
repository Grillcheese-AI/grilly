# Grilly GPU-First Architecture — Design Spec

**Date:** 2026-03-18
**Author:** grillcheese + Claude
**Project:** grilly (GPU-accelerated neural network framework)

## Summary

Eliminate CPU↔GPU ping-pong by making `VulkanTensor` the standard tensor type (like `torch.Tensor`). All computation stays on GPU via C++ backend buffer handles. Data touches CPU exactly twice: initial upload, final download.

### Objectives

1. Optimize grilly for maximum GPU performance — close gap with PyTorch benchmarks
2. Achieve 1:1 PyTorch API compatibility for common nn/optim/functional operations
3. Surpass PyTorch in areas where Vulkan has advantages (cross-platform, multi-vendor GPU)
4. All computation on GPU via Vulkan/C++ backend — zero numpy in hot paths
5. Fallback: if no Vulkan GPU detected at runtime, error with clear message
6. Link all Python code to C++ backend — no standalone Python compute paths
7. Eliminate CPU↔GPU ping-pong — data stays in Vulkan buffers between ops

### Non-Goals

- Computation graph / JIT compilation (future 0.3.0)
- Python-side Vulkan ctypes path (being replaced entirely by C++ backend)
- Numba fallback (removed — C++ backend is always available when grilly is installed)

---

## The Problem

Current architecture does CPU↔GPU transfer at EVERY operation boundary:

```
Python (numpy) → _bridge.py → bindings.cpp → upload to GPU → compute → download to numpy
                                                                              ↓
    next op: numpy → _bridge.py → bindings.cpp → upload AGAIN → compute → download AGAIN
```

A simple `linear → relu → layernorm` chain does 3 upload/download cycles. This dominates runtime for small-to-medium tensors and wastes PCIe bandwidth for large ones.

### Root Cause

`bindings.cpp` (4,221 lines, single file) converts every input from `py::array_t<float>` (numpy) and every output back to `py::array_t<float>`. There's no concept of a GPU buffer handle that Python can hold and pass to the next op.

The C++ backend already has `BufferPool`, `CommandBatch`, and `PipelineCache` for efficient GPU execution — but the Python boundary throws away all that work by downloading after every op.

---

## Section 1: GpuBuffer — C++ Primitive

A lightweight C++ struct exposed to Python via pybind11. Represents a Vulkan buffer allocation on GPU.

```cpp
// cpp/include/grilly/gpu_buffer.h
struct GpuBuffer {
    VmaAllocation allocation;
    VkBuffer handle;
    size_t byte_size;
    std::vector<int64_t> shape;
    VkDeviceMemory memory;  // for mapping if needed
};
```

**Python sees it as opaque.** You can:
- Pass it to any C++ op (zero-copy)
- Call `.to_numpy()` to download
- Check `.shape`, `.byte_size`

Buffer lifecycle managed by the C++ `BufferPool` — when Python wrapper is garbage collected, buffer returns to pool for reuse.

### Universal Converter

Every binding function uses `to_gpu_buffer()` to accept either numpy or `GpuBuffer`:

```cpp
// bindings_core.h
GpuBuffer* to_gpu_buffer(py::object input, BufferPool& pool) {
    if (py::isinstance<GpuBuffer>(input)) {
        return input.cast<GpuBuffer*>();  // already on GPU — zero copy
    }
    auto arr = input.cast<py::array_t<float>>();
    auto contiguous = py::array_t<float>::ensure(arr, py::array::c_style);
    return pool.upload(contiguous.data(), contiguous.size() * sizeof(float), shape_from(contiguous));
}
```

---

## Section 2: VulkanTensor — Python Standard Tensor

Refactored from "numpy wrapper with optional GPU" to "GPU-first tensor that exports to numpy on demand." Like `torch.Tensor`.

```python
class VulkanTensor:
    def __init__(self, data=None, gpu_buffer=None):
        if gpu_buffer is not None:
            self._gpu = gpu_buffer
            self._cpu = None
        elif data is not None:
            self._cpu = np.asarray(data, dtype=np.float32)
            self._gpu = _bridge.upload(self._cpu)

    @property
    def shape(self): return tuple(self._gpu.shape)

    def numpy(self) -> np.ndarray:
        """Explicit download — the ONLY way data leaves the GPU."""
        if self._cpu is None:
            self._cpu = self._gpu.to_numpy()
        return self._cpu

    def cpu(self) -> np.ndarray:
        return self.numpy()

    def item(self) -> float:
        return float(self.numpy().ravel()[0])

    def __array__(self, dtype=None):
        """numpy interop — allows np.array(tensor) during transition."""
        arr = self.numpy()
        return arr if dtype is None else arr.astype(dtype)

    def __matmul__(self, other):
        return VulkanTensor(gpu_buffer=_bridge.matmul(self._gpu, other._gpu))

    def __add__(self, other):
        return VulkanTensor(gpu_buffer=_bridge.add(self._gpu, other._gpu))

    def __del__(self):
        # GpuBuffer returned to pool by C++ destructor
        pass
```

**Rules:**
- `VulkanTensor(numpy_array)` uploads once on creation
- All ops return `VulkanTensor` with `GpuBuffer` — never numpy
- `.numpy()` / `.cpu()` is the explicit download
- `.item()` for scalar extraction
- `__array__` protocol for backwards compatibility during transition

---

## Section 3: bindings.cpp Split

4,221 lines → 11 focused files.

| File | Ops | Est. Lines |
|------|-----|-----------|
| `bindings_core.cpp` | `GpuBuffer` class, `to_gpu_buffer()`, device init, buffer pool, module entry (`PYBIND11_MODULE`) | ~300 |
| `bindings_linear.cpp` | linear, linear_backward | ~150 |
| `bindings_conv.cpp` | conv2d, conv2d_backward_input, conv2d_backward_weight | ~200 |
| `bindings_activations.cpp` | relu, gelu, silu, tanh + all backwards | ~250 |
| `bindings_attention.cpp` | attention_scores, attention_mask, attention_output, concat_heads, flash_attention2, rope | ~300 |
| `bindings_normalization.cpp` | layernorm, layernorm_backward, rmsnorm, batchnorm2d, softmax, softmax_backward | ~250 |
| `bindings_optim.cpp` | sgd_update, adam_update, adamw_update | ~200 |
| `bindings_loss.cpp` | cross_entropy_loss, cross_entropy_backward | ~100 |
| `bindings_snn.cpp` | lif_step, snn_node_forward/backward, hebbian, stdp, synapse_filter, gif_neuron | ~300 |
| `bindings_pooling.cpp` | maxpool2d, avgpool2d, mean_pool | ~150 |
| `bindings_misc.cpp` | dropout, embedding, kv_cache ops | ~200 |

All files include `bindings_core.h` for `to_gpu_buffer()` and shared types.

Each binding function:
1. Calls `to_gpu_buffer()` on each input (numpy → upload, GpuBuffer → passthrough)
2. Allocates output `GpuBuffer` from pool
3. Dispatches C++ op
4. Returns `GpuBuffer` (stays on GPU)

---

## Section 4: _bridge.py Simplification

Shrinks from 950 lines to ~200. Thin pass-through to C++, no numpy prep.

```python
# Before (current)
def linear(x, weight, bias):
    x = _ensure_f32_contiguous(x)
    weight = _ensure_f32_contiguous(weight)
    dev = _get_device()
    return dev.linear(x, weight, bias)  # returns numpy

# After
def linear(x, weight, bias=None):
    return _device.linear(x, weight, bias)  # returns GpuBuffer
```

`_ensure_f32_contiguous` moves into C++ `to_gpu_buffer()`. No `_cpp_available` check — C++ is always available when grilly is installed.

---

## Section 5: nn/ Modules → VulkanTensor

All `nn.Module` subclasses store parameters as `VulkanTensor`, forward takes/returns `VulkanTensor`.

```python
class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weight = VulkanTensor(xavier_init(out_features, in_features))
        self.bias = VulkanTensor(np.zeros(out_features, dtype=np.float32))

    def forward(self, x: VulkanTensor) -> VulkanTensor:
        out = _bridge.linear(x._gpu, self.weight._gpu, self.bias._gpu)
        return VulkanTensor(gpu_buffer=out)
```

**Base `Module` changes:**
- `parameters()` returns list of `VulkanTensor`
- `state_dict()` calls `.numpy()` on each parameter for serialization
- `load_state_dict()` wraps numpy arrays in `VulkanTensor` (upload on load)
- `to(device)` — if "cpu", downloads all params; if "gpu", uploads all params (no-op if already there)

---

## Section 6: functional/ and optim/

**functional/** — Thin wrappers, no `Compute()` instantiation:

```python
def linear(input: VulkanTensor, weight: VulkanTensor, bias=None) -> VulkanTensor:
    out = _bridge.linear(input._gpu, weight._gpu, bias._gpu if bias else None)
    return VulkanTensor(gpu_buffer=out)
```

**optim/** — GPU-side updates, no download/upload per step:

```python
# SGD is the recommended default (no catastrophic forgetting)
class SGD(Optimizer):
    def step(self):
        for param in self.params:
            result = _bridge.sgd_update(param.data._gpu, param.grad._gpu, self.lr, self.momentum)
            param.data = VulkanTensor(gpu_buffer=result)
```

**Optimizer priority:**
1. **SGD** — primary default, no forgetting issues
2. **AutoHypergradientAdamW** — custom optimizer with OSGM surprise signal, handles forgetting
3. **AdamW/Adam** — available but NOT recommended as default (amplifies catastrophic forgetting per recent research)
4. **NaturalGradient, NLMS** — specialty optimizers

**Full training loop on GPU:**
```python
x = VulkanTensor(batch_data)       # upload once
y = VulkanTensor(batch_targets)    # upload once
output = model(x)                  # all GPU
loss = criterion(output, y)        # all GPU
loss.backward()                    # all GPU
optimizer.step()                   # all GPU
print(f"loss: {loss.item()}")      # single scalar download
```

---

## Section 7: Migration Strategy

Incremental, each phase independently testable. No big bang.

### Phase 1: C++ Foundation
- Create `GpuBuffer` class and `to_gpu_buffer()` helper
- Split `bindings.cpp` into 11 files
- Each binding accepts BOTH numpy and `GpuBuffer`, returns `GpuBuffer`
- Add `GpuBuffer.to_numpy()` method
- **Old Python code still works** — numpy in, auto-converts `GpuBuffer` back via `__array__`
- Tests: verify every op produces identical results via both paths

### Phase 2: VulkanTensor Rewrite
- Rewrite `VulkanTensor` to hold `GpuBuffer` internally
- `__array__` protocol for numpy interop during transition
- Constructor accepts numpy (uploads) or `GpuBuffer` (wraps)
- Tests: VulkanTensor creation, download, arithmetic

### Phase 3: nn/ Modules → VulkanTensor
- `module.py` base class: parameters as `VulkanTensor`
- Each layer's `forward()` passes `._gpu` handles
- Tests: each module verified against numpy reference

### Phase 4: functional/ and optim/
- Functional API takes `VulkanTensor`
- Optimizers use C++ kernels with `GpuBuffer` handles
- Remove all `Compute()` instantiation from functional/
- Add SGD as recommended default
- Tests: training loop end-to-end on GPU

### Phase 5: Cleanup
- Remove legacy `backend/core.py` ctypes Vulkan path
- Remove `_ensure_f32_contiguous` from bridge
- Remove numpy imports from hot paths
- Update all tutorials and examples
- Update CLAUDE.md architecture docs

---

## Performance Expectations

| Scenario | Before (ping-pong) | After (GPU-first) |
|----------|--------------------|--------------------|
| Linear chain (3 ops) | 3 upload + 3 download | 1 upload + 1 download |
| Transformer block | ~20 transfers per layer | 2 per forward pass |
| Full training step | O(n) transfers per op | O(1) transfers per step |
| Small tensors | PCIe latency dominated | GPU compute dominated |
| Large tensors | PCIe bandwidth wasted | Full GPU bandwidth |

The speedup depends on tensor size and op count. For a typical transformer with 12 layers and ~20 ops per layer, we go from ~480 PCIe transfers per forward pass to 2. That's where the real performance win is.

---

## Files Changed Summary

### C++ (new/modified)
- `cpp/include/grilly/gpu_buffer.h` — new, GpuBuffer struct
- `cpp/python/bindings_core.cpp` — new, replaces monolithic bindings.cpp
- `cpp/python/bindings_linear.cpp` — new
- `cpp/python/bindings_conv.cpp` — new
- `cpp/python/bindings_activations.cpp` — new
- `cpp/python/bindings_attention.cpp` — new
- `cpp/python/bindings_normalization.cpp` — new
- `cpp/python/bindings_optim.cpp` — new (includes SGD kernel binding)
- `cpp/python/bindings_loss.cpp` — new
- `cpp/python/bindings_snn.cpp` — new
- `cpp/python/bindings_pooling.cpp` — new
- `cpp/python/bindings_misc.cpp` — new
- `cpp/python/bindings.cpp` — deleted (replaced by split files)
- `CMakeLists.txt` — updated to compile split binding files

### Python (modified)
- `utils/tensor_conversion.py` — VulkanTensor rewrite (GPU-first)
- `backend/_bridge.py` — simplified to thin pass-through (~200 lines)
- `backend/core.py` — deprecated, eventually removed
- `nn/module.py` — parameters as VulkanTensor
- `nn/*.py` — all layers forward() with VulkanTensor
- `functional/*.py` — all ops take/return VulkanTensor
- `optim/*.py` — GPU-side updates via GpuBuffer handles
- `optim/sgd.py` — promoted to recommended default
