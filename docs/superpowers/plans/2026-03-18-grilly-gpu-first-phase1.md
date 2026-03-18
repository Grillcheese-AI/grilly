# Grilly GPU-First Phase 1: C++ Tensor Exposure + Bindings Split

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the existing C++ `grilly::nn::Tensor` to Python, split `bindings.cpp` into focused files, and make binding functions accept/return `Tensor` objects so data stays on GPU between operations.

**Architecture:** The C++ `grilly::nn::Tensor` (cpp/include/grilly/nn/tensor.h) already implements dual-validity GPU↔CPU tracking with lazy sync. We expose it to Python via pybind11, then refactor each binding function to accept `Tensor` (zero-copy GPU passthrough) OR numpy (auto-upload via `Tensor::from_numpy`). This eliminates CPU↔GPU ping-pong at the binding boundary.

**Tech Stack:** C++17, pybind11, Vulkan/VMA, CMake

**Spec:** `docs/superpowers/specs/2026-03-18-grilly-gpu-first-design.md`

**Key discovery:** `cpp/include/grilly/nn/tensor.h` already has everything we need — `buffer_handle_`, `cpu_data_`, `gpu_valid_`/`cpu_valid_` tracking, `ensure_gpu()`/`ensure_cpu()`, `from_numpy()`/`.numpy()`, `from_gpu()`, `mark_gpu_modified()`, autograd support. Zero new C++ data structures needed.

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `cpp/python/bindings_core.h` | Shared header: `to_tensor()` converter, forward declarations, common includes |
| `cpp/python/bindings_core.cpp` | `PYBIND11_MODULE` entry, `Tensor` class binding, device init |
| `cpp/python/bindings_linear.cpp` | `linear`, `linear_backward` — accept/return `Tensor` |
| `cpp/python/bindings_activations.cpp` | All activations + backwards |
| `cpp/python/bindings_conv.cpp` | `conv2d`, backwards |
| `cpp/python/bindings_attention.cpp` | Attention ops, flash_attention2, rope |
| `cpp/python/bindings_normalization.cpp` | layernorm, rmsnorm, batchnorm, softmax + backwards |
| `cpp/python/bindings_optim.cpp` | adam, adamw + future sgd_update |
| `cpp/python/bindings_loss.cpp` | cross_entropy + backward |
| `cpp/python/bindings_snn.cpp` | LIF, SNN node, hebbian, stdp, synapse, GIF |
| `cpp/python/bindings_pooling.cpp` | maxpool2d, avgpool2d, mean_pool |
| `cpp/python/bindings_misc.cpp` | dropout, embedding, kv_cache ops |
| `tests/test_cpp_tensor.py` | Python tests for C++ Tensor exposure |

### Modified Files

| File | Changes |
|------|---------|
| `CMakeLists.txt` | Update `pybind11_add_module` to include split binding files |
| `cpp/python/bindings.cpp` | Deleted (replaced by split files) |

---

## Task 1: Expose C++ Tensor to Python

**Files:**
- Create: `cpp/python/bindings_core.h`
- Create: `cpp/python/bindings_core.cpp`
- Modify: `CMakeLists.txt`
- Create: `tests/test_cpp_tensor.py`

- [ ] **Step 1: Create bindings_core.h — shared header**

```cpp
// cpp/python/bindings_core.h
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "grilly/nn/tensor.h"
#include "grilly/compute_backend.h"
#include "grilly/device.h"
#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace py = pybind11;
using grilly::nn::Tensor;

/// Global device context (initialized on first use)
struct DeviceContext {
    grilly::Device device;
    bool initialized = false;

    void ensure_init(const std::string& shader_dir = "");
    grilly::ComputeBackend* backend();
};

DeviceContext& get_context();

/// Universal input converter: accepts Tensor or numpy, returns Tensor on GPU.
/// If input is already a Tensor with valid GPU data, returns it directly (zero-copy).
/// If input is numpy, creates a Tensor and uploads.
Tensor to_tensor(py::object input);

/// Register all binding submodules (called from PYBIND11_MODULE)
void register_linear_ops(py::module_& m);
void register_activation_ops(py::module_& m);
void register_conv_ops(py::module_& m);
void register_attention_ops(py::module_& m);
void register_normalization_ops(py::module_& m);
void register_optim_ops(py::module_& m);
void register_loss_ops(py::module_& m);
void register_snn_ops(py::module_& m);
void register_pooling_ops(py::module_& m);
void register_misc_ops(py::module_& m);
```

- [ ] **Step 2: Create bindings_core.cpp — module entry + Tensor binding**

```cpp
// cpp/python/bindings_core.cpp
#include "bindings_core.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

// ── Global device context ────────────────────────────────────────────
static DeviceContext g_ctx;

DeviceContext& get_context() { return g_ctx; }

void DeviceContext::ensure_init(const std::string& shader_dir) {
    if (initialized) return;
    device.init(shader_dir);
    initialized = true;
}

grilly::ComputeBackend* DeviceContext::backend() {
    if (!initialized) throw std::runtime_error("Device not initialized. Call load_shaders() first.");
    return device.backend();
}

// ── to_tensor converter ──────────────────────────────────────────────
Tensor to_tensor(py::object input) {
    if (py::isinstance<Tensor>(input)) {
        return input.cast<Tensor>();
    }
    // numpy path: convert to Tensor and upload
    auto arr = py::array_t<float>::ensure(input, py::array::c_style | py::array::forcecast);
    if (!arr) throw std::runtime_error("Cannot convert input to float32 array");
    return Tensor::from_numpy(arr, get_context().backend());
}

// ── PYBIND11_MODULE ──────────────────────────────────────────────────
PYBIND11_MODULE(grilly_core, m) {
    m.doc() = "grilly C++ backend — GPU-accelerated neural network operations";

    // ── Tensor class ─────────────────────────────────────────────────
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def_static("from_numpy", [](py::array_t<float> arr) {
            return Tensor::from_numpy(arr, get_context().backend());
        }, py::arg("arr"), "Create tensor from numpy array (uploads to GPU)")
        .def_static("zeros", [](std::vector<int64_t> shape) {
            return Tensor::zeros(shape, get_context().backend());
        }, py::arg("shape"), "Create zero-filled tensor on GPU")
        .def_static("empty", [](std::vector<int64_t> shape) {
            return Tensor::empty(shape, get_context().backend());
        }, py::arg("shape"), "Create uninitialized tensor on GPU")
        .def("numpy", &Tensor::numpy, "Download to CPU and return as numpy array")
        .def("gpu_handle", &Tensor::gpu_handle, "Get GPU buffer handle (uploads if needed)")
        .def("ensure_gpu", &Tensor::ensure_gpu, "Ensure data is on GPU")
        .def("release_gpu", &Tensor::release_gpu, "Release GPU buffer, keep CPU data")
        .def("mark_gpu_modified", &Tensor::mark_gpu_modified)
        .def("mark_cpu_modified", &Tensor::mark_cpu_modified)
        .def_property_readonly("shape", [](const Tensor& t) { return t.shape(); })
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property_readonly("numel", &Tensor::numel)
        .def_property_readonly("nbytes", &Tensor::nbytes)
        .def_property_readonly("on_gpu", &Tensor::on_gpu)
        .def_property_readonly("on_cpu", &Tensor::on_cpu)
        .def_property("requires_grad",
            &Tensor::requires_grad, &Tensor::set_requires_grad)
        .def("__repr__", [](const Tensor& t) {
            std::string s = "grilly.Tensor(shape=[";
            for (size_t i = 0; i < t.shape().size(); ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(t.shape()[i]);
            }
            s += "]";
            s += t.on_gpu() ? ", device=gpu" : ", device=cpu";
            s += ")";
            return s;
        })
    ;

    // ── Device init ──────────────────────────────────────────────────
    m.def("load_shaders", [](const std::string& shader_dir) {
        get_context().ensure_init(shader_dir);
    }, py::arg("shader_dir"), "Initialize Vulkan device and load SPIR-V shaders");

    // ── Register all op submodules ───────────────────────────────────
    register_linear_ops(m);
    register_activation_ops(m);
    register_conv_ops(m);
    register_attention_ops(m);
    register_normalization_ops(m);
    register_optim_ops(m);
    register_loss_ops(m);
    register_snn_ops(m);
    register_pooling_ops(m);
    register_misc_ops(m);
}
```

- [ ] **Step 3: Update CMakeLists.txt**

Change line 263 from:
```cmake
pybind11_add_module(grilly_core cpp/python/bindings.cpp)
```
To:
```cmake
pybind11_add_module(grilly_core
    cpp/python/bindings_core.cpp
    cpp/python/bindings_linear.cpp
    cpp/python/bindings_activations.cpp
    cpp/python/bindings_conv.cpp
    cpp/python/bindings_attention.cpp
    cpp/python/bindings_normalization.cpp
    cpp/python/bindings_optim.cpp
    cpp/python/bindings_loss.cpp
    cpp/python/bindings_snn.cpp
    cpp/python/bindings_pooling.cpp
    cpp/python/bindings_misc.cpp
)
```

- [ ] **Step 4: Create stub binding files**

Create each `bindings_*.cpp` with a minimal stub that compiles but has empty `register_*_ops` functions. This lets us build incrementally.

Example for each file:
```cpp
// cpp/python/bindings_linear.cpp
#include "bindings_core.h"

void register_linear_ops(py::module_& m) {
    // TODO: migrate from bindings.cpp
}
```

- [ ] **Step 5: Write test**

```python
# tests/test_cpp_tensor.py
import sys
import numpy as np
sys.path.insert(0, ".")

def test_tensor_class_exists():
    """C++ Tensor should be importable from grilly_core."""
    import grilly_core
    assert hasattr(grilly_core, "Tensor")

def test_tensor_from_numpy():
    """Tensor.from_numpy should create a GPU tensor."""
    import grilly_core
    grilly_core.load_shaders("shaders/spv")
    arr = np.random.randn(4, 8).astype(np.float32)
    t = grilly_core.Tensor.from_numpy(arr)
    assert t.shape == [4, 8]
    assert t.numel == 32

def test_tensor_roundtrip():
    """Upload numpy → GPU → download should preserve data."""
    import grilly_core
    grilly_core.load_shaders("shaders/spv")
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t = grilly_core.Tensor.from_numpy(arr)
    result = t.numpy()
    np.testing.assert_allclose(result, arr, atol=1e-6)

def test_tensor_zeros():
    """Tensor.zeros should create a GPU tensor of zeros."""
    import grilly_core
    grilly_core.load_shaders("shaders/spv")
    t = grilly_core.Tensor.zeros([3, 5])
    result = t.numpy()
    assert result.shape == (3, 5)
    np.testing.assert_allclose(result, 0.0)
```

- [ ] **Step 6: Build and test**

```bash
cd C:/Users/grill/Documents/GitHub/grilly
cmake --build build --config Release
uv run pytest tests/test_cpp_tensor.py -v
```

- [ ] **Step 7: Commit**

```bash
git add cpp/python/bindings_core.h cpp/python/bindings_core.cpp
git add cpp/python/bindings_linear.cpp cpp/python/bindings_activations.cpp
git add cpp/python/bindings_conv.cpp cpp/python/bindings_attention.cpp
git add cpp/python/bindings_normalization.cpp cpp/python/bindings_optim.cpp
git add cpp/python/bindings_loss.cpp cpp/python/bindings_snn.cpp
git add cpp/python/bindings_pooling.cpp cpp/python/bindings_misc.cpp
git add CMakeLists.txt tests/test_cpp_tensor.py
git commit -m "feat: expose C++ Tensor to Python, split bindings into 11 files"
```

---

## Task 2: Migrate Linear Ops to Tensor-Based Bindings

**Files:**
- Modify: `cpp/python/bindings_linear.cpp`
- Modify: `tests/test_cpp_tensor.py`

- [ ] **Step 1: Write test**

```python
def test_linear_with_tensor():
    """linear() should accept Tensor and return Tensor (GPU-resident)."""
    import grilly_core
    grilly_core.load_shaders("shaders/spv")
    x = grilly_core.Tensor.from_numpy(np.random.randn(2, 4).astype(np.float32))
    w = grilly_core.Tensor.from_numpy(np.random.randn(8, 4).astype(np.float32))
    b = grilly_core.Tensor.from_numpy(np.zeros(8, dtype=np.float32))
    out = grilly_core.linear(x, w, b)
    assert isinstance(out, grilly_core.Tensor)
    assert out.shape == [2, 8]
    assert out.on_gpu  # should still be on GPU

def test_linear_with_numpy_fallback():
    """linear() should also accept numpy (auto-upload)."""
    import grilly_core
    grilly_core.load_shaders("shaders/spv")
    x = np.random.randn(2, 4).astype(np.float32)
    w = np.random.randn(8, 4).astype(np.float32)
    b = np.zeros(8, dtype=np.float32)
    out = grilly_core.linear(x, w, b)
    assert isinstance(out, grilly_core.Tensor)
    result = out.numpy()
    assert result.shape == (2, 8)

def test_linear_chain_no_download():
    """Chaining two linears should NOT download between them."""
    import grilly_core
    grilly_core.load_shaders("shaders/spv")
    x = grilly_core.Tensor.from_numpy(np.random.randn(2, 4).astype(np.float32))
    w1 = grilly_core.Tensor.from_numpy(np.random.randn(8, 4).astype(np.float32))
    b1 = grilly_core.Tensor.from_numpy(np.zeros(8, dtype=np.float32))
    w2 = grilly_core.Tensor.from_numpy(np.random.randn(3, 8).astype(np.float32))
    b2 = grilly_core.Tensor.from_numpy(np.zeros(3, dtype=np.float32))

    h = grilly_core.linear(x, w1, b1)   # GPU → GPU
    out = grilly_core.linear(h, w2, b2)  # GPU → GPU (no download between)
    assert out.on_gpu
    result = out.numpy()  # only download here
    assert result.shape == (2, 3)
```

- [ ] **Step 2: Implement bindings_linear.cpp**

Migrate the `linear` and `linear_backward` functions from `bindings.cpp`, changing them to use `to_tensor()` for input and return `Tensor`:

```cpp
// cpp/python/bindings_linear.cpp
#include "bindings_core.h"
#include "grilly/ops/linear.h"

void register_linear_ops(py::module_& m) {
    m.def("linear", [](py::object x, py::object weight, py::object bias) -> Tensor {
        auto& ctx = get_context();
        Tensor x_t = to_tensor(x);
        Tensor w_t = to_tensor(weight);

        auto M = x_t.shape(0);
        auto K = x_t.shape(x_t.ndim() - 1);
        auto N = w_t.shape(0);

        Tensor out = Tensor::empty({M, N}, ctx.backend());
        out.ensure_gpu();

        grilly::ops::LinearParams p;
        p.M = static_cast<uint32_t>(M);
        p.K = static_cast<uint32_t>(K);
        p.N = static_cast<uint32_t>(N);
        p.has_bias = !bias.is_none();

        // Dispatch on GPU — all data stays in Vulkan buffers
        auto* be = ctx.backend();
        auto& pool = be->buffer_pool();
        auto& cache = be->pipeline_cache();
        auto batch = be->command_batch();

        if (p.has_bias) {
            Tensor b_t = to_tensor(bias);
            grilly::ops::linear(batch, pool, cache,
                reinterpret_cast<const float*>(x_t.gpu_handle()),
                reinterpret_cast<const float*>(w_t.gpu_handle()),
                reinterpret_cast<const float*>(b_t.gpu_handle()),
                reinterpret_cast<float*>(out.gpu_handle()), p);
        } else {
            grilly::ops::linear(batch, pool, cache,
                reinterpret_cast<const float*>(x_t.gpu_handle()),
                reinterpret_cast<const float*>(w_t.gpu_handle()),
                nullptr,
                reinterpret_cast<float*>(out.gpu_handle()), p);
        }

        batch.submit_and_wait();
        out.mark_gpu_modified();
        return out;
    }, py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
       "GPU linear: output = x @ W^T + bias. Returns Tensor (stays on GPU).");

    // linear_backward — same pattern
    m.def("linear_backward", [](py::object grad_output, py::object input, py::object weights) -> py::dict {
        auto& ctx = get_context();
        Tensor go_t = to_tensor(grad_output);
        Tensor in_t = to_tensor(input);
        Tensor w_t = to_tensor(weights);

        // Allocate output tensors on GPU
        Tensor grad_input = Tensor::empty(in_t.shape(), ctx.backend());
        Tensor grad_weight = Tensor::empty(w_t.shape(), ctx.backend());
        Tensor grad_bias = Tensor::empty({w_t.shape(0)}, ctx.backend());

        grad_input.ensure_gpu();
        grad_weight.ensure_gpu();
        grad_bias.ensure_gpu();

        // TODO: call actual backward op
        // grilly::ops::linear_backward(...)

        grad_input.mark_gpu_modified();
        grad_weight.mark_gpu_modified();
        grad_bias.mark_gpu_modified();

        py::dict result;
        result["grad_input"] = grad_input;
        result["grad_weight"] = grad_weight;
        result["grad_bias"] = grad_bias;
        return result;
    }, py::arg("grad_output"), py::arg("input"), py::arg("weights"),
       "GPU linear backward. Returns dict of Tensors.");
}
```

**NOTE:** The exact C++ op dispatch API may differ from what's shown — the implementer must read the actual `linear.cpp`/`linear.h` to match the real function signatures. The pattern above shows the architecture; adapt to the actual ops API.

- [ ] **Step 3: Build and test**

```bash
cd C:/Users/grill/Documents/GitHub/grilly
cmake --build build --config Release
uv run pytest tests/test_cpp_tensor.py -v
```

- [ ] **Step 4: Commit**

```bash
git add cpp/python/bindings_linear.cpp tests/test_cpp_tensor.py
git commit -m "feat: migrate linear ops to Tensor-based bindings (zero GPU ping-pong)"
```

---

## Task 3: Migrate Activation Ops

**Files:**
- Modify: `cpp/python/bindings_activations.cpp`

Same pattern as Task 2. Migrate: relu, gelu, silu, tanh + all backwards. Each function uses `to_tensor()` for input, `Tensor::empty()` for output, dispatches C++ op, `mark_gpu_modified()`, returns `Tensor`.

The existing `bindings.cpp` has a `defActivation` helper macro — preserve that pattern in the split file:

```cpp
// cpp/python/bindings_activations.cpp
#include "bindings_core.h"
#include "grilly/ops/activations.h"

void register_activation_ops(py::module_& m) {
    auto defActivation = [&](const char* name, auto op_fn) {
        m.def(name, [op_fn](py::object x) -> Tensor {
            auto& ctx = get_context();
            Tensor x_t = to_tensor(x);
            Tensor out = Tensor::empty(x_t.shape(), ctx.backend());
            out.ensure_gpu();
            op_fn(*ctx.backend(), x_t.gpu_handle(), out.gpu_handle(), x_t.numel());
            out.mark_gpu_modified();
            return out;
        }, py::arg("x"));
    };

    defActivation("relu", grilly::ops::relu);
    defActivation("gelu", grilly::ops::gelu);
    defActivation("silu", grilly::ops::silu);
    defActivation("tanh_act", grilly::ops::tanh_act);

    // Backwards follow same pattern with two inputs (grad_output, input/output)
    // ...
}
```

- [ ] **Step 1: Migrate all activations + backwards from bindings.cpp**
- [ ] **Step 2: Add test for activation chain staying on GPU**
- [ ] **Step 3: Build and test**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: migrate activation ops to Tensor-based bindings"
```

---

## Task 4: Migrate Conv, Attention, Normalization Ops

**Files:**
- Modify: `cpp/python/bindings_conv.cpp`
- Modify: `cpp/python/bindings_attention.cpp`
- Modify: `cpp/python/bindings_normalization.cpp`

Same `to_tensor()` → dispatch → `mark_gpu_modified()` → return `Tensor` pattern. These are the most complex ops (multi-input, multi-output).

- [ ] **Step 1: Migrate conv2d + backwards**
- [ ] **Step 2: Migrate attention ops (scores, mask, output, concat, flash_attention2, rope)**
- [ ] **Step 3: Migrate normalization ops (layernorm, rmsnorm, batchnorm, softmax + backwards)**
- [ ] **Step 4: Add tests for attention chain staying on GPU**
- [ ] **Step 5: Build and test**
- [ ] **Step 6: Commit**

```bash
git commit -m "feat: migrate conv, attention, normalization ops to Tensor bindings"
```

---

## Task 5: Migrate Remaining Ops (optim, loss, snn, pooling, misc)

**Files:**
- Modify: `cpp/python/bindings_optim.cpp`
- Modify: `cpp/python/bindings_loss.cpp`
- Modify: `cpp/python/bindings_snn.cpp`
- Modify: `cpp/python/bindings_pooling.cpp`
- Modify: `cpp/python/bindings_misc.cpp`

- [ ] **Step 1: Migrate optimizer ops (adam, adamw) — return Tensor dict**
- [ ] **Step 2: Migrate loss ops (cross_entropy + backward)**
- [ ] **Step 3: Migrate SNN ops (lif, snn_node, hebbian, stdp, synapse, gif)**
- [ ] **Step 4: Migrate pooling ops (maxpool, avgpool, mean_pool)**
- [ ] **Step 5: Migrate misc ops (dropout, embedding, kv_cache)**
- [ ] **Step 6: Build full test suite**
- [ ] **Step 7: Commit**

```bash
git commit -m "feat: migrate remaining ops to Tensor bindings"
```

---

## Task 6: Delete Old bindings.cpp and Final Validation

**Files:**
- Delete: `cpp/python/bindings.cpp`
- Modify: `CMakeLists.txt` (confirm old file removed from sources)

- [ ] **Step 1: Verify all ops from old bindings.cpp are migrated**

Count exported functions in old file vs new files. Must match.

- [ ] **Step 2: Delete bindings.cpp**

```bash
git rm cpp/python/bindings.cpp
```

- [ ] **Step 3: Full build + test suite**

```bash
cd C:/Users/grill/Documents/GitHub/grilly
cmake --build build --config Release
uv run pytest tests/ -v
```

- [ ] **Step 4: Run existing grilly test suite to verify backward compatibility**

Existing tests pass numpy to `_bridge.py` which calls the C++ module. Since `to_tensor()` auto-converts numpy, existing tests should pass without modification.

```bash
uv run pytest tests/ -v --tb=short
```

- [ ] **Step 5: Commit**

```bash
git commit -m "chore: remove old monolithic bindings.cpp (replaced by 11 focused files)"
```

---

## Phase 1 Complete — Summary

After all 6 tasks:

| Before | After |
|--------|-------|
| `bindings.cpp`: 4,221 lines, single file | 11 focused files, ~200-300 lines each |
| All ops take `py::array_t<float>` (numpy) | All ops take `Tensor` or numpy (auto-convert) |
| All ops return `py::array_t<float>` (numpy) | All ops return `Tensor` (stays on GPU) |
| No `Tensor` exposed to Python | Full `Tensor` class with GPU lifecycle |
| Every op boundary: upload + download | Zero-copy between chained ops |

**Next:** Phase 2 rewrites Python `VulkanTensor` to wrap C++ `Tensor`, Phase 3 updates nn/ modules, Phase 4 updates functional/optim.
