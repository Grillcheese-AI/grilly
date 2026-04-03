# GPU dispatch performance (Python `VulkanCore` / `VulkanCompute`)

This document summarizes **non-blocking** and **batched** Vulkan dispatch APIs and how they relate to the PyTorch parity performance workstreams (B1, B2, D4).

## Synchronous vs async (`backend/core.py`)

- **`VulkanCore._dispatch_compute(...)`** — Records one command buffer, submits, **waits on a fence** (default `wait_previous=True`). Safe and simple; one fence wait per call.
- **`VulkanCore._dispatch_compute_async(...)`** — Submits work **without** waiting; pair with **`_wait_async()`** before reading GPU results.
- **`VulkanCore.record_commands()`** — Returns a **`CommandRecorder`** context manager: multiple `dispatch` + `barrier` calls, then **one** submit + wait on `__exit__`. Used by FlashAttention2 tiling, RMSNorm two-pass, and **`VulkanFNN._linear_relu_recorded_chain`** (Linear→ReLU fallback when `fused-linear-relu.spv` is missing).

## Public aliases on `VulkanCompute` (`backend/compute.py`)

After `Compute()` / `VulkanCompute()` construction:

| Attribute | Underlying API |
|-----------|----------------|
| `record_commands` | `core.record_commands()` |
| `dispatch_compute` | `core._dispatch_compute` |
| `dispatch_compute_async` | `core._dispatch_compute_async` |
| `wait_async` | `core._wait_async` |
| `wait_fence` | `core._wait_fence` |

Prefer **`record_commands`** for dependent kernels that share the same queue and do not need host-visible results between dispatches.

## `nn.Sequential` fusion

`nn.Sequential` already detects **Linear → ReLU / GELU / SiLU** and calls **`backend.fnn.fused_*`** when those shaders exist. If the fused shader is absent, **`fused_linear_relu`** tries **`_linear_relu_recorded_chain`** (two dispatches, **one** submit) before falling back to separate `linear` + `activation_relu` calls.

## Pybind11 GIL policy (workstream B3)

Heavy `grilly_core` entrypoints should release the GIL around GPU work:

```cpp
{
    py::gil_scoped_release release;
    grilly::ops::someOp(...);
}
```

Bindings should **request** `py::buffer_info` / output arrays **before** releasing the GIL. New array-based inputs should satisfy **`require_c_contiguous_float`** / **`require_c_contiguous_uint32`** / **`require_c_contiguous_int8`** (`bindings_core.h`) so kernels see dense row-major data.

## Code review checklist (bindings)

1. GIL held only for buffer prep and return value wrapping; **not** during `CommandBatch` / pool work.
2. `float32` numpy arrays are **C-contiguous** (or explicitly copied) for GPU kernels.
3. No duplicate `request()` on invalid buffers after GIL re-acquire without re-verifying lifetimes.

Covered in this policy (non-exhaustive): activations, attention, linear, conv, normalization, loss, SNN, pooling, optim GPU steps, **misc** (dropout, embedding, KV cache GPU paths, `swizzle_kv`), **Hamming** (GPU path), **SigLIP** / **Perceiver** / **MoQE train**, **`ShaderFusionEngine.fuse`**, **`OpGraph.optimize` / `OpGraph.execute`**.
