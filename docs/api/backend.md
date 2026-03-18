# API Reference: backend

`grilly.backend` provides the low-level Vulkan GPU dispatch layer.

Source: [`backend/`](https://github.com/grillcheese-ai/grilly/tree/main/backend)

---

## Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `VulkanCompute` | `backend/compute.py` | Main compute interface. Composes all op modules into namespaced access. |
| `SNNCompute` | `backend/snn_compute.py` | SNN-specific compute operations. |
| `VulkanLearning` | `backend/learning.py` | Learning rule operations (STDP, Hebbian, EWC, NLMS). |
| `VulkanLoRA` | `backend/lora.py` | LoRA operations on GPU. |

---

## Entry Point

```python
import grilly

backend = grilly.Compute()  # Alias for VulkanCompute
print(backend.device_name)
```

`VulkanCompute` exposes namespaced op modules:

- `backend.snn.*` -- SNN neuron operations
- `backend.fnn.*` -- Feedforward operations (linear, activations)
- `backend.attention.*` -- Attention mechanisms
- `backend.memory.*` -- Memory read/write
- `backend.learning.*` -- Learning rules
- `backend.fft.*` -- FFT operations
- `backend.conv.*` -- Convolution operations
- `backend.normalization.*` -- Normalization operations
- `backend.faiss.*` -- FAISS similarity search
- `backend.cells.*` -- Place/time cells
- `backend.lora.*` -- LoRA operations

---

## Bridge Layer

| Symbol | Module | Description |
|--------|--------|-------------|
| `_bridge.is_available()` | `backend/_bridge.py` | Check if C++ backend is loaded. |
| `_bridge.linear(x, w, b)` | `backend/_bridge.py` | Route linear op through C++. |
| `_bridge.q_similarity(q)` | `backend/_bridge.py` | Q-similarity via C++ dispatch. |

The bridge lazily initializes a `grilly_core.Device` singleton and routes ops through C++ with a fallback to the Python Vulkan path.

---

## Autograd Core

| Class | Module | Description |
|-------|--------|-------------|
| `GradientTape` | `backend/autograd_core.py` | Record operations for gradient computation. |
| `ComputationNode` | `backend/autograd_core.py` | Single node in the computation graph. |
| `TrainingContext` | `backend/autograd_core.py` | Training context with gradient tracking. |
| `ModuleTracer` | `backend/autograd_core.py` | Trace module operations for backward. |

---

## JIT

| Class | Module | Description |
|-------|--------|-------------|
| `Tracer` | `backend/jit.py` | Records ops during forward pass. |
| `TracedGraph` | `backend/jit.py` | Captured computation graph. |
| `OpRecord` | `backend/jit.py` | Single recorded operation. |
| `trace(fn, inputs)` | `backend/jit.py` | Trace a function and return a TracedGraph. |

---

## AMP

| Class | Module | Description |
|-------|--------|-------------|
| `autocast` | `backend/amp.py` | Context manager for mixed precision. |
| `GradScaler` | `backend/amp.py` | Loss scaling for float16 training. |
| `is_autocast_enabled()` | `backend/amp.py` | Check if autocast is active. |

---

## Shader Registry

| Class | Module | Description |
|-------|--------|-------------|
| `ShaderRegistry` | `backend/shader_registry.py` | Select architecture-specific shaders with generic fallback. |

---

## Pipeline Cache

| Class | Module | Description |
|-------|--------|-------------|
| `PipelineCache` | `backend/pipelines.py` | LRU cache for VkComputePipeline objects. |

---

## Utilities

| Symbol | Module | Description |
|--------|--------|-------------|
| `VULKAN_AVAILABLE` | `backend/base.py` | Boolean: True if Vulkan is available on this system. |
