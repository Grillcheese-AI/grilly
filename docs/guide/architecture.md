# Architecture

Grilly has three layers: Python modules, a C++ bridge, and Vulkan compute shaders.

---

## Layer Stack

```
┌─────────────────────────────────────────────────────────┐
│  Python API                                             │
│  nn.Module, nn.Linear, F.relu, AdamW, Variable          │
│  grilly.nn / grilly.functional / grilly.optim            │
├─────────────────────────────────────────────────────────┤
│  C++ Bridge (grilly_core)                               │
│  pybind11 bindings, VMA persistent mapping              │
│  BufferPool bucketed allocation, CommandBatch            │
│  backend/_bridge.py dispatches to C++ or falls back     │
├─────────────────────────────────────────────────────────┤
│  Vulkan Compute Shaders                                 │
│  194 GLSL shaders compiled to SPIR-V                    │
│  Dispatched via VkComputePipeline                       │
│  AMD / NVIDIA / Intel -- no CUDA                        │
└─────────────────────────────────────────────────────────┘
```

---

## Package Layout

The repo root **is** the `grilly` package. `pyproject.toml` uses `tool.setuptools.package-dir` to map subpackages:

```
grilly/
├── backend/              # Vulkan GPU dispatch
│   ├── _bridge.py        # Routes ops through grilly_core C++
│   ├── core.py           # Vulkan instance/device init, buffer alloc
│   ├── compute.py        # VulkanCompute: composes all op modules
│   ├── pipelines.py      # Pipeline/descriptor-set creation + LRU cache
│   ├── autograd_core.py  # GradientTape, ComputationNode, backward ops
│   ├── jit.py            # Trace-based JIT compilation
│   ├── amp.py            # Automatic mixed precision
│   └── shader_registry.py # Architecture-specific shader selection
├── cpp/                  # C++ pybind11 extension
│   ├── src/              # Core: device, buffer_pool, pipeline_cache, autograd
│   │   ├── ops/          # Op implementations: linear, conv, attention, etc.
│   │   └── vulkan/       # Vulkan abstraction layer
│   └── python/           # pybind11 binding files (11 focused files)
├── nn/                   # PyTorch-like nn.Module subclasses
│   ├── module.py         # Base Module (parameters, train/eval, state_dict)
│   ├── linear.py         # Linear
│   ├── conv.py           # Conv1d, Conv2d
│   ├── attention.py      # MultiheadAttention, FlashAttention2, HYLA, SympFormer
│   ├── autograd.py       # Variable, reverse-mode autodiff
│   ├── snn_*.py          # SNN framework (neurons, containers, surrogate grads)
│   └── lora.py           # LoRA fine-tuning
├── functional/           # Stateless F.* API
├── optim/                # Optimizers + LR schedulers
├── utils/                # DataLoader, VulkanTensor, HuggingFaceBridge
├── shaders/              # 194 GLSL compute shaders + SPIR-V in spv/
└── experimental/         # VSA, MoE, temporal reasoning, cognitive
```

---

## Data Flow

### Without C++ Backend (Pure Python)

```
numpy array --> backend/core.py --> struct.pack --> vkMapMemory
    --> ctypes.memmove --> Vulkan dispatch --> fence wait
    --> vkMapMemory (read back) --> numpy array
```

Each operation maps/unmaps GPU memory. This CPU-GPU ping-pong is the bottleneck.

### With C++ Backend (grilly_core)

```
numpy array --> _bridge.py --> grilly_core.linear()
    --> VMA persistent mapping (single memcpy, no map/unmap)
    --> BufferPool bucketed allocation (reuse buffers)
    --> Vulkan dispatch --> result stays GPU-resident
    --> next op reads from same GPU buffer
    --> final result: single download to numpy
```

The bridge uses a **try/fallback** pattern. If the C++ backend is available, ops go through `grilly_core`. Otherwise, they fall back to the pure Python Vulkan path:

```python
result = _bridge.linear(x, weight, bias)
if result is not None:
    return result
# Fall back to legacy backend
```

---

## Shader Dispatch

Each neural network operation is a GLSL compute shader compiled to SPIR-V bytecode. The `shader_registry.py` selects architecture-specific variants (BERT, GPT, T5) with a generic fallback.

Shaders are loaded at device initialization from `shaders/spv/`. The pipeline cache (`pipelines.py`) creates and LRU-caches `VkComputePipeline` objects for each shader + specialization constant combination.

See [Compute Shaders](shaders.md) for details on the shader system.

---

## Entry Points

The main entry point is `grilly.Compute()` (alias for `VulkanCompute`):

```python
import grilly

backend = grilly.Compute()
print(backend.device_name)  # "AMD Radeon RX 6750 XT"
```

`VulkanCompute` composes all operation modules into a single object with namespaced ops:

- `backend.snn.lif_step()` -- SNN neuron operations
- `backend.fnn.linear()` -- feedforward operations
- `backend.attention.flash_attention2()` -- attention mechanisms
- `backend.learning.stdp_update()` -- learning rules

Most users interact through `nn.Module` subclasses or `grilly.functional` rather than calling `VulkanCompute` directly.
