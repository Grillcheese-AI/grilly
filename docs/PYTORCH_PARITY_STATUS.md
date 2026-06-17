# PyTorch vs Grilly Parity: Features and Status

This document tracks practical API/feature parity between PyTorch and Grilly.

Scope:
- User-facing Python APIs (`grilly.nn`, `grilly.functional`, `grilly.optim`, autograd)
- Runtime behavior (GPU backend, fallbacks, mixed precision, JIT)
- Ecosystem capabilities commonly expected by PyTorch users

Status key:
- **Implemented**: Available in Grilly with a direct equivalent
- **Partial**: Available with caveats, subset support, or different ergonomics
- **Not Yet**: No direct equivalent today
- **Different by Design**: Solved differently than PyTorch

---

## 1) Core Modeling API

| PyTorch Capability | Grilly Equivalent | Status | Notes |
|---|---|---|---|
| `nn.Module` base class | `grilly.nn.Module` | Implemented | Supports `parameters()`, `train()/eval()`, `state_dict()`, `load_state_dict()`. |
| `nn.Parameter` | `grilly.nn.Parameter` | Implemented | Parameter registration and gradient storage supported. |
| `nn.Sequential` | `grilly.nn.Sequential` | Implemented | Standard container composition. |
| Residual blocks/patterns | `grilly.nn.Residual` | Implemented | Explicit residual container exists. |
| Forward + backward pattern | `module(x)`, `module.backward(...)` | Partial | PyTorch-style usage works, but some workflows are more explicit in Grilly. |

---

## 2) Layers and Operators

| PyTorch Capability | Grilly Equivalent | Status | Notes |
|---|---|---|---|
| Linear | `nn.Linear`, `F.linear` | Implemented | Native GPU path through Vulkan shaders/C++ bridge. |
| Embedding | `nn.Embedding`, functional embedding ops | Implemented | Core embedding lookup supported. |
| Convolution | `nn.Conv1d`, `nn.Conv2d` | Implemented | Includes backward kernels and GEMM/direct paths. |
| Recurrent layers | `nn.LSTM`, `nn.GRU`, `Cell` variants | Implemented | Present in API docs and module exports. |
| Normalization | `LayerNorm`, `RMSNorm`, `BatchNorm1d/2d` | Implemented | Available in `nn` and functional/backends. |
| Activations | `ReLU`, `GELU`, `SiLU`, `Softmax`, etc. | Implemented | Includes additional non-PyTorch activations (GCU, RoSwish, SwiGLU). |
| Pooling | `MaxPool2d`, `AvgPool2d`, adaptive pooling | Implemented | Core pooling family present. |
| Loss functions | `MSE`, `CrossEntropy`, `BCE` | Implemented | Loss + backward entry points exist. |
| Attention | `MultiheadAttention`, FlashAttention2/3, RoPE | Partial | Strong support for supported forms; not a 1:1 drop-in for all PyTorch attention variants. |
| Transformer blocks | Encoder/Decoder layers | Partial | Available modules exist; ecosystem-level compatibility depends on model pattern. |

---

## 3) Autograd and Training

| PyTorch Capability | Grilly Equivalent | Status | Notes |
|---|---|---|---|
| Reverse-mode autograd | `Variable`, `Function`, graph tracking | Implemented | Core autodiff support exists in `grilly.nn` APIs. |
| Gradient enable/disable contexts | `no_grad()`, `enable_grad()` | Implemented | Present in API reference. |
| Standard optimizer set | `Adam`, `AdamW`, `SGD` | Partial | Available in `grilly.optim`; CPU stepping for **SGD** / **Adam** checked vs `torch.optim` in `tests/parity/test_optimizers_parity.py`. |
| Scheduler family | `StepLR`, cosine, plateau, one-cycle | Implemented | Documented scheduler support. |
| AMP (mixed precision) | `autocast()`, `GradScaler` | Partial | Available in Grilly, but hardware/backend behavior differs from CUDA AMP expectations. |
| End-to-end training loops | `zero_grad()`, backward, step | Implemented | Supported in the documented quick-start flow. |

---

## 4) Runtime and Backend

| PyTorch Capability | Grilly Equivalent | Status | Notes |
|---|---|---|---|
| CUDA backend | Vulkan backend (`grilly_core` + shaders) | Different by Design | Grilly targets Vulkan for cross-vendor GPU support. |
| CPU fallback | Numpy/Numba fallback paths | Implemented | Same codepaths can run without GPU acceleration. |
| Bridge strict mode + fallback telemetry | `GRILLY_BRIDGE_STRICT`, `_bridge.get_fallback_stats()` | Implemented | Strict mode raises on bridge failures; counters expose fallback frequency for perf/debug workflows. |
| Device targeting | Vulkan GPU selection + env controls | Partial | Works, but differs from `torch.device` UX patterns. |
| Kernel fusion/JIT | `@grilly.jit`, fused shader paths | Partial | Present, but differs from TorchInductor/`torch.compile` workflow. |
| Memory management | C++ buffer pool + VMA | Different by Design | Not CUDA allocator semantics, but optimized for Vulkan backend. |
| Kernel / throughput (Workstream C) | Conv GEMM backward weight, INT8 packed GEMM, FA2 batching, documented transfer batching | Implemented (scoped) | See `docs/PYTORCH_PARITY_TASKLIST.md` **Workstream C** (closed) and **Workstream C — future** for optional follow-ups (e.g. dedicated transfer queue, INT8 tiling). |
| Legacy API deprecation cleanup | Migration from `Compute()`/legacy paths to `_bridge` | Partial | `functional/*.py`, `nn/module.py`, and `utils/tensor_conversion.py` no longer rely on direct `Compute()` usage; remaining work is policy/CI enforcement and legacy-path quarantine. |

---

## 5) Functional API Parity

`grilly.functional` mirrors a meaningful subset of `torch.nn.functional`.

| Area | Status | Notes |
|---|---|---|
| Core activations/linear/softmax/dropout | Implemented | Direct functional calls available. |
| Attention ops including flash attention | Partial | Available for supported signatures; not full PyTorch signature parity everywhere. |
| Normalization and common losses | Implemented | Layer norm and key loss functions exposed. |
| Domain-specific extras (memory, FAISS-like, bridge ops) | Different by Design | Grilly includes GPU ops outside typical PyTorch core functional surface. |
| Numerical parity tests vs numpy / optional PyTorch | Partial | `tests/parity/` — `linear`, `relu`, chains; extend for full matrix (see tasklist A1). |
| Migration cookbook | Partial | `docs/MIGRATION_PYTORCH.md` (device model, layouts, lifecycle). |

---

## 6) Ecosystem and Tooling Parity

| PyTorch Ecosystem Capability | Grilly Status | Notes |
|---|---|---|
| HuggingFace model ecosystem | Partial | Bridge tooling exists (e.g. HuggingFace integrations), but not universal model parity. |
| ONNX workflows | Partial | Import/export tooling exists; coverage depends on operators/model patterns. |
| Distributed training (`torch.distributed`, DDP/FSDP) | Not Yet | No equivalent documented as first-class today. |
| TorchScript / `torch.compile` parity | Not Yet | Different compilation model; Grilly has its own JIT/fusion approach. |
| CUDA-specific libraries (cuDNN, NCCL ecosystem) | Different by Design | Grilly avoids CUDA lock-in and uses Vulkan stack. |

---

## 7) SNN and Research Features (Beyond PyTorch Core)

Grilly includes native capabilities that are not first-class in standard PyTorch:

- Spiking neural network neurons and surrogate gradients
- Temporal and cognitive experimental modules
- VSA / symbolic-neural tooling
- LoRA-focused modules and multimodal fusion blocks

These are not parity gaps; they are Grilly extensions.

---

## 8) Practical Parity Guidance

If you are migrating from PyTorch:

1. **High parity today**: Standard MLP/CNN training loops, common losses, Adam/AdamW/SGD, core activations, basic attention workflows.
2. **Read** `docs/MIGRATION_PYTORCH.md` for layouts (`F.linear`-compatible weight shape), backend lifecycle, and debugging tips.
3. **Validate carefully**: Complex transformer variants, full ecosystem integrations, advanced mixed-precision and compiler workflows.
4. **Plan adaptations**: Distributed training stacks, TorchScript/`torch.compile` assumptions, and CUDA-specific tooling.
5. **Use bridge observability in perf testing**: Set `GRILLY_BRIDGE_STRICT=1` to fail fast on unexpected bridge fallbacks, and inspect fallback counters via `grilly.backend._bridge.get_fallback_stats()`.
6. **Run parity smoke tests**: `pytest tests/parity/` (install optional `torch` for cross-checks against `torch.nn.functional`).

---

## 9) Suggested Maintenance Policy

To keep this document useful across releases:

- Update status per release (`Implemented`/`Partial`/`Not Yet`)
- Add links to concrete docs/tests for each upgraded item
- Track parity blockers as checklist items in release notes

