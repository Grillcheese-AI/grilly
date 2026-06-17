# PyTorch → Grilly migration cookbook

This guide helps teams port models and training scripts from PyTorch to Grilly. Pair it with
`docs/PYTORCH_PARITY_STATUS.md` (what is implemented) and `docs/PYTORCH_PARITY_TASKLIST.md`
(roadmap).

## Mental model

| Topic | PyTorch | Grilly |
|--------|-----------|--------|
| Default compute | CUDA / CPU | Vulkan GPU + numpy/numba CPU fallback |
| Core arrays | `torch.Tensor` | `numpy.ndarray` (float32 for GPU ops) |
| Device API | `torch.device`, `.to(device)` | Env (`VK_GPU_INDEX`), `utils.device_manager`, Vulkan-focused |
| Autograd | `tensor.backward()` | `Variable` / explicit `module.backward()` patterns (see API docs) |

Grilly is **not** a drop-in `import torch` replacement: expect to adjust device handling,
tensor types, and a few module signatures.

## Functional API (`grilly.functional`)

- **Linear**: `functional.linear(x, weight, bias=None)` with `weight` shaped `(out_features, in_features)`.
  Matches `torch.nn.functional.linear` layout (same as `nn.Linear.weight`).
- **Activations**: Named like PyTorch (`relu`, `gelu`, `softmax`, …). Prefer `dim=-1` where applicable;
  verify behavior for non-default axes against a small reference (see `tests/parity/`).
- **Fallbacks**: Functions try `backend._bridge` (C++ GPU) first, then numpy CPU. Set
  `GRILLY_BRIDGE_STRICT=1` to fail fast when the bridge errors instead of falling back
  (see `backend/_bridge.py`).

## Modules (`grilly.nn`)

- Subclass `nn.Module`, compose `nn.Linear`, `nn.Conv2d`, etc.
- **Backend lifecycle**: Each `Module` lazily resolves a Vulkan backend via
  `utils.device_manager.get_device_manager().vulkan` (a `VulkanCompute` instance). The
  device manager **reuses** that backend for the process; you normally do **not** create a
  new GPU context per layer.
- Prefer **numpy float32** activations and parameters for GPU shaders unless you use
  `VulkanTensor` / C++ `Tensor` for zero-copy paths.

## Training loop sketch

```python
import numpy as np
import grilly.nn as nn

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
loss_fn = nn.CrossEntropyLoss()  # check API for your installed version

for x, y in batches:  # x: (N, 784), y: (N,) int labels
    logits = model.forward(x)
    loss = loss_fn(logits, y)
    # backward + optimizer step per your autograd / Variable setup
```

Adapt optimizer usage to `grilly.optim` (Adam, AdamW, SGD, …). Validate updates against
PyTorch on a tiny network if you need step-for-step parity (`tests/parity/` will grow).

## Attention

- `nn.MultiheadAttention` supports bridge-accelerated paths when query/key **sequence lengths match**
  and masking matches the implemented contract. Cross-attention with `Sq ≠ Sk` may use legacy
  numpy/Vulkan paths; see `docs/GPU_OPTIMIZATION_REVIEW.md` (attention / optimization notes).

## Debugging performance

1. `GRILLY_BRIDGE_STRICT=1` — surface bridge failures early.
2. `grilly.backend._bridge.get_fallback_stats()` — count GPU→CPU fallbacks after a run.
3. `benchmarks/profile_gpu_bottlenecks.py` — Python-side hotspots (dispatch / fences).

## What to validate in your project

1. **Numerics**: Run `pytest tests/parity/` (install optional `torch` for cross-checks).
2. **Shapes**: Confirm `Linear` / `Conv` weight layouts match your checkpoint conversion.
3. **I/O**: HuggingFace and ONNX workflows remain **partial** parity; test your exact model.
