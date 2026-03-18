# Autograd Integration Design — Merge Variable into VulkanTensor

**Date:** 2026-03-18
**Goal:** Unify VulkanTensor (GPU compute) and Variable (autograd graph) into one tensor type

## Current Architecture (two worlds)

```
VulkanTensor (GPU)          Variable (autograd)
├── _t (C++ Tensor)         ├── data (numpy)
├── shape, dtype             ├── requires_grad
├── numpy(), cpu()           ├── grad, grad_fn
├── on_gpu, ensure_gpu()     ├── backward()
└── No gradient tracking     └── No GPU compute
```

## Target Architecture (unified)

```
VulkanTensor (GPU + autograd)
├── _t (C++ Tensor)          # GPU compute
├── requires_grad             # autograd flag
├── grad (VulkanTensor)       # accumulated gradient
├── grad_fn (GradFn)          # backward function node
├── is_leaf                   # leaf node flag
├── backward()                # trigger backprop
├── detach()                  # detach from graph
└── All existing methods
```

## Implementation Strategy

### Phase A: Add autograd fields to VulkanTensor

In `utils/tensor_conversion.py`, add to `VulkanTensor.__init__`:
```python
self.requires_grad = requires_grad
self.grad = None
self.grad_fn = None
self._is_leaf = True
```

Add methods:
- `backward(grad_output=None)` — walks `grad_fn` chain, computes gradients
- `detach()` — returns copy with `requires_grad=False`, no `grad_fn`
- `is_leaf` property

### Phase B: Make Variable a thin wrapper

In `nn/autograd.py`, change `Variable` to subclass or alias `VulkanTensor`:
```python
class Variable(VulkanTensor):
    """Backward-compatible alias. VulkanTensor now has full autograd."""
    def __init__(self, data, requires_grad=False, grad_fn=None):
        super().__init__(data)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
```

Or even simpler: `Variable = VulkanTensor`

### Phase C: Autograd ops use VulkanTensor

The 80+ differentiable functions in `nn/autograd.py` currently create `Variable` objects.
They should create `VulkanTensor` objects instead. Since Variable becomes an alias,
this happens automatically.

The backward functions currently use `numpy()` — they should use `_bridge` backward
ops when available (already partially done via `gpu_backward_fn`).

### Phase D: Split nn/autograd.py

2,373 lines → split into:
- `nn/autograd/variable.py` — Variable/VulkanTensor with autograd (or just use VulkanTensor)
- `nn/autograd/function.py` — Function, FunctionCtx, FunctionMeta
- `nn/autograd/ops.py` — the 80+ differentiable op functions
- `nn/autograd/grad_fn.py` — GradFn class
- `nn/autograd/__init__.py` — re-exports

## Key Insight

The C++ `Tensor` already has `requires_grad` and `grad` fields (from tensor.h).
The C++ `autograd.cpp` already has backward dispatch for 20+ ops.
The Python `Variable` already has 80+ ops with backward functions.

We just need to wire them together through VulkanTensor.
