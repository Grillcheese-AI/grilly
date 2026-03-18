# API Reference: optim

`grilly.optim` provides GPU-accelerated optimizers and learning rate schedulers.

Source: [`optim/`](https://github.com/grillcheese-ai/grilly/tree/main/optim)

---

## Base Class

| Class | Description |
|-------|-------------|
| `Optimizer` | Base optimizer. Provides `zero_grad()`, `step()`, `state_dict()`, `load_state_dict()`. |

---

## Optimizers

| Class | Signature | Description |
|-------|-----------|-------------|
| `Adam` | `Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)` | Adam optimizer. |
| `AdamW` | `AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)` | AdamW with decoupled weight decay. |
| `SGD` | `SGD(params, lr, momentum=0.0, nesterov=False)` | Stochastic gradient descent. |
| `NLMS` | `NLMS(params, lr=0.01, mu=0.1)` | Normalized least mean squares. |
| `NaturalGradient` | `NaturalGradient(params, lr=0.01)` | Natural gradient descent. |
| `HypergradientAdamW` | `HypergradientAdamW(params, lr=1e-3, hyper_lr=1e-7)` | AdamW with hypergradient LR tuning. |
| `AutoHypergradientAdamW` | `AutoHypergradientAdamW(params, lr=1e-3)` | OSGM-style auto LR with surprise signal. |
| `AffectAdam` | `AffectAdam(params, lr=1e-3)` | Adam with affect-modulated updates. |

---

## Learning Rate Schedulers

All schedulers accept an optimizer and modify its learning rate via `scheduler.step()`.

| Class | Signature | Description |
|-------|-----------|-------------|
| `StepLR` | `StepLR(optimizer, step_size, gamma=0.1)` | Decay LR by `gamma` every `step_size` epochs. |
| `CosineAnnealingLR` | `CosineAnnealingLR(optimizer, T_max, eta_min=0)` | Cosine annealing to `eta_min`. |
| `ReduceLROnPlateau` | `ReduceLROnPlateau(optimizer, patience=10, factor=0.1)` | Reduce LR when metric plateaus. Call `step(metric)`. |
| `OneCycleLR` | `OneCycleLR(optimizer, max_lr, total_steps)` | Super-convergence 1cycle policy. |

---

## Usage Pattern

```python
from grilly.optim import AdamW, CosineAnnealingLR

optimizer = AdamW(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    for x, y in loader:
        logits = model(x)
        loss = loss_fn(logits, y)
        grad = loss_fn.backward(np.ones_like(loss), logits, y)
        model.zero_grad()
        model.backward(grad)
        optimizer.step()
    scheduler.step()
```

---

## GPU-Accelerated Updates

Adam and AdamW updates are implemented as Vulkan compute shaders (`adam-update.glsl`, `adamw-update.glsl`). When the C++ backend is available, parameter updates run on GPU without downloading gradients to CPU.
