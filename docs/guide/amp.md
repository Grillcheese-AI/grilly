# Automatic Mixed Precision (AMP)

AMP uses float16 compute with float32 accumulation for eligible ops, reducing memory usage and improving throughput. Mirrors the PyTorch `torch.cuda.amp` API.

---

## Basic Usage

```python
from grilly.backend.amp import autocast, GradScaler

scaler = GradScaler()

for x_batch, y_batch in train_loader:
    with autocast():
        output = model(x_batch)
        loss = criterion(output, y_batch)

    scaled_loss = scaler.scale(loss)
    # backward pass
    grad = criterion.backward(np.ones_like(scaled_loss), output, y_batch)
    model.zero_grad()
    model.backward(grad)

    scaler.step(optimizer)
    scaler.update()
```

---

## autocast

Context manager that enables mixed precision for eligible operations:

```python
from grilly.backend.amp import autocast, is_autocast_enabled

with autocast():
    assert is_autocast_enabled()
    output = model(x)  # Eligible ops use float16 compute

# Outside the context, autocast is off
assert not is_autocast_enabled()
```

**Eligible operations**: linear, conv2d, attention. These are flagged to use float16 compute paths in the bridge when autocast is active.

**Non-eligible operations**: loss computation, normalization, softmax. These stay in float32 for numerical stability.

Autocast contexts can be nested:

```python
with autocast():
    h = model.encoder(x)
    with autocast(enabled=False):
        # Force float32 for a specific section
        loss = loss_fn(h, targets)
```

---

## GradScaler

Scales loss values to prevent float16 gradient underflow:

```python
from grilly.backend.amp import GradScaler

scaler = GradScaler(
    init_scale=65536.0,       # Initial loss scale (2^16)
    growth_factor=2.0,        # Scale up when no overflow
    backoff_factor=0.5,       # Scale down on overflow
    growth_interval=2000,     # Steps between growth attempts
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `scaler.scale(loss)` | Multiply loss by current scale factor |
| `scaler.step(optimizer)` | Unscale gradients, check for overflow, then optimizer.step() |
| `scaler.update()` | Update the scale factor based on overflow history |

---

## How It Works

1. **autocast** sets a thread-local flag via `is_autocast_enabled()`
2. Bridge functions check this flag and dispatch float16 compute paths when safe
3. **GradScaler** multiplies loss by a large scale factor before backward
4. Before the optimizer step, gradients are unscaled by dividing by the same factor
5. If overflow (inf/nan) is detected, the step is skipped and the scale is reduced
6. If no overflow for `growth_interval` steps, the scale is increased

This matches PyTorch's AMP behavior exactly.
