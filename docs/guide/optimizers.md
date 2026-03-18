# Optimizers

All optimizers inherit from `optim.Optimizer` and follow the PyTorch API: `zero_grad()`, `step()`, `state_dict()`, `load_state_dict()`.

---

## AdamW

```python
from grilly.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

Decoupled weight decay regularization (Loshchilov & Hutter, 2019). The default optimizer for most tasks.

**Parameters:**

| Param | Default | Description |
|-------|---------|-------------|
| `lr` | `1e-3` | Learning rate |
| `betas` | `(0.9, 0.999)` | Exponential decay rates for moment estimates |
| `eps` | `1e-8` | Numerical stability |
| `weight_decay` | `0.01` | Decoupled weight decay coefficient |

---

## Adam

```python
from grilly.optim import Adam

optimizer = Adam(model.parameters(), lr=1e-3)
```

Standard Adam with L2 regularization (not decoupled).

---

## SGD

```python
from grilly.optim import SGD

optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
```

Stochastic gradient descent with optional momentum and Nesterov acceleration.

**Parameters:**

| Param | Default | Description |
|-------|---------|-------------|
| `lr` | required | Learning rate |
| `momentum` | `0.0` | Momentum factor |
| `nesterov` | `False` | Nesterov momentum |

---

## AutoHypergradientAdamW

```python
from grilly.optim import AutoHypergradientAdamW

optimizer = AutoHypergradientAdamW(model.parameters(), lr=1e-3)
```

OSGM-style automatic learning rate tuning via a hypergradient surprise signal. Adjusts the learning rate each step based on gradient alignment with the previous update direction. No manual LR scheduling needed for many tasks.

---

## NLMS

```python
from grilly.optim import NLMS

optimizer = NLMS(model.parameters(), lr=0.01, mu=0.1)
```

Normalized Least Mean Squares optimizer. Useful for adaptive filtering tasks.

---

## NaturalGradient

```python
from grilly.optim import NaturalGradient

optimizer = NaturalGradient(model.parameters(), lr=0.01)
```

Natural gradient descent using Fisher information matrix approximation.

---

## Learning Rate Schedulers

All schedulers follow the PyTorch API: call `scheduler.step()` each epoch (or step).

### StepLR

```python
from grilly.optim import StepLR

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# LR *= 0.1 every 10 epochs
```

### CosineAnnealingLR

```python
from grilly.optim import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100)
```

### ReduceLROnPlateau

```python
from grilly.optim import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
scheduler.step(val_loss)  # Pass the monitored metric
```

### OneCycleLR

```python
from grilly.optim import OneCycleLR

scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=1000)
```

---

## Training Loop

```python
from grilly import nn
from grilly.optim import AdamW, CosineAnnealingLR

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
optimizer = AdamW(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=50)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(50):
    for x_batch, y_batch in train_loader:
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        grad = loss_fn.backward(np.ones_like(loss), logits, y_batch)

        model.zero_grad()
        model.backward(grad)
        optimizer.step()

    scheduler.step()
```

---

## Full API

See [API Reference: optim](../api/optim.md) for all optimizer and scheduler classes.
