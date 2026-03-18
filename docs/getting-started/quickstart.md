# Quick Start

Build a model, train it, and see GPU output in under 20 lines.

---

## Your First Model

```python
import numpy as np
from grilly import nn
from grilly.optim import AdamW

# Define a simple classifier
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

optimizer = AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Dummy data (batch of 32, 784 features, 10 classes)
x = np.random.randn(32, 784).astype(np.float32)
targets = np.random.randint(0, 10, (32,))

# Forward pass
logits = model(x)
loss = loss_fn(logits, targets)

# Backward pass
grad = loss_fn.backward(np.ones_like(loss), logits, targets)
model.zero_grad()
model.backward(grad)

# Update weights
optimizer.step()

print(f"Loss: {loss.mean():.4f}")
```

All data is `np.float32` numpy arrays. The backend handles GPU upload and download transparently.

---

## Autograd

For automatic differentiation with a computation graph:

```python
from grilly.nn import Variable, tensor

x = Variable(tensor([1.0, 2.0, 3.0]), requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad)  # [2.0, 4.0, 6.0]
```

`Variable` tracks operations during the forward pass and computes gradients via reverse-mode autodiff. See [Autograd](../guide/autograd.md) for details.

---

## Functional API

Stateless operations that mirror `torch.nn.functional`:

```python
import grilly.functional as F

out = F.linear(x, weight, bias)
out = F.relu(out)
out = F.softmax(out, dim=-1)
out = F.flash_attention2(q, k, v)
```

See [API Reference: functional](../api/functional.md) for the full list.

---

## Using the DataLoader

```python
from grilly.utils import DataLoader, ArrayDataset, Compose, ToFloat32, Normalize

transform = Compose([ToFloat32(), Normalize(mean=0.5, std=0.5)])

dataset = ArrayDataset(x_train, y_train)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(10):
    for x_batch, y_batch in loader:
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        grad = loss_fn.backward(np.ones_like(loss), logits, y_batch)
        model.zero_grad()
        model.backward(grad)
        optimizer.step()
```

---

## Saving and Loading

```python
from grilly.utils import save_checkpoint, load_checkpoint

# Save
save_checkpoint(model, optimizer, epoch=10, path="checkpoint.npz")

# Load
load_checkpoint(model, optimizer, path="checkpoint.npz")
```

---

## Next Steps

- [Architecture](../guide/architecture.md) -- how the layer stack works
- [GPU-First Design](../guide/gpu-first.md) -- the C++ Tensor and zero ping-pong
- [Neural Network Modules](../guide/nn-modules.md) -- all available layers
- [Optimizers](../guide/optimizers.md) -- AdamW, SGD, auto-hypergradient
