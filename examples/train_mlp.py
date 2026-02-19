"""Full training loop: 3-layer MLP on synthetic data."""

import grilly.nn as nn
import grilly.optim as optim
import numpy as np

# Model: 3-layer MLP with GELU activations
model = nn.Sequential(
    nn.Linear(64, 128),
    nn.GELU(),
    nn.Linear(128, 64),
    nn.GELU(),
    nn.Linear(64, 10),
)

optimizer = optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

# Synthetic dataset: 256 samples, 10 classes
np.random.seed(42)
X = np.random.randn(25600, 64).astype(np.float32)
y = np.random.randint(0, 10, size=25600)

batch_size = 32
epochs = 100

for epoch in range(epochs):
    epoch_loss = 0.0
    n_batches = 0

    indices = np.random.permutation(len(X))
    for start in range(0, len(X), batch_size):
        idx = indices[start : start + batch_size]
        xb, yb = X[idx], y[idx]

        # Forward
        logits = model(xb)
        loss = loss_fn(logits, yb)

        # Backward
        grad = loss_fn.backward(np.ones_like(loss), logits, yb)
        model.zero_grad()
        model.backward(grad)

        # Update
        optimizer.step()

        epoch_loss += float(np.mean(loss))
        n_batches += 1

    print(f"Epoch {epoch + 1:2d}/{epochs}  loss={epoch_loss / n_batches:.4f}")

print("Training complete.")
