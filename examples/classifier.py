"""
Digit Classifier Example
=========================

Trains a 3-layer MLP on the sklearn digits dataset (8x8 images, 10 classes)
using the Grilly framework with GPU-accelerated Vulkan shaders.

Demonstrates:
  - Data loading with TensorDataset / DataLoader
  - Sequential model with Linear + ReLU layers
  - CrossEntropyLoss + Adam optimizer
  - Full forward -> loss -> backward -> step training loop
  - Evaluation with accuracy reporting

Usage:
    uv run python examples/classifier.py
"""

import sys
import time
from pathlib import Path

import numpy as np

# Ensure grilly is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grilly.nn import CrossEntropyLoss, Dropout, Linear, ReLU, Sequential
from grilly.optim import Adam
from grilly.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2, seed=42):
    """Load and preprocess sklearn digits dataset."""
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0  # normalize to [0, 1]
    y = digits.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def evaluate(model, X, y):
    """Compute accuracy on a dataset."""
    was_training = model.training
    model.eval()
    logits = model(X)
    model.train(was_training)
    preds = np.argmax(logits, axis=-1)
    return np.mean(preds == y)


def main():
    print("=" * 60)
    print("  Grilly Digit Classifier  (sklearn digits, 10 classes)")
    print("=" * 60)

    # -- hyperparameters --
    batch_size = 64
    epochs = 50
    lr = 1e-3
    log_every = 5

    # -- data --
    X_train, X_test, y_train, y_test = load_data()
    print(f"\nDataset : {X_train.shape[0]} train / {X_test.shape[0]} test")
    print(f"Features: {X_train.shape[1]}   Classes: {len(np.unique(y_train))}")

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # -- model --
    model = Sequential(
        Linear(64, 128),
        ReLU(),
        Dropout(0.2),
        Linear(128, 64),
        ReLU(),
        Dropout(0.2),
        Linear(64, 10),
    )
    print(f"\nModel:\n{model}")

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # -- training --
    print(f"\nTraining for {epochs} epochs  (batch_size={batch_size}, lr={lr})")
    print("-" * 60)

    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch_x, batch_y = batch

            # Forward
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            # Backward
            grad = criterion.backward(input=logits, target=batch_y)
            model.backward(grad)

            # Optimizer step
            optimizer.step()

            # Zero gradients for next iteration
            for p in model.parameters():
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad = None

            epoch_loss += float(loss)
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % log_every == 0 or epoch == 1:
            train_acc = evaluate(model, X_train, y_train)
            test_acc = evaluate(model, X_test, y_test)
            elapsed = time.perf_counter() - t_start
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"loss={avg_loss:.4f}  "
                f"train_acc={train_acc:.3f}  "
                f"test_acc={test_acc:.3f}  "
                f"[{elapsed:.1f}s]"
            )

    t_total = time.perf_counter() - t_start

    # -- final evaluation --
    print("-" * 60)
    final_train_acc = evaluate(model, X_train, y_train)
    final_test_acc = evaluate(model, X_test, y_test)

    print("\nFinal results:")
    print(f"  Train accuracy: {final_train_acc:.4f}")
    print(f"  Test accuracy:  {final_test_acc:.4f}")
    print(f"  Total time:     {t_total:.2f}s  ({t_total / epochs * 1000:.1f} ms/epoch)")

    # -- per-class breakdown --
    model.eval()
    logits = model(X_test)
    preds = np.argmax(logits, axis=-1)

    print("\nPer-class accuracy:")
    for c in range(10):
        mask = y_test == c
        if mask.sum() > 0:
            acc = np.mean(preds[mask] == c)
            print(f"  digit {c}: {acc:.3f}  ({mask.sum()} samples)")

    print("\nDone.")


if __name__ == "__main__":
    main()
