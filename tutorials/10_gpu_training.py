"""
GPU Training Tutorial
=====================

Training neural networks on GPU using Vulkan compute shaders.
This demonstrates FULLY GPU-accelerated forward AND backward passes
using custom Vulkan shaders for all operations.

Features:
- Fused Linear+ReLU forward (GPU)
- ReLU backward (GPU shader: activation-relu-backward.spv)
- Linear backward (GPU shader: fnn-linear-backward.spv)
- Cross-entropy backward (GPU shader: cross-entropy-backward.spv)
"""

import _path_setup  # noqa: F401

import numpy as np
import time
from pathlib import Path

from grilly import Compute
from grilly.datasets import MNIST
from grilly.utils.data import DataLoader


class GPUNeuralNetwork:
    """
    Fully GPU-accelerated neural network using Vulkan compute shaders.

    Architecture: 784 -> 256 -> 128 -> 10

    Both forward AND backward passes run on GPU:
    - Forward: fused_linear_relu, linear (GPU)
    - Backward: activation_relu_backward, linear_backward (GPU)
    """

    def __init__(self, compute: Compute):
        self.compute = compute
        self.fnn = compute.fnn

        # Initialize weights with Xavier initialization
        self.w1 = self._xavier_init(784, 256)
        self.b1 = np.zeros(256, dtype=np.float32)

        self.w2 = self._xavier_init(256, 128)
        self.b2 = np.zeros(128, dtype=np.float32)

        self.w3 = self._xavier_init(128, 10)
        self.b3 = np.zeros(10, dtype=np.float32)

        # Gradients
        self.dw1 = np.zeros_like(self.w1)
        self.db1 = np.zeros_like(self.b1)
        self.dw2 = np.zeros_like(self.w2)
        self.db2 = np.zeros_like(self.b2)
        self.dw3 = np.zeros_like(self.w3)
        self.db3 = np.zeros_like(self.b3)

        # Cache for backward pass
        self.cache = {}

    def _xavier_init(self, fan_in, fan_out):
        """Xavier/Glorot initialization."""
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(fan_out, fan_in).astype(np.float32) * std

    def forward(self, x):
        """
        Forward pass using GPU-accelerated operations.

        Args:
            x: Input batch (batch_size, 784)

        Returns:
            logits: Output logits (batch_size, 10)
        """
        # Layer 1: Linear + ReLU (fused on GPU)
        h1_pre = self.fnn.linear(x, self.w1, self.b1)  # Pre-activation for backward
        h1 = self.fnn.activation_relu(h1_pre)

        # Layer 2: Linear + ReLU (fused on GPU)
        h2_pre = self.fnn.linear(h1, self.w2, self.b2)  # Pre-activation for backward
        h2 = self.fnn.activation_relu(h2_pre)

        # Layer 3: Linear (logits)
        logits = self.fnn.linear(h2, self.w3, self.b3)

        # Cache activations for backward pass
        self.cache = {
            'x': x,
            'h1_pre': h1_pre,
            'h1': h1,
            'h2_pre': h2_pre,
            'h2': h2,
            'logits': logits,
        }

        return logits

    def backward(self, grad_output):
        """
        Backward pass using GPU-accelerated operations.

        Uses Vulkan shaders:
        - activation-relu-backward.spv for ReLU backward
        - fnn-linear-backward.spv for linear backward

        Args:
            grad_output: Gradient of loss w.r.t. logits (batch_size, 10)
        """
        batch_size = grad_output.shape[0]

        # Layer 3 backward (no activation, just linear)
        # GPU-accelerated linear backward
        grad_h2, self.dw3, self.db3 = self.fnn.linear_backward(
            grad_output, self.cache['h2'], self.w3, self.b3
        )
        self.dw3 = self.dw3 / batch_size
        self.db3 = self.db3 / batch_size

        # Layer 2 backward (ReLU + Linear)
        # GPU-accelerated ReLU backward
        grad_h2_pre = self.fnn.activation_relu_backward(grad_h2, self.cache['h2_pre'])

        # GPU-accelerated linear backward
        grad_h1, self.dw2, self.db2 = self.fnn.linear_backward(
            grad_h2_pre, self.cache['h1'], self.w2, self.b2
        )
        self.dw2 = self.dw2 / batch_size
        self.db2 = self.db2 / batch_size

        # Layer 1 backward (ReLU + Linear)
        # GPU-accelerated ReLU backward
        grad_h1_pre = self.fnn.activation_relu_backward(grad_h1, self.cache['h1_pre'])

        # GPU-accelerated linear backward
        _, self.dw1, self.db1 = self.fnn.linear_backward(
            grad_h1_pre, self.cache['x'], self.w1, self.b1
        )
        self.dw1 = self.dw1 / batch_size
        self.db1 = self.db1 / batch_size

    def update(self, lr):
        """Update weights using SGD."""
        self.w1 -= lr * self.dw1
        self.b1 -= lr * self.db1
        self.w2 -= lr * self.dw2
        self.b2 -= lr * self.db2
        self.w3 -= lr * self.dw3
        self.b3 -= lr * self.db3

    def zero_grad(self):
        """Zero all gradients."""
        self.dw1.fill(0)
        self.db1.fill(0)
        self.dw2.fill(0)
        self.db2.fill(0)
        self.dw3.fill(0)
        self.db3.fill(0)


class GPUCrossEntropyLoss:
    """
    GPU-accelerated cross-entropy loss with softmax.

    Uses the cross-entropy-backward shader for gradient computation.
    """

    def __init__(self, fnn):
        self.fnn = fnn

    def __call__(self, logits, targets):
        """
        Compute cross-entropy loss and gradient.

        Args:
            logits: Raw logits (batch_size, num_classes)
            targets: Target class indices (batch_size,)

        Returns:
            loss: Scalar loss value
            grad: Gradient w.r.t. logits (batch_size, num_classes)
        """
        batch_size = logits.shape[0]

        # Compute softmax for loss (CPU - just for the scalar loss value)
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Cross-entropy loss
        log_probs = np.log(probs + 1e-10)
        loss = -np.mean(log_probs[np.arange(batch_size), targets])

        # GPU-accelerated gradient: softmax - one_hot(targets)
        grad = self.fnn.cross_entropy_backward(logits, targets)

        return loss, grad


def softmax(x):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss_cpu(logits, targets):
    """
    CPU fallback cross-entropy loss with softmax.

    Returns:
        loss: Scalar loss value
        grad: Gradient w.r.t. logits
    """
    batch_size = logits.shape[0]

    # Softmax
    probs = softmax(logits)

    # Cross-entropy loss
    log_probs = np.log(probs + 1e-10)
    loss = -np.mean(log_probs[np.arange(batch_size), targets])

    # Gradient: softmax - one_hot(targets)
    grad = probs.copy()
    grad[np.arange(batch_size), targets] -= 1

    return loss, grad


def main():
    # Initialize GPU
    print("=" * 70)
    print("GPU Training Tutorial - Fully GPU-Accelerated")
    print("=" * 70)

    compute = Compute()
    print(f"GPU: {compute.core.device_properties.deviceName}")

    # Check available shaders
    available_shaders = list(compute.fnn.shaders.keys())
    backward_shaders = [s for s in available_shaders if 'backward' in s]
    print(f"\nBackward shaders available: {len(backward_shaders)}")
    for s in sorted(backward_shaders):
        print(f"  - {s}")

    # Load MNIST
    datasets_root = Path(__file__).parent.parent / 'datasets'

    def normalize(x):
        x = x.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return x.reshape(-1)

    print("\nLoading MNIST...")
    train_data = MNIST(root=str(datasets_root), train=True, transform=normalize)
    test_data = MNIST(root=str(datasets_root), train=False, transform=normalize)
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Create model
    print("\nCreating GPU neural network: 784 -> 256 -> 128 -> 10")
    model = GPUNeuralNetwork(compute)

    # Create GPU-accelerated loss function
    loss_fn = GPUCrossEntropyLoss(compute.fnn)

    # Training parameters
    learning_rate = 0.1
    epochs = 5

    print(f"\nTraining parameters:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"\nGPU Operations:")
    print(f"  Forward: linear (GPU), activation_relu (GPU)")
    print(f"  Backward: linear_backward (GPU), activation_relu_backward (GPU)")
    print(f"  Loss grad: cross_entropy_backward (GPU)")

    # Training loop
    print("\n" + "=" * 70)
    print("Training (Fully GPU-Accelerated)")
    print("=" * 70)

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        total_loss = 0
        n_batches = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            # Flatten images
            X_flat = X.reshape(X.shape[0], -1).astype(np.float32)
            y = y.astype(np.int64)

            # Forward pass (GPU)
            logits = model.forward(X_flat)

            # Compute loss and gradient (GPU gradient)
            loss, grad = loss_fn(logits, y)

            # Backward pass (GPU)
            model.zero_grad()
            model.backward(grad)

            # Update weights
            model.update(learning_rate)

            total_loss += loss
            n_batches += 1

        epoch_time = time.perf_counter() - epoch_start
        avg_loss = total_loss / n_batches
        samples_per_sec = len(train_data) / epoch_time

        # Evaluate on test set
        correct = 0
        total = 0
        test_loss = 0
        n_test = 0

        for X, y in test_loader:
            X_flat = X.reshape(X.shape[0], -1).astype(np.float32)
            logits = model.forward(X_flat)

            loss, _ = loss_fn(logits, y.astype(np.int64))
            test_loss += loss
            n_test += 1

            predictions = np.argmax(logits, axis=1)
            correct += (predictions == y).sum()
            total += len(y)

        accuracy = 100 * correct / total
        test_loss /= n_test

        print(f"Epoch {epoch + 1}/{epochs}: "
              f"train_loss={avg_loss:.4f}, "
              f"test_loss={test_loss:.4f}, "
              f"accuracy={accuracy:.1f}%, "
              f"time={epoch_time:.2f}s, "
              f"{samples_per_sec:.0f} samples/sec")

    # Final benchmark
    print("\n" + "=" * 70)
    print("Performance Benchmark")
    print("=" * 70)

    # Benchmark forward pass
    X_bench = np.random.randn(1000, 784).astype(np.float32)
    y_bench = np.random.randint(0, 10, 1000).astype(np.int64)

    # Warmup
    for _ in range(10):
        logits = model.forward(X_bench)
        loss, grad = loss_fn(logits, y_bench)
        model.backward(grad)

    # Benchmark forward only
    n_iters = 100
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = model.forward(X_bench)
    elapsed = time.perf_counter() - start

    samples_per_sec = (n_iters * 1000) / elapsed
    print(f"Forward pass only: {samples_per_sec:.0f} samples/sec ({elapsed/n_iters*1000:.2f}ms per batch)")

    # Benchmark full training iteration (forward + loss + backward)
    start = time.perf_counter()
    for _ in range(n_iters):
        logits = model.forward(X_bench)
        loss, grad = loss_fn(logits, y_bench)
        model.backward(grad)
    elapsed = time.perf_counter() - start

    samples_per_sec = (n_iters * 1000) / elapsed
    print(f"Full training step: {samples_per_sec:.0f} samples/sec ({elapsed/n_iters*1000:.2f}ms per batch)")

    # Compare with CPU-only backward
    print("\n--- Comparison: GPU vs CPU backward ---")

    # GPU backward benchmark
    logits = model.forward(X_bench)
    loss, grad = loss_fn(logits, y_bench)

    start = time.perf_counter()
    for _ in range(n_iters):
        model.backward(grad)
    elapsed_gpu = time.perf_counter() - start
    print(f"GPU backward: {elapsed_gpu/n_iters*1000:.2f}ms per batch")

    # CPU backward benchmark (using numpy operations)
    batch_size = grad.shape[0]

    start = time.perf_counter()
    for _ in range(n_iters):
        # Layer 3 backward (CPU)
        dw3 = model.cache['h2'].T @ grad / batch_size
        db3 = np.sum(grad, axis=0) / batch_size
        dh2 = grad @ model.w3

        # Layer 2 backward (CPU)
        relu_mask2 = (model.cache['h2_pre'] > 0).astype(np.float32)
        dh2_pre = dh2 * relu_mask2
        dw2 = model.cache['h1'].T @ dh2_pre / batch_size
        db2 = np.sum(dh2_pre, axis=0) / batch_size
        dh1 = dh2_pre @ model.w2

        # Layer 1 backward (CPU)
        relu_mask1 = (model.cache['h1_pre'] > 0).astype(np.float32)
        dh1_pre = dh1 * relu_mask1
        dw1 = model.cache['x'].T @ dh1_pre / batch_size
        db1 = np.sum(dh1_pre, axis=0) / batch_size
    elapsed_cpu = time.perf_counter() - start
    print(f"CPU backward: {elapsed_cpu/n_iters*1000:.2f}ms per batch")

    speedup = elapsed_cpu / elapsed_gpu if elapsed_gpu > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
