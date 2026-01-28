"""
Neural Network Tutorial
=======================

Training a simple neural network using Grilly's GPU-accelerated operations.
Demonstrates: Linear layers, activations, and backward passes.

Based on PyTorch's neural network tutorials.
"""

import _path_setup  # noqa: F401 - must be first to ensure grilly is importable

import numpy as np
import math

from grilly import Compute

compute = Compute()
print(f"GPU: {compute.core.device_properties.deviceName}")

# Generate training data: y = sin(x)
N = 64  # batch size
D_in = 1  # input dimension
H = 32  # hidden dimension
D_out = 1  # output dimension

# Create random input data
np.random.seed(42)
x = np.linspace(-math.pi, math.pi, N).reshape(N, D_in).astype(np.float32)
y = np.sin(x).astype(np.float32)

# Initialize weights with Xavier/Glorot initialization
def xavier_init(in_dim, out_dim):
    scale = np.sqrt(2.0 / (in_dim + out_dim))
    return (np.random.randn(out_dim, in_dim) * scale).astype(np.float32)

w1 = xavier_init(D_in, H)   # (32, 1)
b1 = np.zeros(H, dtype=np.float32)
w2 = xavier_init(H, D_out)  # (1, 32)
b2 = np.zeros(D_out, dtype=np.float32)

learning_rate = 1e-3
epochs = 500

print(f"\nTraining neural network: {D_in} -> {H} (ReLU) -> {D_out}")
print(f"Data: {N} samples, fitting sin(x)")
print("=" * 60)

for epoch in range(epochs):
    # ============ Forward Pass (GPU) ============
    # Layer 1: Linear + ReLU
    h = compute.fnn.linear(x, w1, b1)  # (N, H)
    h_relu = compute.fnn.activation_relu(h)  # (N, H)

    # Layer 2: Linear
    y_pred = compute.fnn.linear(h_relu, w2, b2)  # (N, D_out)

    # ============ Compute Loss ============
    diff = y_pred - y
    loss = np.mean(diff ** 2)

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}: loss = {loss:.6f}")

    # ============ Backward Pass ============
    # Gradient of MSE loss
    grad_y_pred = (2.0 / N) * diff  # (N, D_out)

    # Backward through layer 2 (manual gradient computation)
    # grad_input = grad_output @ weights
    # grad_weight = grad_output.T @ input
    # grad_bias = sum(grad_output)
    grad_h_relu = grad_y_pred @ w2  # (N, H)
    grad_w2 = grad_y_pred.T @ h_relu  # (D_out, H)
    grad_b2 = np.sum(grad_y_pred, axis=0)  # (D_out,)

    # Backward through ReLU
    grad_h = grad_h_relu * (h > 0).astype(np.float32)  # ReLU gradient

    # Backward through layer 1
    grad_w1 = grad_h.T @ x  # (H, D_in)
    grad_b1 = np.sum(grad_h, axis=0)  # (H,)

    # ============ Update Weights ============
    w1 -= learning_rate * grad_w1
    b1 -= learning_rate * grad_b1
    w2 -= learning_rate * grad_w2
    b2 -= learning_rate * grad_b2

print("=" * 60)
print(f"Final loss: {loss:.6f}")

# Test predictions
test_x = np.array([[-2.0], [0.0], [1.57], [3.14]]).astype(np.float32)
test_y_true = np.sin(test_x)

h = compute.fnn.linear(test_x, w1, b1)
h_relu = compute.fnn.activation_relu(h)
test_y_pred = compute.fnn.linear(h_relu, w2, b2)

print("\nTest predictions:")
print(f"  x = -2.0: pred = {test_y_pred[0,0]:.4f}, true = {test_y_true[0,0]:.4f}")
print(f"  x =  0.0: pred = {test_y_pred[1,0]:.4f}, true = {test_y_true[1,0]:.4f}")
print(f"  x =  1.57: pred = {test_y_pred[2,0]:.4f}, true = {test_y_true[2,0]:.4f}")
print(f"  x =  3.14: pred = {test_y_pred[3,0]:.4f}, true = {test_y_true[3,0]:.4f}")

# Show buffer pool stats
pool = compute.fnn.buffer_pool
if pool:
    stats = pool.get_stats()
    print(f"\nGPU Buffer pool: {stats['total_acquired']} ops, {stats['hit_rate']:.1%} hit rate")
