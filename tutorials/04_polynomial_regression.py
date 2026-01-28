"""
Polynomial Regression Tutorial
==============================

Adapted from PyTorch tutorial: Fitting y = sin(x) with a polynomial.
Demonstrates manual gradient computation and weight updates.

This shows how Grilly can work alongside numpy for training loops.
"""

import _path_setup  # noqa: F401 - must be first to ensure grilly is importable

import numpy as np
import math

# Grilly for GPU-accelerated operations
from grilly import Compute

compute = Compute()
print(f"GPU: {compute.core.device_properties.deviceName}")

# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000).astype(np.float32)
y = np.sin(x).astype(np.float32)

# Randomly initialize weights
np.random.seed(42)
a = np.float32(np.random.randn())
b = np.float32(np.random.randn())
c = np.float32(np.random.randn())
d = np.float32(np.random.randn())

learning_rate = 1e-6

print("\nTraining polynomial regression: y = a + bx + cx^2 + dx^3")
print("=" * 60)

for t in range(2000):
    # Forward pass: compute predicted y
    # y_pred = a + b*x + c*x^2 + d*x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss (MSE * n)
    loss = np.sum((y_pred - y) ** 2)
    if t % 100 == 99:
        print(f"Iteration {t+1:4d}: loss = {loss:.4f}")

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print("=" * 60)
print(f"Result: y = {a:.4f} + {b:.4f}x + {c:.4f}x^2 + {d:.4f}x^3")

# Verify with Taylor series of sin(x) around 0:
# sin(x) ≈ x - x^3/6 = 0 + 1*x + 0*x^2 - 0.1667*x^3
print(f"\nExpected (Taylor): y ≈ 0 + 1.0x + 0x^2 - 0.1667x^3")

# Test GPU operations on the result
print("\n--- GPU Acceleration Demo ---")
y_final = a + b * x + c * x ** 2 + d * x ** 3
y_final_2d = y_final.reshape(1, -1)

# Apply activation functions via GPU
relu_out = compute.fnn.activation_relu(y_final_2d)
gelu_out = compute.fnn.activation_gelu(y_final_2d)

print(f"ReLU on predictions: min={relu_out.min():.4f}, max={relu_out.max():.4f}")
print(f"GELU on predictions: min={gelu_out.min():.4f}, max={gelu_out.max():.4f}")

# Show buffer pool stats
pool = compute.fnn.buffer_pool
if pool:
    stats = pool.get_stats()
    print(f"\nBuffer pool: {stats['hits']} hits, {stats['allocations']} allocs, {stats['hit_rate']:.1%} hit rate")
