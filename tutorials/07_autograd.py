"""
Autograd Tutorial
=================

Automatic differentiation with Grilly - similar to PyTorch's autograd.
Fitting y = exp(x) with a polynomial using gradient descent.

Based on PyTorch's autograd tutorial.
"""

import _path_setup  # noqa: F401 - must be first to ensure grilly is importable

import numpy as np

from grilly import Compute
from grilly.nn.autograd import Variable, linspace, randn, no_grad

# Initialize GPU (for later use)
compute = Compute()
print(f"Using GPU: {compute.core.device_properties.deviceName}")

# Create input and target data
# x in [-1, 1], y = exp(x)
x = linspace(-1, 1, 2000)
y = x.exp()  # Target: exp(x)

# Create random weights for polynomial: y = a + bx + cx^2 + dx^3
# Taylor expansion of exp(x): 1 + x + x^2/2 + x^3/6 + ...
np.random.seed(42)
a = randn(requires_grad=True)
b = randn(requires_grad=True)
c = randn(requires_grad=True)
d = randn(requires_grad=True)

initial_loss = 1.0
learning_rate = 1e-5

print("\nTraining: y = a + bx + cx^2 + dx^3  to fit y = exp(x)")
print("=" * 80)

for t in range(5000):
    # Forward pass using operator overloading (just like PyTorch!)
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute loss: sum((y_pred - y)^2)
    loss = ((y_pred - y) ** 2).sum()

    # Store initial loss for comparison
    if t == 0:
        initial_loss = loss.item()

    if t % 100 == 99:
        rel_loss = loss.item() / initial_loss
        print(f"t = {t+1:4d}  loss(t)/loss(0) = {rel_loss:10.6f}  "
              f"a = {a.data.item():10.6f}  b = {b.data.item():10.6f}  "
              f"c = {c.data.item():10.6f}  d = {d.data.item():10.6f}")

    # Backward pass: autograd computes gradients automatically
    a.zero_grad()
    b.zero_grad()
    c.zero_grad()
    d.zero_grad()
    loss.backward()

    # Update weights using gradient descent
    with no_grad():
        a.data -= learning_rate * a.grad
        b.data -= learning_rate * b.grad
        c.data -= learning_rate * c.grad
        d.data -= learning_rate * d.grad

print("=" * 80)
print(f"\nResult: y = {a.data.item():.4f} + {b.data.item():.4f}x + "
      f"{c.data.item():.4f}x^2 + {d.data.item():.4f}x^3")
print(f"Expected (Taylor): y ≈ 1.0 + 1.0x + 0.5x^2 + 0.167x^3")

# Final comparison
y_pred_final = a.data + b.data * x.data + c.data * x.data**2 + d.data * x.data**3
y_true = np.exp(x.data)
mse = np.mean((y_pred_final - y_true) ** 2)
print(f"\nFinal MSE: {mse:.6f}")
