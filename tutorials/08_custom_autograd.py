"""
Custom Autograd Functions Tutorial
===================================

Create custom differentiable operations using grilly.nn.Function.
Similar to torch.autograd.Function.

This tutorial shows:
1. Creating a custom ReLU function
2. Creating a custom linear function with bias
3. Combining custom functions in a network
"""

import _path_setup  # noqa: F401 - must be first to ensure grilly is importable

import numpy as np

from grilly.nn.autograd import (
    Variable, Function, FunctionCtx,
    tensor, randn, no_grad
)


# ============================================================================
# Example 1: Custom ReLU
# ============================================================================

class MyReLU(Function):
    """
    Custom ReLU implementation.

    Forward: max(0, x)
    Backward: 1 if x > 0, else 0
    """

    @staticmethod
    def forward(ctx: FunctionCtx, x: Variable) -> np.ndarray:
        # Save input for backward pass
        ctx.save_for_backward(x)
        return np.maximum(x.data, 0)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: np.ndarray):
        x, = ctx.saved_tensors
        # Gradient is 1 where input was positive, 0 otherwise
        return grad_output * (x.data > 0).astype(np.float32)


# ============================================================================
# Example 2: Custom Linear Layer
# ============================================================================

class MyLinear(Function):
    """
    Custom linear transformation: y = x @ W.T + b
    """

    @staticmethod
    def forward(ctx: FunctionCtx, x: Variable, weight: Variable, bias: Variable) -> np.ndarray:
        ctx.save_for_backward(x, weight, bias)
        return x.data @ weight.data.T + bias.data

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: np.ndarray):
        x, weight, bias = ctx.saved_tensors

        # Gradient w.r.t. input: grad @ W
        grad_x = grad_output @ weight.data

        # Gradient w.r.t. weight: grad.T @ x
        grad_w = grad_output.T @ x.data

        # Gradient w.r.t. bias: sum over batch
        grad_b = grad_output.sum(axis=0)

        return grad_x, grad_w, grad_b


# ============================================================================
# Example 3: Custom Softplus (smooth ReLU)
# ============================================================================

class MySoftplus(Function):
    """
    Softplus activation: log(1 + exp(x))
    A smooth approximation of ReLU.
    """

    @staticmethod
    def forward(ctx: FunctionCtx, x: Variable, beta: float = 1.0) -> np.ndarray:
        # Numerically stable softplus
        ctx.beta = beta
        ctx.save_for_backward(x)
        # Use stable computation: softplus(x) = x + log(1 + exp(-x)) for x >= 0
        x_data = x.data * beta
        result = np.where(
            x_data >= 0,
            x_data + np.log1p(np.exp(-x_data)),
            np.log1p(np.exp(x_data))
        ) / beta
        return result

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: np.ndarray):
        x, = ctx.saved_tensors
        # d/dx softplus(x) = sigmoid(beta * x)
        sigmoid = 1.0 / (1.0 + np.exp(-ctx.beta * x.data))
        return grad_output * sigmoid


# ============================================================================
# Example 4: Custom Squared Hinge Loss
# ============================================================================

class SquaredHingeLoss(Function):
    """
    Squared hinge loss for binary classification.
    L = mean(max(0, 1 - y * pred)^2)

    Args:
        pred: Predictions (any real number)
        target: Labels in {-1, +1}
    """

    @staticmethod
    def forward(ctx: FunctionCtx, pred: Variable, target: Variable) -> np.ndarray:
        ctx.save_for_backward(pred, target)
        margin = 1 - target.data * pred.data
        hinge = np.maximum(0, margin)
        return np.mean(hinge ** 2)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: np.ndarray):
        pred, target = ctx.saved_tensors
        margin = 1 - target.data * pred.data
        # Gradient: -2 * y * max(0, margin) / n
        n = pred.data.size
        mask = (margin > 0).astype(np.float32)
        grad_pred = grad_output * (-2 * target.data * margin * mask) / n
        return grad_pred, None  # No gradient for target


# ============================================================================
# Demo: Training with Custom Functions
# ============================================================================

def main():
    print("Custom Autograd Functions Tutorial")
    print("=" * 60)

    # Test 1: Custom ReLU
    print("\n1. Custom ReLU")
    x = tensor([-2, -1, 0, 1, 2], requires_grad=True)
    y = MyReLU.apply(x)  # or MyReLU(x)
    print(f"   Input:  {x.data}")
    print(f"   Output: {y.data}")

    # Backward
    loss = y.sum()
    loss.backward()
    print(f"   Gradient: {x.grad}")

    # Test 2: Custom Linear
    print("\n2. Custom Linear Layer")
    np.random.seed(42)
    batch_size, in_features, out_features = 4, 8, 3

    x = randn(batch_size, in_features, requires_grad=True)
    w = randn(out_features, in_features, requires_grad=True)
    b = randn(out_features, requires_grad=True)

    y = MyLinear.apply(x, w, b)
    print(f"   Input shape:  {x.shape}")
    print(f"   Weight shape: {w.shape}")
    print(f"   Output shape: {y.shape}")

    loss = y.sum()
    loss.backward()
    print(f"   x.grad shape: {x.grad.shape}")
    print(f"   w.grad shape: {w.grad.shape}")
    print(f"   b.grad shape: {b.grad.shape}")

    # Test 3: Softplus
    print("\n3. Custom Softplus (smooth ReLU)")
    x = tensor([-2, -1, 0, 1, 2], requires_grad=True)
    y = MySoftplus.apply(x)
    print(f"   Input:  {x.data}")
    print(f"   Output: {np.round(y.data, 4)}")

    loss = y.sum()
    loss.backward()
    print(f"   Gradient: {np.round(x.grad, 4)}")

    # Test 4: Simple 2-layer network with custom functions
    print("\n4. Training 2-Layer Network with Custom Functions")
    print("-" * 60)

    # Generate simple data: y = 2x + 1 (with noise)
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 4).astype(np.float32)
    y_true = (X @ np.array([1, 2, -1, 0.5])).astype(np.float32) + 0.5

    # Initialize weights
    w1 = randn(8, 4, requires_grad=True)
    b1 = randn(8, requires_grad=True)
    w2 = randn(1, 8, requires_grad=True)
    b2 = randn(1, requires_grad=True)

    # Scale weights
    with no_grad():
        w1.data *= 0.1
        w2.data *= 0.1
        b1.data *= 0.0
        b2.data *= 0.0

    learning_rate = 0.01
    initial_loss = None

    for epoch in range(500):
        # Forward pass using custom functions
        x = tensor(X, requires_grad=False)
        target = tensor(y_true.reshape(-1, 1), requires_grad=False)

        h1 = MyLinear.apply(x, w1, b1)
        h1_act = MyReLU.apply(h1)
        pred = MyLinear.apply(h1_act, w2, b2)

        # MSE loss
        diff = pred - target
        loss = (diff * diff).mean()

        if epoch == 0:
            initial_loss = loss.item()

        if epoch % 100 == 0:
            rel_loss = loss.item() / initial_loss
            print(f"   Epoch {epoch:4d}: loss = {loss.item():.6f} (rel: {rel_loss:.4f})")

        # Backward
        w1.zero_grad()
        b1.zero_grad()
        w2.zero_grad()
        b2.zero_grad()
        loss.backward()

        # Update weights
        with no_grad():
            w1.data -= learning_rate * w1.grad
            b1.data -= learning_rate * b1.grad
            w2.data -= learning_rate * w2.grad
            b2.data -= learning_rate * b2.grad

    final_loss = loss.item()
    print(f"\n   Training complete!")
    print(f"   Initial loss: {initial_loss:.6f}")
    print(f"   Final loss:   {final_loss:.6f}")
    print(f"   Improvement:  {(1 - final_loss/initial_loss)*100:.1f}%")

    # Test 5: Squared Hinge Loss for binary classification
    print("\n5. Squared Hinge Loss for Binary Classification")
    print("-" * 60)

    # Generate binary classification data
    np.random.seed(123)
    X = np.random.randn(50, 2).astype(np.float32)
    # Simple linear boundary
    y_true = np.sign(X[:, 0] + X[:, 1]).astype(np.float32)  # {-1, +1}

    # Simple linear classifier
    w = randn(2, requires_grad=True)
    w.data *= 0.1

    learning_rate = 0.1
    initial_loss = None

    for epoch in range(200):
        x = tensor(X, requires_grad=False)
        target = tensor(y_true, requires_grad=False)

        # Linear prediction
        pred = (x * w).sum(dim=1)

        # Custom hinge loss
        loss = SquaredHingeLoss.apply(pred, target)

        if epoch == 0:
            initial_loss = loss.item()

        if epoch % 50 == 0:
            # Compute accuracy
            pred_labels = np.sign(pred.data)
            accuracy = (pred_labels == y_true).mean() * 100
            print(f"   Epoch {epoch:4d}: loss = {loss.item():.6f}, accuracy = {accuracy:.1f}%")

        w.zero_grad()
        loss.backward()

        with no_grad():
            w.data -= learning_rate * w.grad

    pred_labels = np.sign((X * w.data).sum(axis=1))
    final_accuracy = (pred_labels == y_true).mean() * 100
    print(f"\n   Final accuracy: {final_accuracy:.1f}%")
    print(f"   Learned weights: {w.data}")

    print("\n" + "=" * 60)
    print("Custom autograd functions work correctly!")


if __name__ == "__main__":
    main()
