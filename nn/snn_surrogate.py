"""
Surrogate gradient functions for Spiking Neural Networks.

Forward pass uses Heaviside step function (binary spikes).
Backward pass uses smooth surrogate gradients for differentiability.
"""

import numpy as np


class SurrogateFunction:
    """Base class for surrogate gradient functions.

    All surrogate functions use Heaviside in the forward pass
    and a smooth approximation for the backward gradient.
    """

    def __init__(self, alpha=2.0):
        self.alpha = alpha

    def heaviside(self, x):
        """Heaviside step function: 1 if x >= 0, else 0."""
        return (x >= 0).astype(np.float32)

    def forward(self, x):
        """Forward: Heaviside step function."""
        return self.heaviside(x)

    def gradient(self, x):
        """Backward: surrogate gradient (override in subclasses)."""
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha})"


class ATan(SurrogateFunction):
    """ATan surrogate gradient.

    Backward: alpha / (2 * (1 + (pi * alpha * x / 2)^2))
    """

    def __init__(self, alpha=2.0):
        super().__init__(alpha)

    def gradient(self, x):
        return self.alpha / (2.0 * (1.0 + (np.pi * self.alpha * x / 2.0) ** 2))


class Sigmoid(SurrogateFunction):
    """Sigmoid surrogate gradient.

    Backward: alpha * sig(alpha*x) * (1 - sig(alpha*x))
    """

    def __init__(self, alpha=4.0):
        super().__init__(alpha)

    def gradient(self, x):
        sig = 1.0 / (1.0 + np.exp(-self.alpha * x))
        return self.alpha * sig * (1.0 - sig)


class FastSigmoid(SurrogateFunction):
    """Fast Sigmoid surrogate gradient.

    Backward: alpha / (2 * (1 + alpha * |x|)^2)
    """

    def __init__(self, alpha=2.0):
        super().__init__(alpha)

    def gradient(self, x):
        return self.alpha / (2.0 * (1.0 + self.alpha * np.abs(x)) ** 2)
