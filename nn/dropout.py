"""
Dropout module.
Uses: fnn-dropout.glsl
"""

import numpy as np

from .module import Module


class Dropout(Module):
    """
    Dropout layer
    Uses: fnn-dropout.glsl
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        """Initialize the instance."""

        super().__init__()
        self.p = p
        self.inplace = inplace
        self._mask = None  # Store mask from forward pass for backward

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using fnn-dropout.glsl"""
        if not self.training or self.p == 0.0:
            self._mask = None
            return x

        backend = self._get_backend()
        if hasattr(backend, "fnn") and hasattr(backend.fnn, "dropout"):
            try:
                # GPU dropout - need to get mask for backward pass
                # For now, use CPU to get mask, then apply
                mask = np.random.binomial(1, 1 - self.p, size=x.shape).astype(np.float32)
                self._mask = mask  # Save mask for backward pass
                output = x * mask / (1 - self.p)
                return output
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        mask = np.random.binomial(1, 1 - self.p, size=x.shape).astype(np.float32)
        self._mask = mask  # Save mask for backward pass
        return x * mask / (1 - self.p)

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for Dropout.

        Dropout: y = x * mask / (1 - p) during training
        Gradient: grad_input = grad_output * mask / (1 - p)

        Args:
            grad_output: Gradient w.r.t. output
            x: Input from forward pass (not used, but kept for API consistency)

        Returns:
            grad_input: Gradient w.r.t. input
        """
        if not self.training or self.p == 0.0:
            return grad_output

        # Use saved mask from forward pass
        if self._mask is None:
            # If mask wasn't saved (shouldn't happen), return scaled gradient
            return grad_output / (1 - self.p)

        # grad_input = grad_output * mask / (1 - p)
        grad_input = grad_output * self._mask / (1 - self.p)

        return grad_input

    def __repr__(self):
        """Return a debug representation."""

        return f"Dropout(p={self.p})"
