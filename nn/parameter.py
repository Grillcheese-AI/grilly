"""
Parameter class for storing parameters with gradients (PyTorch-like)

Parameters can have .grad attribute for storing gradients from backward pass.
"""

import numpy as np


class Parameter(np.ndarray):
    """
    A Parameter is a numpy array that can store gradients.

    Similar to torch.nn.Parameter, but based on numpy arrays.
    Gradients are stored in the .grad attribute.
    """

    def __new__(cls, data: np.ndarray, requires_grad: bool = True):
        """
        Create a new Parameter from a numpy array.

        Args:
            data: Numpy array containing parameter values
            requires_grad: Whether this parameter requires gradients (default: True)
        """
        obj = np.asarray(data, dtype=np.float32).view(cls)
        obj.requires_grad = requires_grad
        obj.grad = None
        return obj

    def __array_finalize__(self, obj):
        """Called when creating new arrays from this one"""
        if obj is None:
            return
        self.requires_grad = getattr(obj, "requires_grad", True)
        self.grad = getattr(obj, "grad", None)

    def zero_grad(self):
        """Clear gradients"""
        if self.grad is not None:
            self.grad.fill(0.0)
        else:
            self.grad = np.zeros_like(self, dtype=np.float32)

    def __repr__(self):
        """Return a debug representation."""

        return f"Parameter(shape={self.shape}, requires_grad={self.requires_grad}, grad={'set' if self.grad is not None else 'None'})"


def parameter(data: np.ndarray, requires_grad: bool = True) -> Parameter:
    """
    Create a Parameter from a numpy array.

    Args:
        data: Numpy array
        requires_grad: Whether gradients are needed (default: True)

    Returns:
        Parameter object
    """
    return Parameter(data, requires_grad=requires_grad)
