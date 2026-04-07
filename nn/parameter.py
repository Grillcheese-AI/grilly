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
        if hasattr(data, "data") and not isinstance(data, np.ndarray):
            data = getattr(data, "data", data)
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

    def uniform_(self, a: float = 0.0, b: float = 1.0) -> "Parameter":
        """In-place uniform fill (PyTorch ``Tensor.uniform_``)."""
        self[:] = np.random.uniform(a, b, self.shape).astype(np.float32)
        return self

    def zero_(self) -> "Parameter":
        """Fill with zeros (PyTorch ``Tensor.zero_``)."""
        self.fill(0.0)
        return self

    # ------------------------------------------------------------------
    # PyTorch-style shape methods so user ``forward`` code can do
    # ``self.weight.unsqueeze(0)`` / ``.view(...)`` / ``.mean(dim=...)``
    # without having to know that Parameter is a numpy ndarray subclass.
    # ------------------------------------------------------------------
    def unsqueeze(self, dim: int) -> np.ndarray:
        """Insert a length-1 axis at *dim* (PyTorch ``Tensor.unsqueeze``)."""
        return np.expand_dims(np.asarray(self), dim)

    def view(self, *shape) -> np.ndarray:
        """Alias for ``reshape`` matching PyTorch's ``Tensor.view`` signature."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return np.asarray(self).reshape(shape)

    def mean(self, dim=None, keepdims: bool = False, **kwargs):  # type: ignore[override]
        """``Tensor.mean(dim=...)`` — accepts both ``dim=`` (torch) and
        ``axis=`` (numpy) so it composes cleanly with ``np.mean(p, axis=...)``.
        """
        axis = kwargs.pop("axis", dim)
        if kwargs:
            # Forward any remaining numpy kwargs (dtype, out, where, ...).
            return np.asarray(self).mean(axis=axis, keepdims=keepdims, **kwargs)
        return np.asarray(self).mean(axis=axis, keepdims=keepdims)

    def detach(self) -> "Parameter":
        """Return a Parameter detached from any (future) autograd graph."""
        return Parameter(np.array(self, copy=True), requires_grad=False)

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
