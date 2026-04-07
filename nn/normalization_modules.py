"""
Normalization modules — LayerNorm, RMSNorm.
Note: BatchNorm1d/BatchNorm2d live in nn/normalization.py (already split).
Uses: fnn-layernorm.glsl, rms-norm.glsl
"""

import numpy as np

from ._helpers import (
    _USE_CPP_BRIDGE,
    _bridge,
    _bridge_to_numpy,
    _create_param_wrapper,
    _get_param_array,
)
from .module import Module


class LayerNorm(Module):
    """
    Layer Normalization
    Uses: fnn-layernorm.glsl
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """Initialize the instance."""

        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Create learnable parameters
        self.weight = _create_param_wrapper(np.ones(normalized_shape, dtype=np.float32))
        self.bias = _create_param_wrapper(np.zeros(normalized_shape, dtype=np.float32))

        # Register parameters
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def forward(self, x):
        """Forward pass using fnn-layernorm.glsl (GPU-first).

        Autograd: when ``x`` is a ``Variable``, the output is wrapped in a
        ``Variable`` whose ``GradFn`` calls ``self.backward(grad_output, x)``
        on loss.backward(). ``self.backward`` already populates
        ``self.weight.grad`` and ``self.bias.grad`` in-place, so AdamW picks
        them up through ``param.grad``. Mirrors ``nn.Linear.forward``'s
        autograd wiring — see that file for the long-form rationale.
        """
        try:
            from grilly.nn.autograd import GradFn as _GradFn
            from grilly.nn.autograd import Variable as _Variable
            from grilly.nn.autograd import _grad_enabled
        except ImportError:
            _Variable = None  # type: ignore[assignment]
            _GradFn = None  # type: ignore[assignment]
            _grad_enabled = False

        x_var = None
        if _Variable is not None and isinstance(x, _Variable):
            x_var = x
            x_data = x.data
        else:
            x_data = x

        weight = _get_param_array(self.weight)
        bias = _get_param_array(self.bias)

        # ---- Existing forward path ----
        result = None
        if _USE_CPP_BRIDGE:
            result = _bridge_to_numpy(_bridge.layernorm(x_data, weight, bias, self.eps))

        if result is None:
            backend = self._get_backend()
            if backend is not None and hasattr(backend, "fnn"):
                result = backend.fnn.layernorm(
                    x_data,
                    weight,
                    bias,
                    eps=self.eps,
                    return_gpu_tensor=self._return_gpu_tensor,
                )
            else:
                # CPU fallback — matches self.backward() math
                mean = np.mean(x_data, axis=-1, keepdims=True)
                var = np.var(x_data, axis=-1, keepdims=True)
                normalized = (x_data - mean) / np.sqrt(var + self.eps)
                result = normalized * weight + bias

        # ---- Autograd wiring ----
        if (
            x_var is not None
            and _GradFn is not None
            and _grad_enabled
            and isinstance(result, np.ndarray)
            and not isinstance(result, _Variable)
        ):
            x_data_for_backward = x_data

            def backward_fn(grad_output):
                grad_input = self.backward(np.asarray(grad_output), x_data_for_backward)
                return (grad_input,)

            grad_fn = _GradFn("LayerNorm", backward_fn, [x_var])
            return _Variable(np.asarray(result), requires_grad=True, grad_fn=grad_fn)

        return result

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for LayerNorm.

        LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias

        Gradients:
        - grad_weight = sum(grad_output * normalized_x, dim=normalized_dims)
        - grad_bias = sum(grad_output, dim=normalized_dims)
        - grad_input = grad_output * weight / sqrt(var + eps) - mean(grad_output * weight) / N
                       - normalized_x * mean(grad_output * weight * normalized_x) / N

        Args:
            grad_output: Gradient w.r.t. output (same shape as x)
            x: Input from forward pass (required for LayerNorm backward)

        Returns:
            grad_input: Gradient w.r.t. input (same shape as x)
        """
        if x is None:
            raise ValueError("Input x is required for LayerNorm backward pass")
        # Compute mean and variance
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        normalized_x = (x - mean) / std

        # Get weight
        weight = _get_param_array(self.weight)

        # Compute gradients w.r.t. weight and bias
        # Sum over all dimensions except the last (normalized dimension)
        reduce_dims = tuple(range(len(x.shape) - 1))

        if self.weight is not None:
            grad_weight = np.sum(grad_output * normalized_x, axis=reduce_dims)
            if not hasattr(self.weight, "grad") or self.weight.grad is None:
                self.weight.grad = grad_weight
            else:
                self.weight.grad += grad_weight

        if self.bias is not None:
            grad_bias = np.sum(grad_output, axis=reduce_dims)
            if not hasattr(self.bias, "grad") or self.bias.grad is None:
                self.bias.grad = grad_bias
            else:
                self.bias.grad += grad_bias

        # Compute gradient w.r.t. input
        # grad_input = (grad_output * weight) / std - mean((grad_output * weight) / std) / N
        #            - normalized_x * mean((grad_output * weight) * normalized_x) / N
        x.shape[-1]
        grad_weighted = grad_output * weight
        grad_scaled = grad_weighted / std

        # First term: grad_scaled
        grad_input = grad_scaled

        # Second term: subtract mean of grad_scaled
        grad_input = grad_input - np.mean(grad_scaled, axis=-1, keepdims=True)

        # Third term: subtract normalized_x * mean(grad_weighted * normalized_x)
        grad_norm = np.mean(grad_weighted * normalized_x, axis=-1, keepdims=True)
        grad_input = grad_input - normalized_x * grad_norm

        return grad_input

    def __repr__(self):
        """Return a debug representation."""

        return f"LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps})"


class RMSNorm(Module):
    """
    RMS Normalization layer.
    Unlike LayerNorm, RMSNorm has no mean subtraction and no bias.
    Used by LLaMA, Mistral, T5, Qwen architectures.
    Uses: rms-norm.glsl
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """Initialize the instance."""

        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        weight_data = np.ones(normalized_shape, dtype=np.float32)
        self.weight = _create_param_wrapper(weight_data)
        self.register_parameter("weight", self.weight)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using rms-norm.glsl (GPU-first)"""
        weight = _get_param_array(self.weight)

        # Try C++ bridge fast path
        if _USE_CPP_BRIDGE:
            result = _bridge_to_numpy(_bridge.rmsnorm(x, weight, self.eps))
            if result is not None:
                return result

        # CPU fallback
        x = np.asarray(x, dtype=np.float32)
        mean_sq = np.mean(x**2, axis=-1, keepdims=True)
        normed = x * (1.0 / np.sqrt(mean_sq + self.eps))
        return normed * weight

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """Backward pass for RMSNorm."""
        weight = _get_param_array(self.weight)
        x = np.asarray(x, dtype=np.float32)

        # Recompute forward intermediates
        mean_sq = np.mean(x**2, axis=-1, keepdims=True)
        inv_rms = 1.0 / np.sqrt(mean_sq + self.eps)
        normed = x * inv_rms

        # Gradient w.r.t. weight
        grad_weight = np.sum(grad_output * normed, axis=tuple(range(grad_output.ndim - 1)))
        if hasattr(self.weight, "grad") and self.weight.grad is not None:
            self.weight.grad += grad_weight
        elif hasattr(self.weight, "grad"):
            self.weight.grad = grad_weight

        # Gradient w.r.t. input
        grad_normed = grad_output * weight
        normed.shape[-1]
        grad_input = inv_rms * (
            grad_normed - normed * np.mean(grad_normed * normed, axis=-1, keepdims=True)
        )

        return grad_input

    def __repr__(self):
        """Return a debug representation."""

        return f"RMSNorm(normalized_shape={self.normalized_shape}, eps={self.eps})"
