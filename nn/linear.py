"""
Linear layer — fully-connected layer with optional bias.
Uses: fnn-linear.glsl, fnn-linear-backward.glsl
"""

import numpy as np

from ._helpers import (
    _PARAMETER_AVAILABLE,
    _USE_CPP_BRIDGE,
    ParameterClass,
    _bridge,
    _bridge_to_numpy,
    _get_param_array,
)
from .module import Module


class Linear(Module):
    """
    Linear (fully connected) layer
    Uses: fnn-linear.glsl, fnn-linear-backward.glsl
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """Initialize the instance."""

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # Initialize weights using Xavier initialization (fnn-xavier-init.glsl)
        backend = self._get_backend()
        if (
            hasattr(backend, "fnn")
            and hasattr(backend.fnn, "xavier_init")
            and hasattr(backend, "core")
            and "fnn-xavier-init" in backend.core.shaders
        ):
            try:
                # Use GPU Xavier init
                weight_data = backend.fnn.xavier_init(in_features, out_features)
            except Exception:
                # CPU fallback
                limit = np.sqrt(6.0 / (in_features + out_features))
                weight_data = np.random.uniform(-limit, limit, (out_features, in_features)).astype(
                    np.float32
                )
        else:
            # CPU fallback
            limit = np.sqrt(6.0 / (in_features + out_features))
            weight_data = np.random.uniform(-limit, limit, (out_features, in_features)).astype(
                np.float32
            )

        # Create Parameter objects (support .grad attribute)
        if _PARAMETER_AVAILABLE and ParameterClass is not None:
            self.weight = ParameterClass(weight_data, requires_grad=True)
        else:
            # Fallback: use wrapper class to add .grad attribute
            class ParamWrapper:
                """Lightweight parameter wrapper with gradient storage."""

                def __init__(self, data):
                    """Initialize the wrapped parameter array."""
                    self.data = (
                        data.copy()
                        if isinstance(data, np.ndarray)
                        else np.array(data, dtype=np.float32)
                    )
                    self.grad = None

                def __array__(self):
                    """Expose the wrapped array to numpy operations."""
                    return self.data

                def __getitem__(self, key):
                    """Read parameter slices by index."""
                    return self.data[key]

                def __setitem__(self, key, value):
                    """Write parameter slices by index."""
                    self.data[key] = value

                def __sub__(self, other):
                    """Return elementwise subtraction as a wrapped parameter."""
                    result = self.data - (other.data if hasattr(other, "data") else other)
                    return ParamWrapper(result)

                def __isub__(self, other):
                    """Apply in-place subtraction to the wrapped array."""
                    self.data -= other.data if hasattr(other, "data") else other
                    return self

                def copy(self):
                    """Return a copy of the wrapped parameter."""
                    return ParamWrapper(self.data.copy())

                @property
                def shape(self):
                    """Expose the wrapped array shape."""
                    return self.data.shape

                @property
                def dtype(self):
                    """Expose the wrapped array dtype."""
                    return self.data.dtype

                def zero_grad(self):
                    """Reset gradients to zeros."""
                    if self.grad is not None:
                        self.grad.fill(0.0)
                    else:
                        self.grad = np.zeros_like(self.data, dtype=np.float32)

            self.weight = ParamWrapper(weight_data)

        if bias:
            if _PARAMETER_AVAILABLE and ParameterClass is not None:
                self.bias = ParameterClass(
                    np.zeros(out_features, dtype=np.float32), requires_grad=True
                )
            else:
                # Use same wrapper approach
                class ParamWrapper:
                    """Lightweight bias wrapper with gradient storage."""

                    def __init__(self, data):
                        """Initialize the wrapped bias array."""
                        self.data = (
                            data.copy()
                            if isinstance(data, np.ndarray)
                            else np.array(data, dtype=np.float32)
                        )
                        self.grad = None

                    def __array__(self):
                        """Expose the wrapped array to numpy operations."""
                        return self.data

                    def __getitem__(self, key):
                        """Read bias entries by index."""
                        return self.data[key]

                    def __setitem__(self, key, value):
                        """Write bias entries by index."""
                        self.data[key] = value

                    def __sub__(self, other):
                        """Return elementwise subtraction as a wrapped bias."""
                        result = self.data - (other.data if hasattr(other, "data") else other)
                        return ParamWrapper(result)

                    def __isub__(self, other):
                        """Apply in-place subtraction to the wrapped bias."""
                        self.data -= other.data if hasattr(other, "data") else other
                        return self

                    def copy(self):
                        """Return a copy of the wrapped bias."""
                        return ParamWrapper(self.data.copy())

                    @property
                    def shape(self):
                        """Expose the wrapped array shape."""
                        return self.data.shape

                    @property
                    def dtype(self):
                        """Expose the wrapped array dtype."""
                        return self.data.dtype

                    def zero_grad(self):
                        """Reset gradients to zeros."""
                        if self.grad is not None:
                            self.grad.fill(0.0)
                        else:
                            self.grad = np.zeros_like(self.data, dtype=np.float32)

                self.bias = ParamWrapper(np.zeros(out_features, dtype=np.float32))
        else:
            self.bias = None

        # Register parameters
        self.register_parameter("weight", self.weight)
        if self.bias is not None:
            self.register_parameter("bias", self.bias)

    def forward(self, x) -> np.ndarray:
        """Forward pass — GPU-first, always dispatches through GPU backend."""
        weight = _get_param_array(self.weight)
        bias = _get_param_array(self.bias) if self.bias is not None else None

        # C++ bridge fast path (handles both numpy and VulkanTensor via __array__)
        if _USE_CPP_BRIDGE:
            result = _bridge_to_numpy(_bridge.linear(x, weight, bias))
            if result is not None:
                return result

        # Legacy Python Vulkan backend (fallback)
        backend = self._get_backend()
        if hasattr(backend, "fnn") and hasattr(backend.fnn, "linear"):
            return backend.fnn.linear(
                x,
                weight,
                bias,
                return_gpu_tensor=self._return_gpu_tensor,
            )

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass using fnn-linear-backward.glsl

        Computes gradients and stores them in self.weight.grad and self.bias.grad.

        Args:
            grad_output: Gradient w.r.t. output (batch, out_features)
            x: Input from forward pass (batch, in_features)

        Returns:
            grad_input: Gradient w.r.t. input (batch, in_features)
        """
        backend = self._get_backend()

        # Extract numpy arrays for computation
        weight = _get_param_array(self.weight)
        bias = _get_param_array(self.bias) if self.bias is not None else None

        # Try GPU shader if available (2D only; 3D uses CPU for numerical parity)
        use_gpu = (
            grad_output.ndim == 2
            and hasattr(backend, "fnn")
            and hasattr(backend.fnn, "linear_backward")
        )
        if use_gpu:
            try:
                grad_input, grad_weight, grad_bias = backend.fnn.linear_backward(
                    grad_output, x, weight, bias
                )

                # Store gradients in parameters (from backward pass)
                if self.weight is not None:
                    if not hasattr(self.weight, "grad") or self.weight.grad is None:
                        self.weight.grad = grad_weight
                    else:
                        self.weight.grad += grad_weight

                if self.bias is not None and grad_bias is not None:
                    if not hasattr(self.bias, "grad") or self.bias.grad is None:
                        self.bias.grad = grad_bias
                    else:
                        self.bias.grad += grad_bias

                return grad_input
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        # Handle both 2D and 3D inputs
        grad_output_shape = grad_output.shape

        # Flatten to 2D for gradient computation
        if grad_output.ndim == 3:
            batch, seq, out_features = grad_output.shape
            grad_output_2d = grad_output.reshape(batch * seq, out_features)
            x_2d = x.reshape(batch * seq, x.shape[-1])
        else:
            grad_output_2d = grad_output
            x_2d = x

        grad_input_2d = grad_output_2d @ weight  # (batch*seq, in_features) or (batch, in_features)
        grad_weight = grad_output_2d.T @ x_2d  # (out_features, in_features)
        grad_bias = np.sum(grad_output_2d, axis=0) if bias is not None else None

        # Reshape grad_input back to original shape
        if grad_output.ndim == 3:
            grad_input = grad_input_2d.reshape(grad_output_shape[0], grad_output_shape[1], -1)
        else:
            grad_input = grad_input_2d

        # Store gradients in parameters (from backward pass)
        if self.weight is not None:
            if not hasattr(self.weight, "grad") or self.weight.grad is None:
                self.weight.grad = grad_weight
            else:
                self.weight.grad += grad_weight

        if self.bias is not None and grad_bias is not None:
            if not hasattr(self.bias, "grad") or self.bias.grad is None:
                self.bias.grad = grad_bias
            else:
                self.bias.grad += grad_bias

        return grad_input

    def __repr__(self):
        """Return a debug representation."""

        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
