"""
Neural Network Modules (PyTorch-like)
All modules use GPU-accelerated Vulkan shaders
"""

import os

import numpy as np

from .module import Module

# Try to import Parameter class
try:
    from .parameter import Parameter as ParameterClass

    _PARAMETER_AVAILABLE = True
except ImportError:
    _PARAMETER_AVAILABLE = False
    ParameterClass = None


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
        """Forward pass with GEMM fast path if available."""
        backend = self._get_backend()
        weight = _get_param_array(self.weight)
        bias = _get_param_array(self.bias) if self.bias is not None else None

        from ..utils.tensor_conversion import VulkanTensor

        is_vt = isinstance(x, VulkanTensor)

        if not is_vt:
            x = np.asarray(x, dtype=np.float32)
        weight = np.asarray(weight, dtype=np.float32)

        # Use fnn.linear() which handles x @ W^T + bias in a single dispatch
        # without needing a CPU weight transpose copy
        if hasattr(backend, "fnn") and hasattr(backend.fnn, "linear"):
            try:
                return backend.fnn.linear(
                    x,
                    weight,
                    bias,
                    return_gpu_tensor=self._return_gpu_tensor,
                )
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback (force numpy for CPU path)
        if is_vt:
            x = x.numpy()

        # x: (batch, in_features) or (batch, seq, in_features)
        if x.ndim == 2:
            batch_seq, in_features = x.shape
            x_2d = x
        elif x.ndim == 3:
            b, s, in_features = x.shape
            batch_seq = b * s
            x_2d = x.reshape(batch_seq, in_features)
        else:
            raise ValueError(f"Linear expects 2D or 3D input, got shape {x.shape}")

        out_features = self.out_features

        out_2d = x_2d @ weight.T
        if bias is not None:
            out_2d += bias.reshape(1, out_features)

        # Reshape back to original batch shape
        if x.ndim == 2:
            return out_2d
        else:
            return out_2d.reshape(b, s, out_features)

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


# Helper function to create Parameter wrapper
def _get_param_array(param) -> np.ndarray:
    """
    Extract numpy array from a parameter (ParamWrapper, Parameter, or numpy array).

    Handles:
    - ParamWrapper: returns .data (numpy array)
    - Parameter (np.ndarray subclass): returns the array directly
    - numpy array: returns directly
    - memoryview: converts to numpy array
    """
    if isinstance(param, np.ndarray):
        # Parameter is a numpy subclass, or plain numpy array
        return param
    elif hasattr(param, "data") and not isinstance(param.data, memoryview):
        # ParamWrapper with .data as numpy array
        return param.data
    elif hasattr(param, "__array__"):
        # Has __array__ method
        return np.asarray(param)
    else:
        return np.asarray(param)


def _create_param_wrapper(data: np.ndarray):
    """Create a Parameter wrapper with .grad support"""
    if _PARAMETER_AVAILABLE and ParameterClass is not None:
        return ParameterClass(data, requires_grad=True)
    else:

        class ParamWrapper:
            """Fallback parameter wrapper used when Parameter is unavailable."""

            def __init__(self, data):
                """Initialize the wrapped array."""
                # Ensure data is a numpy array
                if isinstance(data, np.ndarray):
                    self.data = data.copy()
                elif hasattr(data, "__array__"):
                    self.data = np.array(data, dtype=np.float32)
                else:
                    self.data = np.array(data, dtype=np.float32)
                # Ensure it's contiguous and writable
                if not self.data.flags["C_CONTIGUOUS"]:
                    self.data = np.ascontiguousarray(self.data)
                self.grad = None

            def __array__(self):
                """Expose the wrapped array to numpy operations."""
                return self.data

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                """Delegate numpy ufuncs to the wrapped array."""
                # Delegate to numpy
                return getattr(ufunc, method)(*inputs, **kwargs)

            def __getitem__(self, key):
                """Read wrapped values by index."""
                return self.data[key]

            def __setitem__(self, key, value):
                """Write wrapped values by index."""
                self.data[key] = value

            def __sub__(self, other):
                """Return elementwise subtraction as a wrapped value."""
                result = self.data - (other.data if hasattr(other, "data") else other)
                return ParamWrapper(result)

            def __isub__(self, other):
                """Apply in-place subtraction to wrapped values."""
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

        return ParamWrapper(data)


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

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using fnn-layernorm.glsl"""
        backend = self._get_backend()
        weight = _get_param_array(self.weight)
        bias = _get_param_array(self.bias)
        return backend.fnn.layernorm(
            x,
            weight,
            bias,
            eps=self.eps,
            return_gpu_tensor=self._return_gpu_tensor,
        )

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for LayerNorm.

        LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias

        Gradients:
        - grad_weight = sum(grad_output * normalized_x, dim=normalized_dims)
        - grad_bias = sum(grad_output, dim=normalized_dims)
        - grad_input = grad_output * weight / sqrt(var + eps) - mean(grad_output * weight) / N - normalized_x * mean(grad_output * weight * normalized_x) / N

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


class ReLU(Module):
    """
    ReLU activation: max(0, x)
    Uses: activation-relu.glsl
    """

    def __init__(self, inplace: bool = False):
        """Initialize the instance."""

        super().__init__()
        self.inplace = inplace

    def forward(self, x) -> np.ndarray:
        """Forward pass using activation-relu.glsl"""
        backend = self._get_backend()
        return backend.activation_relu(x, return_gpu_tensor=self._return_gpu_tensor)

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for ReLU.

        Args:
            grad_output: Gradient w.r.t. output
            x: Input from forward pass

        Returns:
            grad_input: Gradient w.r.t. input
        """
        # ReLU backward: gradient is 0 where x < 0, else grad_output
        return grad_output * (x > 0).astype(np.float32)

    def __repr__(self):
        """Return a debug representation."""

        return "ReLU()"


class GELU(Module):
    """
    GELU activation
    Uses: activation-gelu.glsl
    """

    def forward(self, x) -> np.ndarray:
        """Forward pass using activation-gelu.glsl"""
        backend = self._get_backend()
        return backend.activation_gelu(x, return_gpu_tensor=self._return_gpu_tensor)

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for GELU.

        Args:
            grad_output: Gradient w.r.t. output
            x: Input from forward pass (required for GELU backward)

        Returns:
            grad_input: Gradient w.r.t. input
        """
        if x is None:
            raise ValueError("Input x is required for GELU backward pass")

        # GELU backward: d/dx GELU(x) = 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) +
        #                 0.5 * x * (1 - tanh^2(...)) * sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        backend = self._get_backend()
        if hasattr(backend, "activation_gelu_backward"):
            try:
                return backend.activation_gelu_backward(grad_output, x)
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        sqrt_2_pi = np.sqrt(2.0 / np.pi)
        coeff = 0.044715
        x_cubed = x**3
        inner = sqrt_2_pi * (x + coeff * x_cubed)
        tanh_inner = np.tanh(inner)
        gelu_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - tanh_inner**2) * sqrt_2_pi * (
            1.0 + 3.0 * coeff * x**2
        )
        return grad_output * gelu_grad

    def __repr__(self):
        """Return a debug representation."""

        return "GELU()"


class SiLU(Module):
    """
    SiLU (Swish) activation: x * sigmoid(x)
    Uses: activation-silu.glsl
    """

    def forward(self, x) -> np.ndarray:
        """Forward pass using activation-silu.glsl"""
        backend = self._get_backend()
        return backend.activation_silu(x, return_gpu_tensor=self._return_gpu_tensor)

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for SiLU.

        Args:
            grad_output: Gradient w.r.t. output
            x: Input from forward pass

        Returns:
            grad_input: Gradient w.r.t. input
        """
        # SiLU backward: d/dx (x * sigmoid(x)) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        sigmoid_x = 1.0 / (1.0 + np.exp(-x))
        silu_grad = sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))
        return grad_output * silu_grad

    def __repr__(self):
        """Return a debug representation."""

        return "SiLU()"


class GCU(Module):
    """
    GCU (Growing Cosine Unit) activation: x * cos(x)
    Uses: activation-gcu.glsl

    Oscillatory activation function for neuromorphic systems.
    Enables single neurons to learn complex patterns like XOR.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using activation-gcu.glsl"""
        backend = self._get_backend()
        return backend.activation_gcu(x)

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for GCU.

        Args:
            grad_output: Gradient w.r.t. output
            x: Input from forward pass

        Returns:
            grad_input: Gradient w.r.t. input
        """
        # GCU backward: d/dx (x * cos(x)) = cos(x) - x * sin(x)
        backend = self._get_backend()
        return backend.activation_gcu_backward(grad_output, x)

    def __repr__(self):
        """Return a debug representation."""

        return "GCU()"


class RoSwish(Module):
    """
    RoSwish (Rotating Swish) activation: (x + α) * sigmoid(β * x) - 0.5 * α
    Uses: activation-roswish.glsl

    Learnable activation with adaptive gating.
    Shows 6-30% improvement over ReLU/Swish on diverse tasks.

    Args:
        alpha_init: Initial rotation parameter (default: 1.0)
        beta_init: Initial gating parameter (default: 1.0)
        learnable: Whether α and β are learnable (default: True)
    """

    def __init__(self, alpha_init: float = 1.0, beta_init: float = 1.0, learnable: bool = True):
        """Initialize the instance."""

        super().__init__()
        self.learnable = learnable

        if learnable and _PARAMETER_AVAILABLE and ParameterClass is not None:
            # Create learnable parameters
            self.alpha = ParameterClass(
                np.array([alpha_init], dtype=np.float32), requires_grad=True
            )
            self.beta = ParameterClass(np.array([beta_init], dtype=np.float32), requires_grad=True)
        else:
            # Fixed parameters
            self.alpha = np.array([alpha_init], dtype=np.float32)
            self.beta = np.array([beta_init], dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using activation-roswish.glsl"""
        backend = self._get_backend()

        # Extract scalar values from parameters
        alpha_val = float(self.alpha[0] if hasattr(self.alpha, "__getitem__") else self.alpha)
        beta_val = float(self.beta[0] if hasattr(self.beta, "__getitem__") else self.beta)

        return backend.activation_roswish(x, alpha=alpha_val, beta=beta_val)

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for RoSwish.

        Args:
            grad_output: Gradient w.r.t. output
            x: Input from forward pass

        Returns:
            grad_input: Gradient w.r.t. input
        """
        backend = self._get_backend()

        # Extract scalar values
        alpha_val = float(self.alpha[0] if hasattr(self.alpha, "__getitem__") else self.alpha)
        beta_val = float(self.beta[0] if hasattr(self.beta, "__getitem__") else self.beta)

        grad_input = backend.activation_roswish_backward(
            grad_output, x, alpha=alpha_val, beta=beta_val
        )

        # Compute gradients w.r.t. parameters if learnable
        if self.learnable and hasattr(self.alpha, "grad"):
            # d/dα RoSwish = sigmoid(β*x) - 0.5
            # d/dβ RoSwish = (x + α) * x * sigmoid(β*x) * (1 - sigmoid(β*x))
            beta_x = beta_val * x
            sigmoid_bx = 1.0 / (1.0 + np.exp(-beta_x))

            # Gradient w.r.t. α
            grad_alpha = grad_output * (sigmoid_bx - 0.5)
            if self.alpha.grad is None:
                self.alpha.grad = np.sum(grad_alpha).reshape(1).astype(np.float32)
            else:
                self.alpha.grad += np.sum(grad_alpha).reshape(1).astype(np.float32)

            # Gradient w.r.t. β
            grad_beta = grad_output * (alpha_val + x) * x * sigmoid_bx * (1.0 - sigmoid_bx)
            if self.beta.grad is None:
                self.beta.grad = np.sum(grad_beta).reshape(1).astype(np.float32)
            else:
                self.beta.grad += np.sum(grad_beta).reshape(1).astype(np.float32)

        return grad_input

    def __repr__(self):
        """Return a debug representation."""

        alpha_val = float(self.alpha[0] if hasattr(self.alpha, "__getitem__") else self.alpha)
        beta_val = float(self.beta[0] if hasattr(self.beta, "__getitem__") else self.beta)
        return f"RoSwish(alpha={alpha_val:.3f}, beta={beta_val:.3f}, learnable={self.learnable})"


class SwiGLU(Module):
    """
    SwiGLU (Swish-Gated Linear Unit) activation
    Uses: activation-swiglu.glsl

    Used in LLaMA, PaLM, Mistral transformer FFN layers.
    Provides 5-15% perplexity improvement over GELU/ReLU.

    Input shape: (..., 2*hidden_dim)
    Output shape: (..., hidden_dim)

    The input is split into two parts [x1, x2], then output = x1 * silu(x2)
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass using activation-swiglu.glsl

        Args:
            x: Input array of shape (..., 2*hidden_dim)

        Returns:
            Output array of shape (..., hidden_dim)
        """
        if x.shape[-1] % 2 != 0:
            raise ValueError(f"SwiGLU input last dimension must be even, got {x.shape[-1]}")

        backend = self._get_backend()
        return backend.activation_swiglu(x)

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for SwiGLU.

        Args:
            grad_output: Gradient w.r.t. output (shape: (..., hidden_dim))
            x: Input from forward pass (shape: (..., 2*hidden_dim))

        Returns:
            grad_input: Gradient w.r.t. input (shape: (..., 2*hidden_dim))
        """
        backend = self._get_backend()
        return backend.activation_swiglu_backward(grad_output, x)

    def __repr__(self):
        """Return a debug representation."""

        return "SwiGLU()"


class Softmax(Module):
    """
    Softmax activation
    Uses: activation-softmax.glsl
    """

    def __init__(self, dim: int = -1):
        """Initialize the instance."""

        super().__init__()
        self.dim = dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using activation-softmax.glsl"""
        backend = self._get_backend()
        return backend.activation_softmax(x, dim=self.dim)

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for Softmax.

        Args:
            grad_output: Gradient w.r.t. output
            x: Input from forward pass

        Returns:
            grad_input: Gradient w.r.t. input
        """
        # Softmax backward: grad_input = softmax(x) * (grad_output - sum(grad_output * softmax(x)))
        softmax_x = self.forward(x)
        grad_input = softmax_x * (
            grad_output - np.sum(grad_output * softmax_x, axis=self.dim, keepdims=True)
        )
        return grad_input

    def __repr__(self):
        """Return a debug representation."""

        return f"Softmax(dim={self.dim})"


class Softplus(Module):
    """
    Softplus activation: log(1 + exp(x))
    Uses: activation-softplus.glsl
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using activation-softplus.glsl"""
        backend = self._get_backend()
        if hasattr(backend, "fnn") and hasattr(backend.fnn, "activation_softplus"):
            try:
                return backend.fnn.activation_softplus(x)
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        return np.log(1.0 + np.exp(x))

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for Softplus.

        Args:
            grad_output: Gradient w.r.t. output
            x: Input from forward pass

        Returns:
            grad_input: Gradient w.r.t. input
        """
        # Softplus backward: d/dx log(1 + exp(x)) = sigmoid(x) = 1 / (1 + exp(-x))
        sigmoid_x = 1.0 / (1.0 + np.exp(-x))
        return grad_output * sigmoid_x

    def __repr__(self):
        """Return a debug representation."""

        return "Softplus()"


class Embedding(Module):
    """
    Embedding layer
    Uses: embedding-lookup.glsl
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        """Initialize the instance."""

        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize embeddings (normal distribution)
        embedding_data = np.random.normal(0, 1, (num_embeddings, embedding_dim)).astype(np.float32)
        self.weight = _create_param_wrapper(embedding_data)

        # Register parameter
        self.register_parameter("weight", self.weight)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using embedding-lookup.glsl"""
        backend = self._get_backend()
        weight = _get_param_array(self.weight)

        gpu_lookup_enabled = os.getenv("GRILLY_EMBEDDING_GPU_LOOKUP", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        if gpu_lookup_enabled and hasattr(backend, "learning") and hasattr(
            backend.learning, "embedding_lookup"
        ):
            try:
                return backend.learning.embedding_lookup(
                    x,
                    weight,
                    return_gpu_tensor=self._return_gpu_tensor,
                )
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        if isinstance(x, np.ndarray):
            return weight[x.astype(np.int32)]
        return weight[x]

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for Embedding.

        Args:
            grad_output: Gradient w.r.t. output (batch, seq_len, embedding_dim)
            x: Input token IDs (batch, seq_len)

        Returns:
            grad_input: Gradient w.r.t. input (usually None for embedding indices)
        """
        if x is None:
            raise ValueError("Input token IDs x are required for Embedding backward pass")

        # Embedding backward: accumulate gradients into weight matrix
        if self.weight is not None:
            backend = self._get_backend()

            # Try GPU-accelerated backward if available
            if hasattr(backend, "learning") and hasattr(backend.learning, "embedding_backward"):
                try:
                    grad_weight = backend.learning.embedding_backward(
                        grad_output, x, self.num_embeddings, self.embedding_dim
                    )

                    # Store gradients in parameter
                    if not hasattr(self.weight, "grad") or self.weight.grad is None:
                        self.weight.grad = grad_weight
                    else:
                        self.weight.grad += grad_weight

                    # No gradient w.r.t. input (token IDs are discrete)
                    return None
                except Exception:
                    pass  # Fall back to CPU

            # CPU fallback: accumulate gradients for each token
            grad_weight = np.zeros_like(_get_param_array(self.weight))

            x_flat = x.flatten()
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1])

            for i, token_id in enumerate(x_flat):
                if 0 <= token_id < self.num_embeddings:
                    grad_weight[int(token_id)] += grad_flat[i]

            if not hasattr(self.weight, "grad") or self.weight.grad is None:
                self.weight.grad = grad_weight
            else:
                self.weight.grad += grad_weight

        # No gradient w.r.t. input (token IDs are discrete)
        return None

    def __repr__(self):
        """Return a debug representation."""

        return (
            f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"
        )


_FUSED_ACTIVATION_MAP = {
    "ReLU": "fused_linear_relu",
    "GELU": "fused_linear_gelu",
    "SiLU": "fused_linear_silu",
}


class Sequential(Module):
    """
    Sequential container for modules

    Caches intermediate activations during forward pass for efficient backward pass.
    Automatically fuses Linear+Activation pairs when fused GPU shaders are available.
    """

    def __init__(self, *modules):
        """Initialize the instance."""

        super().__init__()
        for i, module in enumerate(modules):
            self._modules[str(i)] = module
        self._cached_activations = []  # Cache intermediate activations
        self._fusion_plan = None  # Lazily computed

    def _compute_fusion_plan(self):
        """Scan module list for fusible Linear → Activation pairs.

        Returns a list of tuples:
          ('fuse', linear_idx, act_idx, fused_method_name)  — fused pair
          ('run', idx)                                       — run module normally
        """
        modules_list = list(self._modules.values())
        n = len(modules_list)
        plan = []
        i = 0
        while i < n:
            if (
                i + 1 < n
                and isinstance(modules_list[i], Linear)
                and type(modules_list[i + 1]).__name__ in _FUSED_ACTIVATION_MAP
            ):
                method_name = _FUSED_ACTIVATION_MAP[type(modules_list[i + 1]).__name__]
                plan.append(("fuse", i, i + 1, method_name))
                i += 2
            else:
                plan.append(("run", i))
                i += 1
        return plan

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with automatic Linear+Activation fusion."""
        # Clear cached activations
        self._cached_activations = [x]  # Store initial input

        modules_list = list(self._modules.values())

        # Lazily compute fusion plan (invalidated if modules change)
        if self._fusion_plan is None or len(modules_list) != sum(
            2 if s[0] == "fuse" else 1 for s in self._fusion_plan
        ):
            self._fusion_plan = self._compute_fusion_plan()

        current = x
        for step in self._fusion_plan:
            if step[0] == "fuse":
                _, lin_idx, act_idx, method_name = step
                linear_mod = modules_list[lin_idx]
                weight = _get_param_array(linear_mod.weight)
                bias = _get_param_array(linear_mod.bias) if linear_mod.bias is not None else None
                backend = linear_mod._get_backend()
                fused_fn = getattr(backend.fnn, method_name, None)
                if fused_fn is not None:
                    try:
                        current = fused_fn(
                            current,
                            weight,
                            bias,
                            return_gpu_tensor=linear_mod._return_gpu_tensor,
                        )
                        # Push two entries for backward indexing consistency
                        self._cached_activations.append(current)  # output of Linear
                        self._cached_activations.append(current)  # output of Activation
                        continue
                    except Exception:
                        pass  # Fall back to sequential execution
                # Fallback: run both modules individually
                current = modules_list[lin_idx](current)
                self._cached_activations.append(current)
                current = modules_list[act_idx](current)
                self._cached_activations.append(current)
            else:
                _, idx = step
                current = modules_list[idx](current)
                self._cached_activations.append(current)

        return current

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass through all modules in reverse order.

        Uses cached activations from forward pass.

        Args:
            grad_output: Gradient w.r.t. output
            x: Original input (optional, uses cached if available)

        Returns:
            grad_input: Gradient w.r.t. input
        """
        # Use cached activations if available
        if len(self._cached_activations) == 0:
            # If no cached activations, we can't do proper backward
            # This shouldn't happen if forward was called first
            raise RuntimeError("No cached activations found. Call forward() before backward().")

        grad = grad_output
        modules_list = list(self._modules.values())

        # Backward through modules in reverse order
        # cached_activations[0] is input, cached_activations[i+1] is output of module i
        for i in range(len(modules_list) - 1, -1, -1):
            module = modules_list[i]
            module_input = self._cached_activations[i]  # Input to module i
            self._cached_activations[i + 1]  # Output of module i

            if hasattr(module, "backward"):
                try:
                    # Pass both input and output for backward (some modules need both)
                    grad = module.backward(grad, module_input)
                except TypeError:
                    # Some backward methods only take grad_output
                    grad = module.backward(grad)
                except NotImplementedError:
                    # If backward not implemented, just pass through
                    pass

        return grad

    def __repr__(self):
        """Return a debug representation."""

        modules_str = ",\n  ".join([str(m) for m in self._modules.values()])
        return f"Sequential(\n  {modules_str}\n)"


class Residual(Module):
    """
    Residual connection: output = input + module(input)
    Uses: fnn-residual.glsl
    """

    def __init__(self, module: Module):
        """Initialize the instance."""

        super().__init__()
        self.module = module
        self._modules["module"] = module
        self._cached_input = None
        self._cached_module_output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: x + module(x)"""
        backend = self._get_backend()
        module_out = self.module(x)

        # Cache input for backward pass
        self._cached_input = x
        self._cached_module_output = module_out

        # Try GPU shader if available
        if hasattr(backend, "fnn") and hasattr(backend.fnn, "residual"):
            try:
                return backend.fnn.residual(x, module_out)
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        return x + module_out

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for Residual.

        Residual: output = input + module(input)
        Gradient: grad_input = grad_output + grad_module(input)

        Args:
            grad_output: Gradient w.r.t. output
            x: Input from forward pass (optional, uses cached if available)

        Returns:
            grad_input: Gradient w.r.t. input
        """
        # Use cached input if available
        if x is None:
            x = self._cached_input
            if x is None:
                raise ValueError("Input x is required for Residual backward pass")

        # Residual backward: grad_input = grad_output + grad_module
        # The gradient flows through both the residual connection and the module
        if hasattr(self.module, "backward"):
            try:
                grad_module = self.module.backward(grad_output, x)
                return grad_output + grad_module
            except (TypeError, NotImplementedError):
                # If backward not properly implemented, just pass through residual gradient
                return grad_output
        return grad_output

    def __repr__(self):
        """Return a debug representation."""

        return f"Residual({self.module})"


class MultiheadAttention(Module):
    """
    Multi-head attention
    Uses: attention-scores.glsl, attention-output.glsl, attention-concat-heads.glsl
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """Initialize the instance."""

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Create projection layers
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

        self._modules["q_proj"] = self.q_proj
        self._modules["k_proj"] = self.k_proj
        self._modules["v_proj"] = self.v_proj
        self._modules["out_proj"] = self.out_proj

        # Initialize cached values for backward pass
        self._cached_query = None
        self._cached_key = None
        self._cached_value = None
        self._cached_mask = None
        self._cached_q = None
        self._cached_k = None
        self._cached_v = None
        self._cached_scores = None
        self._cached_scores_pre_softmax = None
        self._cached_attn_output = None

    def forward(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for multi-head attention.

        Args:
            query: Query tensor (batch, seq_len_q, embed_dim)
            key: Key tensor (batch, seq_len_k, embed_dim)
            value: Value tensor (batch, seq_len_k, embed_dim)
            mask: Optional attention mask

        Returns:
            (output, attention_weights)
        """
        # Cache inputs for backward pass
        self._cached_query = query
        self._cached_key = key
        self._cached_value = value
        self._cached_mask = mask

        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Cache Q, K, V for backward
        self._cached_q = q
        self._cached_k = k
        self._cached_v = v

        # Reshape for multi-head attention
        batch_size, seq_len_q, _ = q.shape
        _, seq_len_k, _ = k.shape

        # Reshape: (batch, seq_len, embed_dim) -> (batch, seq_len, num_heads, head_dim)
        q_4d = q.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k_4d = k.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v_4d = v.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim)

        # Compute attention using backend
        backend = self._get_backend()

        # Reshape for attention computation: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        q_reshaped = q_4d.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len_q, head_dim)
        k_reshaped = k_4d.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len_k, head_dim)
        v_reshaped = v_4d.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len_k, head_dim)

        # Compute attention scores
        scores = backend.attention.attention_scores(
            q_reshaped, k_reshaped, num_heads=self.num_heads, head_dim=self.head_dim
        )

        # Backend may return scores in different shape - normalize to (batch, num_heads, seq_len_q, seq_len_k)
        if scores.shape == (batch_size, seq_len_q, self.num_heads, seq_len_k):
            # Backend returned (batch, seq_len_q, num_heads, seq_len_k) - transpose to (batch, num_heads, seq_len_q, seq_len_k)
            scores = scores.transpose(0, 2, 1, 3)
        elif scores.shape != (batch_size, self.num_heads, seq_len_q, seq_len_k):
            # Unexpected shape - try to infer
            if (
                scores.ndim == 4
                and scores.size == batch_size * self.num_heads * seq_len_q * seq_len_k
            ):
                scores = scores.reshape(batch_size, self.num_heads, seq_len_q, seq_len_k)
            else:
                # Fallback: compute manually
                scores = np.einsum("bhqd,bhkd->bhqk", q_reshaped, k_reshaped) / np.sqrt(
                    self.head_dim
                )

        # Cache pre-softmax scores for backward
        self._cached_scores_pre_softmax = scores.copy()

        # Apply mask if provided
        if mask is not None:
            scores = backend.attention.attention_mask(scores, mask)

        # Apply softmax (CPU for now - backend softmax expects 3D)
        # scores is (batch, num_heads, seq_len_q, seq_len_k)
        scores_max = scores.max(axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        scores_softmax = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

        # Cache softmax scores for backward
        self._cached_scores = scores_softmax.copy()

        # Compute attention output
        # scores_softmax: (batch, num_heads, seq_len_q, seq_len_k)
        # v_reshaped: (batch, num_heads, seq_len_k, head_dim)
        # Output: (batch, num_heads, seq_len_q, head_dim)
        attn_output = np.einsum("bhqk,bhkd->bhqd", scores_softmax, v_reshaped)

        # Cache attention output for backward (in shape: batch, num_heads, seq_len_q, head_dim)
        self._cached_attn_output = attn_output.copy()

        # Reshape back: (batch, num_heads, seq_len_q, head_dim) -> (batch, seq_len_q, embed_dim)
        attn_output_reshaped = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len_q, self.embed_dim
        )
        attn_weights = scores_softmax

        # Project output
        output = self.out_proj(attn_output_reshaped)

        return output, attn_weights

    def backward(
        self,
        grad_output: np.ndarray,
        query: np.ndarray = None,
        key: np.ndarray = None,
        value: np.ndarray = None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for multi-head attention.

        Attention: output = softmax(Q @ K^T / sqrt(d)) @ V
        Then: final_output = out_proj(attention_output)

        Args:
            grad_output: Gradient w.r.t. output (batch, seq_len_q, embed_dim)
            query: Query tensor (optional, uses cached if available)
            key: Key tensor (optional, uses cached if available)
            value: Value tensor (optional, uses cached if available)
            mask: Optional attention mask (optional, uses cached if available)

        Returns:
            (grad_query, grad_key, grad_value)
        """
        # Use cached values if available
        if query is None:
            query = self._cached_query
        if key is None:
            key = self._cached_key
        if value is None:
            value = self._cached_value
        if mask is None:
            mask = self._cached_mask

        if query is None or key is None or value is None:
            raise ValueError(
                "Query, key, and value are required for MultiheadAttention backward pass"
            )

        # Get cached intermediate values
        scores = self._cached_scores  # (batch, num_heads, seq_len_q, seq_len_k)
        q = self._cached_q  # (batch, seq_len_q, embed_dim)
        k = self._cached_k  # (batch, seq_len_k, embed_dim)
        v = self._cached_v  # (batch, seq_len_k, embed_dim)

        batch_size, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape

        # Step 1: Backward through output projection
        # _cached_attn_output is (batch, num_heads, seq_len_q, head_dim)
        # Need to reshape to (batch, seq_len_q, embed_dim) for out_proj backward
        attn_output_for_proj = self._cached_attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len_q, self.embed_dim
        )

        # Flatten batch and seq_len for Linear backward: (batch * seq_len, embed_dim)
        grad_output_flat = grad_output.reshape(-1, self.embed_dim)
        attn_output_flat = attn_output_for_proj.reshape(-1, self.embed_dim)

        if hasattr(self.out_proj, "backward"):
            grad_attn_output_flat = self.out_proj.backward(grad_output_flat, attn_output_flat)
            grad_attn_output = grad_attn_output_flat.reshape(batch_size, seq_len_q, self.embed_dim)
        else:
            # Simplified: assume linear projection
            weight = _get_param_array(self.out_proj.weight)
            grad_attn_output = grad_output @ weight.T

        # Reshape grad_attn_output: (batch, seq_len_q, embed_dim) -> (batch, num_heads, seq_len_q, head_dim)
        grad_attn_output = grad_attn_output.reshape(
            batch_size, seq_len_q, self.num_heads, self.head_dim
        )
        grad_attn_output = grad_attn_output.transpose(
            0, 2, 1, 3
        )  # (batch, num_heads, seq_len_q, head_dim)

        # Reshape V for backward: (batch, seq_len_k, embed_dim) -> (batch, num_heads, seq_len_k, head_dim)
        v_reshaped = v.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # Step 2: Backward through attention output: grad_V and grad_scores
        # grad_V = scores^T @ grad_attn_output
        # grad_scores = grad_attn_output @ V^T
        # v_reshaped: (batch, num_heads, seq_len_k, head_dim) -> transpose to (batch, num_heads, head_dim, seq_len_k)
        v_reshaped_T = v_reshaped.transpose(0, 1, 3, 2)  # (batch, num_heads, head_dim, seq_len_k)
        grad_scores = np.matmul(
            grad_attn_output, v_reshaped_T
        )  # (batch, num_heads, seq_len_q, seq_len_k)

        # scores: (batch, num_heads, seq_len_q, seq_len_k) -> transpose to (batch, num_heads, seq_len_k, seq_len_q)
        scores_T = scores.transpose(0, 1, 3, 2)  # (batch, num_heads, seq_len_k, seq_len_q)
        grad_v = np.matmul(scores_T, grad_attn_output)  # (batch, num_heads, seq_len_k, head_dim)

        # Step 3: Backward through softmax
        # Softmax backward: grad_pre_softmax = scores * (grad_scores - sum(grad_scores * scores, dim=-1, keepdims=True))
        grad_scores_weighted = grad_scores * scores
        grad_scores_sum = np.sum(grad_scores_weighted, axis=-1, keepdims=True)
        grad_pre_softmax = scores * (grad_scores - grad_scores_sum)

        # Step 4: Backward through attention scores: grad_Q and grad_K
        # scores = Q @ K^T / sqrt(head_dim)
        # grad_Q = grad_pre_softmax @ K / sqrt(head_dim)
        # grad_K = grad_pre_softmax^T @ Q / sqrt(head_dim)
        scale = 1.0 / np.sqrt(self.head_dim)

        # Reshape Q and K for backward
        q_reshaped = q.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        k_reshaped = k.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # grad_pre_softmax: (batch, num_heads, seq_len_q, seq_len_k)
        # k_reshaped: (batch, num_heads, seq_len_k, head_dim)
        grad_q_reshaped = (
            np.matmul(grad_pre_softmax, k_reshaped) * scale
        )  # (batch, num_heads, seq_len_q, head_dim)

        # grad_pre_softmax^T: (batch, num_heads, seq_len_k, seq_len_q)
        grad_pre_softmax_T = grad_pre_softmax.transpose(
            0, 1, 3, 2
        )  # (batch, num_heads, seq_len_k, seq_len_q)
        grad_k_reshaped = (
            np.matmul(grad_pre_softmax_T, q_reshaped) * scale
        )  # (batch, num_heads, seq_len_k, head_dim)

        # Reshape back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)
        grad_q = grad_q_reshaped.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len_q, self.embed_dim
        )
        grad_k = grad_k_reshaped.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len_k, self.embed_dim
        )
        grad_v = grad_v.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_k, self.embed_dim)

        # Step 5: Backward through Q, K, V projections
        # Flatten batch and seq_len dimensions for Linear backward
        grad_q_flat = grad_q.reshape(-1, self.embed_dim)
        grad_k_flat = grad_k.reshape(-1, self.embed_dim)
        grad_v_flat = grad_v.reshape(-1, self.embed_dim)
        query_flat = query.reshape(-1, self.embed_dim)
        key_flat = key.reshape(-1, self.embed_dim)
        value_flat = value.reshape(-1, self.embed_dim)

        if hasattr(self.q_proj, "backward"):
            grad_query_flat = self.q_proj.backward(grad_q_flat, query_flat)
            grad_query = grad_query_flat.reshape(batch_size, seq_len_q, self.embed_dim)
        else:
            grad_query = grad_q

        if hasattr(self.k_proj, "backward"):
            grad_key_flat = self.k_proj.backward(grad_k_flat, key_flat)
            grad_key = grad_key_flat.reshape(batch_size, seq_len_k, self.embed_dim)
        else:
            grad_key = grad_k

        if hasattr(self.v_proj, "backward"):
            grad_value_flat = self.v_proj.backward(grad_v_flat, value_flat)
            grad_value = grad_value_flat.reshape(batch_size, seq_len_k, self.embed_dim)
        else:
            grad_value = grad_v

        return grad_query, grad_key, grad_value  # Placeholder

    def __repr__(self):
        """Return a debug representation."""

        return f"MultiheadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"


class FlashAttention2(Module):
    """
    Flash Attention 2 (memory-efficient attention)
    Uses: flash-attention2.glsl, flash-attention2-rope.glsl
    """

    def __init__(self, embed_dim: int, num_heads: int, use_rope: bool = False):
        """Initialize the instance."""

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Initialize cached values for backward pass
        self._cached_q = None
        self._cached_k = None
        self._cached_v = None
        self._cached_mask = None
        self._cached_output = None

    def forward(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Forward pass using Flash Attention 2.

        Args:
            q: Query (batch, seq_len, num_heads, head_dim) or (batch, seq_len, embed_dim)
            k: Key (batch, seq_len, num_heads, head_dim) or (batch, seq_len, embed_dim)
            v: Value (batch, seq_len, num_heads, head_dim) or (batch, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, num_heads, head_dim) or (batch, seq_len, embed_dim)
        """
        # Cache inputs for backward pass
        self._cached_q = q.copy()
        self._cached_k = k.copy()
        self._cached_v = v.copy()
        self._cached_mask = mask

        # Handle different input shapes
        if q.ndim == 3:
            # (batch, seq_len, embed_dim) -> reshape to (batch, seq_len, num_heads, head_dim)
            batch_size, seq_len, _ = q.shape
            q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        backend = self._get_backend()
        if hasattr(backend, "flash_attention2"):
            try:
                output = backend.flash_attention2(q, k, v, mask=mask, use_rope=self.use_rope)
                self._cached_output = output.copy()
                return output
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback: use standard attention computation
        # This is similar to FlashAttention but without tiling
        batch_size, seq_len, _, _ = q.shape

        # Reshape for attention: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        q_reshaped = q.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        k_reshaped = k.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        v_reshaped = v.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)

        # Compute attention scores: Q @ K^T / sqrt(head_dim)
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.einsum("bhqd,bhkd->bhqk", q_reshaped, k_reshaped) * scale

        # Apply mask if provided
        if mask is not None:
            if mask.ndim == 2:
                # (batch, seq_len) -> expand to (batch, num_heads, seq_len, seq_len)
                mask_expanded = mask[:, None, :, None]  # (batch, 1, seq_len, 1)
                mask_expanded = np.broadcast_to(mask_expanded, scores.shape)
                scores = np.where(mask_expanded > 0, scores, -1e9)
            else:
                scores = scores + mask

        # Apply softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        scores_softmax = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

        # Compute attention output: scores @ V
        output = np.einsum("bhqk,bhkd->bhqd", scores_softmax, v_reshaped)

        # Reshape back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        output = output.transpose(0, 2, 1, 3)

        self._cached_output = output.copy()
        return output

    def backward(
        self,
        grad_output: np.ndarray,
        q: np.ndarray = None,
        k: np.ndarray = None,
        v: np.ndarray = None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for Flash Attention 2.

        FlashAttention2 uses the same mathematical operations as standard attention,
        so the backward pass is similar to MultiheadAttention backward.

        Args:
            grad_output: Gradient w.r.t. output (batch, seq_len, num_heads, head_dim) or (batch, seq_len, embed_dim)
            q: Query tensor (optional, uses cached if available)
            k: Key tensor (optional, uses cached if available)
            v: Value tensor (optional, uses cached if available)
            mask: Optional attention mask (optional, uses cached if available)

        Returns:
            (grad_q, grad_k, grad_v)
        """
        # Use cached values if available
        if q is None:
            q = self._cached_q
        if k is None:
            k = self._cached_k
        if v is None:
            v = self._cached_v
        if mask is None:
            mask = self._cached_mask

        if q is None or k is None or v is None:
            raise ValueError("Query, key, and value are required for FlashAttention2 backward pass")

        # Handle different input shapes
        if q.ndim == 3:
            # (batch, seq_len, embed_dim) -> reshape to (batch, seq_len, num_heads, head_dim)
            batch_size, seq_len, _ = q.shape
            q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        else:
            batch_size, seq_len, _, _ = q.shape

        # Handle grad_output shape
        if grad_output.ndim == 3:
            # (batch, seq_len, embed_dim) -> reshape to (batch, seq_len, num_heads, head_dim)
            grad_output = grad_output.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Reshape for attention computation: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        q_reshaped = q.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        k_reshaped = k.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        v_reshaped = v.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        grad_output_reshaped = grad_output.transpose(
            0, 2, 1, 3
        )  # (batch, num_heads, seq_len, head_dim)

        # Recompute attention scores for backward (same as forward)
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.einsum("bhqd,bhkd->bhqk", q_reshaped, k_reshaped) * scale

        # Apply mask if provided
        if mask is not None:
            if mask.ndim == 2:
                mask_expanded = mask[:, None, :, None]
                mask_expanded = np.broadcast_to(mask_expanded, scores.shape)
                scores = np.where(mask_expanded > 0, scores, -1e9)
            else:
                scores = scores + mask

        # Apply softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        scores_softmax = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

        # Step 1: Backward through attention output: grad_V and grad_scores
        # grad_V = scores^T @ grad_output
        # grad_scores = grad_output @ V^T
        v_reshaped_T = v_reshaped.transpose(0, 1, 3, 2)  # (batch, num_heads, head_dim, seq_len)
        grad_scores = np.matmul(
            grad_output_reshaped, v_reshaped_T
        )  # (batch, num_heads, seq_len, seq_len)

        scores_T = scores_softmax.transpose(0, 1, 3, 2)  # (batch, num_heads, seq_len, seq_len)
        grad_v = np.matmul(scores_T, grad_output_reshaped)  # (batch, num_heads, seq_len, head_dim)

        # Step 2: Backward through softmax
        grad_scores_weighted = grad_scores * scores_softmax
        grad_scores_sum = np.sum(grad_scores_weighted, axis=-1, keepdims=True)
        grad_pre_softmax = scores_softmax * (grad_scores - grad_scores_sum)

        # Step 3: Backward through attention scores: grad_Q and grad_K
        # scores = Q @ K^T / sqrt(head_dim)
        # grad_Q = grad_pre_softmax @ K / sqrt(head_dim)
        # grad_K = grad_pre_softmax^T @ Q / sqrt(head_dim)
        grad_q_reshaped = (
            np.matmul(grad_pre_softmax, k_reshaped) * scale
        )  # (batch, num_heads, seq_len, head_dim)
        grad_pre_softmax_T = grad_pre_softmax.transpose(
            0, 1, 3, 2
        )  # (batch, num_heads, seq_len, seq_len)
        grad_k_reshaped = (
            np.matmul(grad_pre_softmax_T, q_reshaped) * scale
        )  # (batch, num_heads, seq_len, head_dim)

        # Reshape back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        grad_q = grad_q_reshaped.transpose(0, 2, 1, 3)  # (batch, seq_len, num_heads, head_dim)
        grad_k = grad_k_reshaped.transpose(0, 2, 1, 3)  # (batch, seq_len, num_heads, head_dim)
        grad_v = grad_v.transpose(0, 2, 1, 3)  # (batch, seq_len, num_heads, head_dim)

        # If original inputs were 3D, reshape back
        if self._cached_q.ndim == 3:
            grad_q = grad_q.reshape(batch_size, seq_len, self.embed_dim)
            grad_k = grad_k.reshape(batch_size, seq_len, self.embed_dim)
            grad_v = grad_v.reshape(batch_size, seq_len, self.embed_dim)

        return grad_q, grad_k, grad_v

    def __repr__(self):
        """Return a debug representation."""

        return f"FlashAttention2(embed_dim={self.embed_dim}, num_heads={self.num_heads}, use_rope={self.use_rope})"
