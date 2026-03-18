"""
Activation function modules — ReLU, GELU, SiLU, GCU, RoSwish, SwiGLU, Softmax, Softplus.
Uses: activation-relu.glsl, activation-gelu.glsl, activation-silu.glsl, etc.
"""

import numpy as np

from ._helpers import (
    _PARAMETER_AVAILABLE,
    _USE_CPP_BRIDGE,
    ParameterClass,
    _bridge,
    _bridge_to_numpy,
)
from .module import Module


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
        """Forward pass using activation-relu.glsl (GPU-first)"""
        if _USE_CPP_BRIDGE:
            result = _bridge_to_numpy(_bridge.relu(x))
            if result is not None:
                return result
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
        return grad_output * (np.asarray(x) > 0).astype(np.float32)

    def __repr__(self):
        """Return a debug representation."""

        return "ReLU()"


class GELU(Module):
    """
    GELU activation
    Uses: activation-gelu.glsl
    """

    def forward(self, x) -> np.ndarray:
        """Forward pass using activation-gelu.glsl (GPU-first)"""
        if _USE_CPP_BRIDGE:
            result = _bridge_to_numpy(_bridge.gelu(x))
            if result is not None:
                return result
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
        """Forward pass using activation-silu.glsl (GPU-first)"""
        if _USE_CPP_BRIDGE:
            result = _bridge_to_numpy(_bridge.silu(x))
            if result is not None:
                return result
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
