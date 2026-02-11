"""
GPU-accelerated backward operations using Vulkan shaders.

This module provides a bridge between Python autograd and Vulkan compute shaders
for gradient computation. It maps high-level operations (Linear, ReLU, SwiGLU, etc.)
to their corresponding GPU backward shaders.

Usage:
    >>> gpu_ops = GPUBackwardOps()
    >>> grad_input, grad_weight, grad_bias = gpu_ops.linear_backward(
    ...     grad_output, input_data, weights
    ... )
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class GPUBackwardOps:
    """
    Wrapper for GPU-accelerated backward operations.

    This class provides methods for computing gradients using Vulkan compute shaders.
    Each method corresponds to a backward operation for a specific layer/activation.

    All GPU operations fall back to CPU if:
    - GPU is not available
    - Shader is not compiled
    - Input shapes are incompatible
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize GPU backward operations.

        Args:
            use_gpu: Whether to use GPU acceleration (default: True)
        """
        self.backend = None
        self.use_gpu = use_gpu

        if use_gpu:
            try:
                from grilly import Compute

                self.backend = Compute()
                logger.info("GPU backward operations initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU backend: {e}. Falling back to CPU.")
                self.use_gpu = False

        # Map operation names to shader names
        self.shader_map = {
            "Linear": "fnn-linear-backward",
            "ReLU": "activation-relu-backward",
            "GELU": "activation-gelu-backward",
            "SiLU": "activation-silu-backward",
            "SwiGLU": "activation-swiglu-backward",
            "RoSwish": "activation-roswish-backward",
            "GCU": "activation-gcu-backward",
            "Softmax": "activation-softmax-backward",
            "LayerNorm": "fnn-layernorm-backward",
            "Attention": "attention-backward",
            "CrossEntropy": "cross-entropy-backward",
            "Conv2D": "conv2d-backward-input",
            "Conv2DWeight": "conv2d-backward-weight",
            "BatchNorm2D": "batchnorm2d-backward",
            "MaxPool2D": "maxpool2d-backward",
            "AvgPool2D": "avgpool2d-backward",
        }

    def is_available(self) -> bool:
        """Check if GPU backend is available."""
        return self.backend is not None and self.use_gpu

    def _has_shader(self, shader_name: str) -> bool:
        """Check if a shader is available."""
        if not self.is_available():
            return False
        try:
            return (
                hasattr(self.backend.core, "shaders") and shader_name in self.backend.core.shaders
            )
        except Exception:
            return False

    # ========================================================================
    # Linear / Fully Connected Layer
    # ========================================================================

    def linear_backward(
        self,
        grad_output: np.ndarray,
        input_data: np.ndarray,
        weights: np.ndarray,
        compute_input_grad: bool = True,
        compute_weight_grad: bool = True,
        compute_bias_grad: bool = True,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """
        Compute gradients for linear layer using GPU shader.

        Forward: output = input @ weights.T + bias

        Args:
            grad_output: Gradient w.r.t. output (batch, output_dim)
            input_data: Input activations (batch, input_dim)
            weights: Weight matrix (output_dim, input_dim)
            compute_input_grad: Whether to compute gradient w.r.t. input
            compute_weight_grad: Whether to compute gradient w.r.t. weights
            compute_bias_grad: Whether to compute gradient w.r.t. bias

        Returns:
            (grad_input, grad_weights, grad_bias)
            - grad_input: (batch, input_dim) or None
            - grad_weights: (output_dim, input_dim) or None
            - grad_bias: (output_dim,) or None
        """
        if not self.is_available():
            # CPU fallback
            return self._linear_backward_cpu(
                grad_output,
                input_data,
                weights,
                compute_input_grad,
                compute_weight_grad,
                compute_bias_grad,
            )

        try:
            # Use Grilly's existing linear_backward method!
            grad_input, grad_weight, grad_bias = self.backend.fnn.linear_backward(
                grad_output, input_data, weights, bias=None
            )

            # Filter outputs based on what was requested
            if not compute_input_grad:
                grad_input = None
            if not compute_weight_grad:
                grad_weight = None
            if not compute_bias_grad:
                grad_bias = None

            return grad_input, grad_weight, grad_bias

        except Exception as e:
            logger.warning(
                f"GPU linear backward failed: {type(e).__name__}: {e}. Falling back to CPU."
            )
            return self._linear_backward_cpu(
                grad_output,
                input_data,
                weights,
                compute_input_grad,
                compute_weight_grad,
                compute_bias_grad,
            )

    def _linear_backward_cpu(
        self,
        grad_output: np.ndarray,
        input_data: np.ndarray,
        weights: np.ndarray,
        compute_input_grad: bool,
        compute_weight_grad: bool,
        compute_bias_grad: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """CPU fallback for linear backward."""
        grad_input = None
        grad_weights = None
        grad_bias = None

        if compute_input_grad:
            # grad_input = grad_output @ weights
            grad_input = np.matmul(grad_output, weights)

        if compute_weight_grad:
            # grad_weights = grad_output.T @ input
            grad_weights = np.matmul(grad_output.T, input_data)

        if compute_bias_grad:
            # grad_bias = sum(grad_output, axis=0)
            grad_bias = np.sum(grad_output, axis=0)

        return grad_input, grad_weights, grad_bias

    # ========================================================================
    # Activation Functions
    # ========================================================================

    def relu_backward(self, grad_output: np.ndarray, input_data: np.ndarray) -> np.ndarray:
        """
        ReLU backward: grad * (input > 0)

        Args:
            grad_output: Gradient w.r.t. output
            input_data: Input to ReLU (saved from forward)

        Returns:
            grad_input: Gradient w.r.t. input
        """
        if not self.is_available() or not self._has_shader("activation-relu-backward"):
            # CPU fallback
            return grad_output * (input_data > 0).astype(np.float32)

        try:
            return self.backend.core.dispatch_shader(
                "activation-relu-backward",
                inputs={"grad_output": grad_output, "input_data": input_data},
                output_shape=grad_output.shape,
            )
        except Exception as e:
            logger.warning(f"GPU ReLU backward failed: {e}. Falling back to CPU.")
            return grad_output * (input_data > 0).astype(np.float32)

    def gelu_backward(self, grad_output: np.ndarray, input_data: np.ndarray) -> np.ndarray:
        """
        GELU backward.

        Args:
            grad_output: Gradient w.r.t. output
            input_data: Input to GELU

        Returns:
            grad_input: Gradient w.r.t. input
        """
        if not self.is_available() or not self._has_shader("activation-gelu-backward"):
            # CPU fallback (approximation)
            x = input_data
            sqrt_2_pi = np.sqrt(2.0 / np.pi)
            cdf_approx = 0.5 * (1.0 + np.tanh(sqrt_2_pi * (x + 0.044715 * x**3)))

            inner = sqrt_2_pi * (x + 0.044715 * x**3)
            tanh_inner = np.tanh(inner)
            sech2 = 1 - tanh_inner**2
            dcdf = 0.5 * sech2 * sqrt_2_pi * (1 + 3 * 0.044715 * x**2)

            return grad_output * (cdf_approx + x * dcdf)

        try:
            return self.backend.core.dispatch_shader(
                "activation-gelu-backward",
                inputs={"grad_output": grad_output, "input_data": input_data},
                output_shape=grad_output.shape,
            )
        except Exception as e:
            logger.warning(f"GPU GELU backward failed: {e}. Falling back to CPU.")
            # CPU fallback
            x = input_data
            sqrt_2_pi = np.sqrt(2.0 / np.pi)
            cdf_approx = 0.5 * (1.0 + np.tanh(sqrt_2_pi * (x + 0.044715 * x**3)))
            inner = sqrt_2_pi * (x + 0.044715 * x**3)
            tanh_inner = np.tanh(inner)
            sech2 = 1 - tanh_inner**2
            dcdf = 0.5 * sech2 * sqrt_2_pi * (1 + 3 * 0.044715 * x**2)
            return grad_output * (cdf_approx + x * dcdf)

    def silu_backward(self, grad_output: np.ndarray, input_data: np.ndarray) -> np.ndarray:
        """
        SiLU/Swish backward: d/dx(x * sigmoid(x))

        Args:
            grad_output: Gradient w.r.t. output
            input_data: Input to SiLU

        Returns:
            grad_input: Gradient w.r.t. input
        """
        if not self.is_available() or not self._has_shader("activation-silu-backward"):
            # CPU fallback
            sigmoid_x = 1.0 / (1.0 + np.exp(-input_data))
            return grad_output * sigmoid_x * (1 + input_data * (1 - sigmoid_x))

        try:
            return self.backend.core.dispatch_shader(
                "activation-silu-backward",
                inputs={"grad_output": grad_output, "input_data": input_data},
                output_shape=grad_output.shape,
            )
        except Exception as e:
            logger.warning(f"GPU SiLU backward failed: {e}. Falling back to CPU.")
            sigmoid_x = 1.0 / (1.0 + np.exp(-input_data))
            return grad_output * sigmoid_x * (1 + input_data * (1 - sigmoid_x))

    def swiglu_backward(
        self, grad_output: np.ndarray, input_data: np.ndarray, gate_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        SwiGLU backward: d/dx(SiLU(gate) * value)

        SwiGLU splits input into two halves: gate and value
        Output = SiLU(gate) * value

        Args:
            grad_output: Gradient w.r.t. output
            input_data: Value half of input
            gate_data: Gate half of input

        Returns:
            (grad_gate, grad_value)
        """
        if not self.is_available() or not self._has_shader("activation-swiglu-backward"):
            # CPU fallback
            sigmoid_gate = 1.0 / (1.0 + np.exp(-gate_data))
            silu_gate = gate_data * sigmoid_gate

            # Gradient w.r.t. value: grad_output * SiLU(gate)
            grad_value = grad_output * silu_gate

            # Gradient w.r.t. gate: grad_output * value * d/dx(SiLU(gate))
            dsilu = sigmoid_gate * (1 + gate_data * (1 - sigmoid_gate))
            grad_gate = grad_output * input_data * dsilu

            return grad_gate, grad_value

        try:
            grads = self.backend.core.dispatch_shader(
                "activation-swiglu-backward",
                inputs={
                    "grad_output": grad_output,
                    "gate_data": gate_data,
                    "value_data": input_data,
                },
                output_shape=(grad_output.shape[0], grad_output.shape[1] * 2),
            )
            # Split output into gate and value gradients
            mid = grad_output.shape[1]
            grad_gate = grads[:, :mid]
            grad_value = grads[:, mid:]
            return grad_gate, grad_value

        except Exception as e:
            logger.warning(f"GPU SwiGLU backward failed: {e}. Falling back to CPU.")
            sigmoid_gate = 1.0 / (1.0 + np.exp(-gate_data))
            silu_gate = gate_data * sigmoid_gate
            grad_value = grad_output * silu_gate
            dsilu = sigmoid_gate * (1 + gate_data * (1 - sigmoid_gate))
            grad_gate = grad_output * input_data * dsilu
            return grad_gate, grad_value

    def roswish_backward(
        self, grad_output: np.ndarray, input_data: np.ndarray, alpha: float, beta: float
    ) -> np.ndarray:
        """
        RoSwish backward (learnable Swish variant).

        Args:
            grad_output: Gradient w.r.t. output
            input_data: Input to RoSwish
            alpha: Learnable parameter
            beta: Learnable parameter

        Returns:
            grad_input: Gradient w.r.t. input
        """
        if not self.is_available() or not self._has_shader("activation-roswish-backward"):
            # CPU fallback
            # RoSwish: x * sigmoid(alpha * x) + beta
            sigmoid_x = 1.0 / (1.0 + np.exp(-alpha * input_data))
            # d/dx = sigmoid + alpha * x * sigmoid * (1 - sigmoid)
            grad = sigmoid_x + alpha * input_data * sigmoid_x * (1 - sigmoid_x)
            return grad_output * grad

        try:
            return self.backend.core.dispatch_shader(
                "activation-roswish-backward",
                inputs={"grad_output": grad_output, "input_data": input_data},
                output_shape=grad_output.shape,
                push_constants={"alpha": alpha, "beta": beta},
            )
        except Exception as e:
            logger.warning(f"GPU RoSwish backward failed: {e}. Falling back to CPU.")
            sigmoid_x = 1.0 / (1.0 + np.exp(-alpha * input_data))
            grad = sigmoid_x + alpha * input_data * sigmoid_x * (1 - sigmoid_x)
            return grad_output * grad

    def gcu_backward(
        self, grad_output: np.ndarray, input_data: np.ndarray, omega: float = 1.0
    ) -> np.ndarray:
        """
        GCU (Gaussian Cosine Unit) backward - oscillatory activation.

        Args:
            grad_output: Gradient w.r.t. output
            input_data: Input to GCU
            omega: Frequency parameter

        Returns:
            grad_input: Gradient w.r.t. input
        """
        if not self.is_available() or not self._has_shader("activation-gcu-backward"):
            # CPU fallback
            # GCU: x * exp(-x^2 / 2) * cos(omega * x)
            # d/dx = exp(-x^2/2) * (cos(omega*x) - x*sin(omega*x)*omega - x*cos(omega*x))
            exp_term = np.exp(-(input_data**2) / 2)
            cos_term = np.cos(omega * input_data)
            sin_term = np.sin(omega * input_data)
            grad = exp_term * (cos_term - input_data * sin_term * omega - input_data * cos_term)
            return grad_output * grad

        try:
            return self.backend.core.dispatch_shader(
                "activation-gcu-backward",
                inputs={"grad_output": grad_output, "input_data": input_data},
                output_shape=grad_output.shape,
                push_constants={"omega": omega},
            )
        except Exception as e:
            logger.warning(f"GPU GCU backward failed: {e}. Falling back to CPU.")
            exp_term = np.exp(-(input_data**2) / 2)
            cos_term = np.cos(omega * input_data)
            sin_term = np.sin(omega * input_data)
            grad = exp_term * (cos_term - input_data * sin_term * omega - input_data * cos_term)
            return grad_output * grad

    # ========================================================================
    # Normalization
    # ========================================================================

    def layernorm_backward(
        self,
        grad_output: np.ndarray,
        input_data: np.ndarray,
        normalized: np.ndarray,
        gamma: np.ndarray,
        eps: float = 1e-5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        LayerNorm backward.

        Args:
            grad_output: Gradient w.r.t. output
            input_data: Original input
            normalized: Normalized values (saved from forward)
            gamma: Scale parameter
            eps: Epsilon for numerical stability

        Returns:
            (grad_input, grad_gamma, grad_beta)
        """
        if not self.is_available() or not self._has_shader("fnn-layernorm-backward"):
            # CPU fallback
            # This is complex - simplified version
            grad_gamma = np.sum(grad_output * normalized, axis=0)
            grad_beta = np.sum(grad_output, axis=0)

            # grad_input computation (simplified)
            mean = np.mean(input_data, axis=-1, keepdims=True)
            var = np.var(input_data, axis=-1, keepdims=True)
            std = np.sqrt(var + eps)

            N = input_data.shape[-1]
            grad_normalized = grad_output * gamma
            grad_var = np.sum(
                grad_normalized * (input_data - mean) * -0.5 * (var + eps) ** (-1.5),
                axis=-1,
                keepdims=True,
            )
            grad_mean = np.sum(
                grad_normalized * -1.0 / std, axis=-1, keepdims=True
            ) + grad_var * np.mean(-2.0 * (input_data - mean), axis=-1, keepdims=True)
            grad_input = (
                grad_normalized / std + grad_var * 2.0 * (input_data - mean) / N + grad_mean / N
            )

            return grad_input, grad_gamma, grad_beta

        try:
            grads = self.backend.core.dispatch_shader(
                "fnn-layernorm-backward",
                inputs={
                    "grad_output": grad_output,
                    "input_data": input_data,
                    "normalized": normalized,
                    "gamma": gamma,
                },
                push_constants={"eps": eps},
            )
            # Parse output
            grad_input = grads["grad_input"]
            grad_gamma = grads["grad_gamma"]
            grad_beta = grads["grad_beta"]
            return grad_input, grad_gamma, grad_beta

        except Exception as e:
            logger.warning(f"GPU LayerNorm backward failed: {e}. Falling back to CPU.")
            # CPU fallback (same as above)
            grad_gamma = np.sum(grad_output * normalized, axis=0)
            grad_beta = np.sum(grad_output, axis=0)
            mean = np.mean(input_data, axis=-1, keepdims=True)
            var = np.var(input_data, axis=-1, keepdims=True)
            std = np.sqrt(var + eps)
            N = input_data.shape[-1]
            grad_normalized = grad_output * gamma
            grad_var = np.sum(
                grad_normalized * (input_data - mean) * -0.5 * (var + eps) ** (-1.5),
                axis=-1,
                keepdims=True,
            )
            grad_mean = np.sum(
                grad_normalized * -1.0 / std, axis=-1, keepdims=True
            ) + grad_var * np.mean(-2.0 * (input_data - mean), axis=-1, keepdims=True)
            grad_input = (
                grad_normalized / std + grad_var * 2.0 * (input_data - mean) / N + grad_mean / N
            )
            return grad_input, grad_gamma, grad_beta

    # ========================================================================
    # Softmax & Loss
    # ========================================================================

    def softmax_backward(
        self, grad_output: np.ndarray, softmax_output: np.ndarray, dim: int = -1
    ) -> np.ndarray:
        """
        Softmax backward.

        Args:
            grad_output: Gradient w.r.t. output
            softmax_output: Softmax output (saved from forward)
            dim: Dimension along which softmax was computed

        Returns:
            grad_input: Gradient w.r.t. input
        """
        if not self.is_available() or not self._has_shader("activation-softmax-backward"):
            # CPU fallback
            # grad_input = softmax * (grad - sum(grad * softmax))
            s = softmax_output
            grad_input = s * (grad_output - np.sum(grad_output * s, axis=dim, keepdims=True))
            return grad_input

        try:
            return self.backend.core.dispatch_shader(
                "activation-softmax-backward",
                inputs={"grad_output": grad_output, "softmax_output": softmax_output},
                output_shape=grad_output.shape,
                push_constants={"dim": dim},
            )
        except Exception as e:
            logger.warning(f"GPU Softmax backward failed: {e}. Falling back to CPU.")
            s = softmax_output
            grad_input = s * (grad_output - np.sum(grad_output * s, axis=dim, keepdims=True))
            return grad_input

    def cross_entropy_backward(self, softmax_output: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Cross-entropy loss backward (combined with softmax).

        Args:
            softmax_output: Softmax probabilities
            targets: Target class indices or one-hot vectors

        Returns:
            grad_input: Gradient w.r.t. logits
        """
        if not self.is_available() or not self._has_shader("cross-entropy-backward"):
            # CPU fallback
            if targets.ndim == softmax_output.ndim:
                # Soft targets
                grad_input = (softmax_output - targets) / targets.shape[0]
            else:
                # Hard targets (class indices)
                grad_input = softmax_output.copy()
                batch_size = softmax_output.shape[0]
                grad_input[np.arange(batch_size), targets.astype(int)] -= 1
                grad_input /= batch_size
            return grad_input

        try:
            return self.backend.core.dispatch_shader(
                "cross-entropy-backward",
                inputs={"softmax_output": softmax_output, "targets": targets},
                output_shape=softmax_output.shape,
            )
        except Exception as e:
            logger.warning(f"GPU CrossEntropy backward failed: {e}. Falling back to CPU.")
            if targets.ndim == softmax_output.ndim:
                grad_input = (softmax_output - targets) / targets.shape[0]
            else:
                grad_input = softmax_output.copy()
                batch_size = softmax_output.shape[0]
                grad_input[np.arange(batch_size), targets.astype(int)] -= 1
                grad_input /= batch_size
            return grad_input


# Global instance (lazy initialization)
_gpu_ops_instance = None


def get_gpu_backward_ops(use_gpu: bool = True) -> GPUBackwardOps:
    """
    Get the global GPU backward operations instance.

    Args:
        use_gpu: Whether to use GPU acceleration

    Returns:
        GPUBackwardOps instance
    """
    global _gpu_ops_instance
    if _gpu_ops_instance is None:
        _gpu_ops_instance = GPUBackwardOps(use_gpu=use_gpu)
    return _gpu_ops_instance
