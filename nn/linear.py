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
    _create_param_wrapper,
    _get_param_array,
)
from ._perf_policy import choose_fastest
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
            self.weight = _create_param_wrapper(weight_data)

        if bias:
            if _PARAMETER_AVAILABLE and ParameterClass is not None:
                self.bias = ParameterClass(
                    np.zeros(out_features, dtype=np.float32), requires_grad=True
                )
            else:
                self.bias = _create_param_wrapper(np.zeros(out_features, dtype=np.float32))
        else:
            self.bias = None

        # Register parameters
        self.register_parameter("weight", self.weight)
        if self.bias is not None:
            self.register_parameter("bias", self.bias)

    def forward(self, x):
        """Forward pass — GPU-first, always dispatches through GPU backend.

        Autograd: when ``x`` is an autograd ``Variable`` (or carries
        ``requires_grad``), the output is wrapped in a ``Variable`` with a
        ``GradFn`` that calls ``self.backward(grad_output, x_data)`` during
        ``loss.backward()``. ``self.backward`` already populates
        ``self.weight.grad`` and ``self.bias.grad`` from the existing
        ``fnn-linear-backward.glsl`` kernel, so the AdamW step picks them up
        through ``param.grad`` like any other PyTorch-style optimizer.

        Without this wiring, ``loss.backward()`` produced gradients on the
        Variable wrapping the cross-entropy logits but the chain stopped at
        ``Linear`` (no ``grad_fn`` -> autograd traversal terminated), so
        weights silently never updated.
        """
        # Detect autograd input (Variable, or Tensor — Tensor is a Variable subclass)
        # and remember its underlying ndarray for the backward closure.
        try:
            from grilly.nn.autograd import GradFn as _GradFn
            from grilly.nn.autograd import Variable as _Variable
            from grilly.nn.autograd import _grad_enabled
        except ImportError:
            _Variable = None  # type: ignore[assignment]
            _GradFn = None  # type: ignore[assignment]
            _grad_enabled = False

        x_var: _Variable | None = None  # type: ignore[name-defined]
        if _Variable is not None and isinstance(x, _Variable):
            x_var = x
            x_data = x.data
        else:
            x_data = x  # raw ndarray (or VulkanTensor)

        weight = _get_param_array(self.weight)
        bias = _get_param_array(self.bias) if self.bias is not None else None

        def cpu_linear():
            x_arr = np.asarray(x_data, dtype=np.float32)
            w_arr = np.asarray(weight, dtype=np.float32)
            out = x_arr @ w_arr.T
            if bias is not None:
                out = out + np.asarray(bias, dtype=np.float32)
            return np.asarray(out, dtype=np.float32)

        # ---- Run the existing forward path (unchanged) ----
        result: np.ndarray | None = None
        if _USE_CPP_BRIDGE:
            def gpu_linear():
                return _bridge_to_numpy(_bridge.linear(x_data, weight, bias))

            # Auto-fastest policy only for numpy-in/numpy-out path.
            if isinstance(x_data, np.ndarray) and not self._return_gpu_tensor:
                batch = int(np.prod(x_data.shape[:-1])) if x_data.ndim > 1 else 1
                in_features = int(x_data.shape[-1]) if x_data.ndim > 0 else self.in_features
                op_key = f"linear:{batch}x{in_features}x{self.out_features}"
                result = choose_fastest(op_key, gpu_linear, cpu_linear)
            else:
                result = gpu_linear()

        if result is None:
            # Legacy Python Vulkan backend (fallback) or final CPU path
            backend = self._get_backend()
            if hasattr(backend, "fnn") and hasattr(backend.fnn, "linear"):
                result = backend.fnn.linear(
                    x_data,
                    weight,
                    bias,
                    return_gpu_tensor=self._return_gpu_tensor,
                )
            else:
                result = cpu_linear()

        # ---- Autograd wiring ----
        # If the input came from autograd, wrap result so loss.backward()
        # can flow back through this layer and populate weight.grad / bias.grad
        # via the existing self.backward() implementation. We bypass
        # ``_make_backward`` (which short-circuits when no input requires grad)
        # because the *weight* always requires grad even if the input doesn't —
        # otherwise a frozen-input training loop silently never updates weights.
        if (
            x_var is not None
            and _GradFn is not None
            and _grad_enabled
            and isinstance(result, np.ndarray)
            and not isinstance(result, _Variable)
        ):
            x_data_for_backward = x_data  # capture for closure

            def backward_fn(grad_output):
                # self.backward populates self.weight.grad and self.bias.grad
                # in-place and returns grad_input.
                grad_input = self.backward(np.asarray(grad_output), x_data_for_backward)
                return (grad_input,)

            grad_fn = _GradFn("Linear", backward_fn, [x_var])
            return _Variable(np.asarray(result), requires_grad=True, grad_fn=grad_fn)

        return result

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
