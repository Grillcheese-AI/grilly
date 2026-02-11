"""
Vulkan Autograd Core - Automatic Differentiation with GPU Acceleration

This module provides tape-based automatic differentiation that integrates with
the grilly nn.Module system and dispatches backward passes to GPU shaders.
"""

import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np

# Thread-local storage for gradient tape
_local = threading.local()


def _get_tape() -> Optional["GradientTape"]:
    """Get the current active gradient tape, if any."""
    return getattr(_local, "tape", None)


def _set_tape(tape: Optional["GradientTape"]):
    """Set the current active gradient tape."""
    _local.tape = tape


class GradientTape:
    """
    Records operations for automatic differentiation.

    Example::

        with GradientTape() as tape:
            y = model(x)
            loss = loss_fn(y, target)
        grads = tape.gradient(loss, model.parameters())

    You can also construct it with `watch=...` for convenience.
    """

    def __init__(self, persistent: bool = False, watch: list = None):
        """
        Args:
            persistent: If True, tape can be used for multiple gradient calls
            watch: Optional list of parameters to watch from the start
        """
        self.persistent = persistent
        self._operations: list[tuple[Any, Callable, list[Any]]] = []
        self._watched: dict[int, np.ndarray] = {}  # id -> array mapping
        self._enabled = True

        if watch:
            for param in watch:
                self.watch(param)

    def __enter__(self):
        """Enter the runtime context."""

        _set_tape(self)
        return self

    def __exit__(self, *args):
        """Exit the runtime context."""

        if not self.persistent:
            self._operations.clear()
        _set_tape(None)

    def watch(self, tensor: np.ndarray):
        """Explicitly watch a tensor for gradient computation."""
        if isinstance(tensor, np.ndarray):
            self._watched[id(tensor)] = tensor
        elif hasattr(tensor, "data"):
            self._watched[id(tensor.data)] = tensor

    def record(self, output: np.ndarray, backward_fn: Callable, inputs: list[Any]):
        """
        Record an operation for later backward pass.

        Args:
            output: The output tensor of the operation
            backward_fn: Function to call during backward (takes grad_output, returns grad_inputs)
            inputs: List of input tensors/values used in forward
        """
        if self._enabled:
            self._operations.append((output, backward_fn, inputs))

    def gradient(
        self, target: np.ndarray, sources: list, output_gradients: np.ndarray = None
    ) -> list[np.ndarray]:
        """
        Compute gradients of target w.r.t. sources.

        Args:
            target: The tensor to differentiate (usually loss)
            sources: List of tensors to compute gradients for
            output_gradients: Initial gradient (default: ones_like(target))

        Returns:
            List of gradients corresponding to each source
        """
        if output_gradients is None:
            output_gradients = np.ones_like(target, dtype=np.float32)

        # Build gradient accumulator for each source
        source_ids = {id(s.data if hasattr(s, "data") else s): i for i, s in enumerate(sources)}
        grads = [None] * len(sources)

        # Current gradient being propagated
        grad_map = {id(target): output_gradients}

        # Backward pass through recorded operations (reverse order)
        for output, backward_fn, inputs in reversed(self._operations):
            out_id = id(output)
            if out_id not in grad_map:
                continue

            grad_output = grad_map[out_id]

            # Call backward function
            grad_inputs = backward_fn(grad_output)
            if not isinstance(grad_inputs, (list, tuple)):
                grad_inputs = [grad_inputs]

            # Distribute gradients to inputs
            for inp, grad_in in zip(inputs, grad_inputs):
                if grad_in is None:
                    continue
                inp_data = inp.data if hasattr(inp, "data") else inp
                inp_id = id(inp_data)

                # Accumulate gradient
                if inp_id in grad_map:
                    grad_map[inp_id] = grad_map[inp_id] + grad_in
                else:
                    grad_map[inp_id] = grad_in

                # Check if this is a source we're computing gradients for
                if inp_id in source_ids:
                    idx = source_ids[inp_id]
                    if grads[idx] is None:
                        grads[idx] = grad_in.copy()
                    else:
                        grads[idx] += grad_in

        if not self.persistent:
            self._operations.clear()

        return grads

    @contextmanager
    def stop_recording(self):
        """Temporarily stop recording operations."""
        old_enabled = self._enabled
        self._enabled = False
        try:
            yield
        finally:
            self._enabled = old_enabled


class ComputationNode:
    """
    A node in the computation graph that tracks forward/backward operations.

    This wraps numpy arrays to enable automatic gradient tracking.
    """

    def __init__(
        self,
        data: np.ndarray,
        requires_grad: bool = False,
        grad_fn: Callable = None,
        inputs: list["ComputationNode"] = None,
        name: str = None,
    ):
        """Initialize the instance."""

        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = grad_fn  # Function to compute gradient
        self.inputs = inputs or []
        self.name = name
        self._backward_hooks = []

    @property
    def shape(self):
        """Execute shape."""

        return self.data.shape

    @property
    def dtype(self):
        """Execute dtype."""

        return self.data.dtype

    def backward(self, grad_output: np.ndarray = None):
        """
        Compute gradients via backpropagation.

        Args:
            grad_output: Gradient w.r.t. this node (default: 1.0 for scalar loss)
        """
        if not self.requires_grad:
            return

        if grad_output is None:
            if self.data.size == 1:
                grad_output = np.ones_like(self.data, dtype=np.float32)
            else:
                raise ValueError("grad_output must be specified for non-scalar tensors")

        # Accumulate gradient
        if self.grad is None:
            self.grad = grad_output.copy()
        else:
            self.grad += grad_output

        # Call backward hooks
        for hook in self._backward_hooks:
            hook(self.grad)

        # Propagate gradients to inputs
        if self.grad_fn is not None:
            grad_inputs = self.grad_fn(grad_output)
            if not isinstance(grad_inputs, (list, tuple)):
                grad_inputs = [grad_inputs]

            for inp, grad_in in zip(self.inputs, grad_inputs):
                if grad_in is not None and hasattr(inp, "backward"):
                    inp.backward(grad_in)

    def zero_grad(self):
        """Clear gradient."""
        self.grad = None

    def detach(self) -> "ComputationNode":
        """Return a new node without gradient tracking."""
        return ComputationNode(self.data.copy(), requires_grad=False)

    def numpy(self) -> np.ndarray:
        """Return underlying numpy array."""
        return self.data

    def register_hook(self, hook: Callable):
        """Register a backward hook."""
        self._backward_hooks.append(hook)

    def __repr__(self):
        """Return a debug representation."""

        name_str = f", name='{self.name}'" if self.name else ""
        return f"ComputationNode(shape={self.shape}, requires_grad={self.requires_grad}{name_str})"


class ModuleTracer:
    """
    Traces forward pass through nn.Modules and enables automatic backward.

    This integrates with existing Module classes to provide automatic gradient
    computation without modifying module code.
    """

    def __init__(self, module, backend=None):
        """
        Args:
            module: The nn.Module to trace
            backend: Optional VulkanCompute backend for GPU operations
        """
        self.module = module
        self.backend = backend
        self._forward_cache: dict[int, dict] = {}  # module_id -> {input, output}
        self._backward_order: list = []  # Modules in backward order

    def __call__(self, *args, **kwargs):
        """
        Traced forward pass that caches inputs for backward.
        """
        self._forward_cache.clear()
        self._backward_order.clear()
        return self._traced_forward(self.module, *args, **kwargs)

    def _traced_forward(self, module, *args, **kwargs):
        """Recursively trace forward pass through submodules."""
        # Cache input
        cache_entry = {"input": args[0] if len(args) == 1 else args, "module": module}

        # Get submodules to trace
        if hasattr(module, "_modules") and module._modules:
            # This is a container module - trace submodules
            x = args[0]
            for name, submodule in module._modules.items():
                x = self._traced_forward(submodule, x)
            output = x
        else:
            # Leaf module - call forward directly
            output = module(*args, **kwargs)

        cache_entry["output"] = output
        self._forward_cache[id(module)] = cache_entry
        self._backward_order.append(module)

        return output

    def backward(self, grad_output: np.ndarray):
        """
        Backward pass through traced modules.

        Args:
            grad_output: Gradient w.r.t. final output
        """
        grad = grad_output

        # Backward through modules in reverse order
        for module in reversed(self._backward_order):
            cache = self._forward_cache.get(id(module))
            if cache is None:
                continue

            x = cache["input"]
            if isinstance(x, tuple):
                x = x[0]

            # Call module's backward method
            if hasattr(module, "backward"):
                grad = module.backward(grad, x)
            else:
                # No backward method - assume identity gradient
                pass

        return grad


class AutogradEngine:
    """
    Central engine for managing automatic differentiation.

    Provides a PyTorch-like interface for gradient computation with
    GPU-accelerated backward passes via Vulkan shaders.
    """

    _instance = None

    def __new__(cls):
        """Create and return a new instance."""

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the instance."""

        if self._initialized:
            return
        self._initialized = True
        self._backend = None
        self._grad_enabled = True
        self._no_grad_level = 0

    @property
    def backend(self):
        """Lazy initialization of Vulkan backend."""
        if self._backend is None:
            try:
                from grilly import Compute

                self._backend = Compute()
            except Exception:
                pass
        return self._backend

    @contextmanager
    def no_grad(self):
        """Context manager to disable gradient computation."""
        self._no_grad_level += 1
        old_enabled = self._grad_enabled
        self._grad_enabled = False
        try:
            yield
        finally:
            self._no_grad_level -= 1
            if self._no_grad_level == 0:
                self._grad_enabled = old_enabled

    @contextmanager
    def enable_grad(self):
        """Context manager to enable gradient computation."""
        old_enabled = self._grad_enabled
        self._grad_enabled = True
        try:
            yield
        finally:
            self._grad_enabled = old_enabled

    def is_grad_enabled(self) -> bool:
        """Check if gradient computation is enabled."""
        return self._grad_enabled

    def backward(
        self,
        tensors: np.ndarray | list[np.ndarray],
        grad_tensors: np.ndarray | list[np.ndarray] = None,
        retain_graph: bool = False,
        create_graph: bool = False,
    ):
        """
        Compute gradients for tensors.

        Args:
            tensors: Tensors to compute gradients for (usually loss)
            grad_tensors: Initial gradients (default: ones)
            retain_graph: Keep computation graph for multiple backward calls
            create_graph: Create graph for higher-order derivatives
        """
        if not isinstance(tensors, list):
            tensors = [tensors]
        if grad_tensors is None:
            grad_tensors = [np.ones_like(t, dtype=np.float32) for t in tensors]
        elif not isinstance(grad_tensors, list):
            grad_tensors = [grad_tensors]

        for tensor, grad in zip(tensors, grad_tensors):
            if hasattr(tensor, "backward"):
                tensor.backward(grad)


# Global autograd engine instance
_engine = AutogradEngine()


# Convenience functions
def no_grad():
    """Context manager to disable gradient computation."""
    return _engine.no_grad()


def enable_grad():
    """Context manager to enable gradient computation."""
    return _engine.enable_grad()


def is_grad_enabled() -> bool:
    """Check if gradient computation is enabled."""
    return _engine.is_grad_enabled()


def backward(tensors, grad_tensors=None, retain_graph=False, create_graph=False):
    """Compute gradients via backpropagation."""
    return _engine.backward(tensors, grad_tensors, retain_graph, create_graph)


# ============================================================================
# GPU-Accelerated Backward Operations
# ============================================================================


class VulkanBackwardOps:
    """
    GPU-accelerated backward pass operations using Vulkan shaders.
    """

    def __init__(self, backend=None):
        """Initialize the instance."""

        self._backend = backend

    @property
    def backend(self):
        """Execute backend."""

        if self._backend is None:
            try:
                from grilly import Compute

                self._backend = Compute()
            except Exception:
                pass
        return self._backend

    def linear_backward(
        self, grad_output: np.ndarray, x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        GPU-accelerated linear layer backward pass.

        Uses fnn-linear-backward.glsl shader.

        Args:
            grad_output: Gradient w.r.t. output (batch, out_features)
            x: Input from forward pass (batch, in_features)
            weight: Weight matrix (out_features, in_features)
            bias: Optional bias vector (out_features,)

        Returns:
            Tuple of (grad_input, grad_weight, grad_bias)
        """
        backend = self.backend
        if backend and hasattr(backend, "fnn") and hasattr(backend.fnn, "linear_backward"):
            try:
                return backend.fnn.linear_backward(grad_output, x, weight, bias)
            except Exception:
                pass

        # CPU fallback
        grad_input = grad_output @ weight
        grad_weight = grad_output.T @ x
        grad_bias = np.sum(grad_output, axis=0) if bias is not None else None
        return grad_input, grad_weight, grad_bias

    def gelu_backward(self, grad_output: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated GELU backward pass.

        Uses activation-gelu-backward.glsl shader.
        """
        backend = self.backend
        if backend and hasattr(backend, "activation_gelu_backward"):
            try:
                return backend.activation_gelu_backward(grad_output, x)
            except Exception:
                pass

        # CPU fallback: d/dx[GELU(x)] = d/dx[x * Phi(x)]
        # where Phi is the CDF of standard normal
        from scipy.special import erf

        sqrt_2 = np.sqrt(2.0)
        sqrt_2_pi = np.sqrt(2.0 / np.pi)

        # Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
        phi = 0.5 * (1.0 + erf(x / sqrt_2))
        # phi'(x) = exp(-x^2/2) / sqrt(2*pi)
        phi_prime = np.exp(-0.5 * x * x) * sqrt_2_pi * 0.5

        # d/dx[x * Phi(x)] = Phi(x) + x * phi'(x)
        grad_input = grad_output * (phi + x * phi_prime)
        return grad_input.astype(np.float32)

    def relu_backward(self, grad_output: np.ndarray, x: np.ndarray) -> np.ndarray:
        """ReLU backward: gradient is 1 where x > 0, else 0."""
        return grad_output * (x > 0).astype(np.float32)

    def silu_backward(self, grad_output: np.ndarray, x: np.ndarray) -> np.ndarray:
        """SiLU (Swish) backward: d/dx[x * sigmoid(x)]."""
        sig = 1.0 / (1.0 + np.exp(-x))
        grad_input = grad_output * (sig + x * sig * (1 - sig))
        return grad_input.astype(np.float32)

    def softmax_backward(
        self, grad_output: np.ndarray, softmax_output: np.ndarray, dim: int = -1
    ) -> np.ndarray:
        """
        Softmax backward pass.

        For softmax output s, the Jacobian is diag(s) - s @ s.T
        """
        # Efficient computation: grad_input = s * (grad_output - sum(grad_output * s, dim))
        sum_term = np.sum(grad_output * softmax_output, axis=dim, keepdims=True)
        grad_input = softmax_output * (grad_output - sum_term)
        return grad_input.astype(np.float32)

    def layernorm_backward(
        self,
        grad_output: np.ndarray,
        x: np.ndarray,
        gamma: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
        eps: float = 1e-5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        LayerNorm backward pass.

        Args:
            grad_output: Gradient w.r.t. output
            x: Input tensor
            gamma: Scale parameter
            mean: Mean computed in forward
            var: Variance computed in forward
            eps: Epsilon for numerical stability

        Returns:
            Tuple of (grad_input, grad_gamma, grad_beta)
        """
        # Normalized input
        std = np.sqrt(var + eps)
        x_norm = (x - mean) / std

        # Gradients w.r.t. gamma and beta
        grad_gamma = np.sum(grad_output * x_norm, axis=0)
        grad_beta = np.sum(grad_output, axis=0)

        # Gradient w.r.t. input
        N = x.shape[-1]
        dx_norm = grad_output * gamma

        # d(x_norm)/dx = (1/std) * (I - (1/N) - x_norm * x_norm.T / N)
        dvar = np.sum(dx_norm * (x - mean) * (-0.5) * (var + eps) ** (-1.5), axis=-1, keepdims=True)
        dmean = np.sum(dx_norm * (-1.0 / std), axis=-1, keepdims=True) + dvar * np.mean(
            -2.0 * (x - mean), axis=-1, keepdims=True
        )

        grad_input = dx_norm / std + dvar * 2.0 * (x - mean) / N + dmean / N

        return (
            grad_input.astype(np.float32),
            grad_gamma.astype(np.float32),
            grad_beta.astype(np.float32),
        )

    def attention_backward(
        self,
        grad_output: np.ndarray,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        attn_weights: np.ndarray,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Multi-head attention backward pass.

        Args:
            grad_output: Gradient w.r.t. attention output
            q, k, v: Query, Key, Value tensors
            attn_weights: Attention weights from forward pass
            scale: Scaling factor (1/sqrt(head_dim))

        Returns:
            Tuple of (grad_q, grad_k, grad_v)
        """
        # grad_v = attn_weights.T @ grad_output
        grad_v = np.matmul(attn_weights.swapaxes(-2, -1), grad_output)

        # grad_attn_weights = grad_output @ v.T
        grad_attn = np.matmul(grad_output, v.swapaxes(-2, -1))

        # Softmax backward
        grad_scores = self.softmax_backward(grad_attn, attn_weights)
        grad_scores = grad_scores * scale

        # grad_q = grad_scores @ k
        grad_q = np.matmul(grad_scores, k)

        # grad_k = grad_scores.T @ q
        grad_k = np.matmul(grad_scores.swapaxes(-2, -1), q)

        return grad_q.astype(np.float32), grad_k.astype(np.float32), grad_v.astype(np.float32)


# Global backward ops instance
backward_ops = VulkanBackwardOps()


# ============================================================================
# Training Context Manager
# ============================================================================


class TrainingContext:
    """
    Context manager for training that handles forward/backward caching.

    Usage:
        with TrainingContext(model) as ctx:
            for batch in dataloader:
                output = ctx.forward(x)
                loss = loss_fn(output, target)
                ctx.backward(loss)
                optimizer.step()
                ctx.zero_grad()
    """

    def __init__(self, model):
        """Initialize the instance."""

        self.model = model
        self.tracer = ModuleTracer(model)
        self._last_output = None
        self._last_loss = None

    def __enter__(self):
        """Enter the runtime context."""

        self.model.train()
        return self

    def __exit__(self, *args):
        """Exit the runtime context."""

        pass

    def forward(self, *args, **kwargs) -> np.ndarray:
        """Traced forward pass."""
        self._last_output = self.tracer(*args, **kwargs)
        return self._last_output

    def backward(self, loss_or_grad: np.ndarray):
        """
        Backward pass through the model.

        Args:
            loss_or_grad: Either a scalar loss or gradient tensor
        """
        if loss_or_grad.size == 1:
            # Scalar loss - start with gradient of 1.0
            grad = np.ones_like(self._last_output, dtype=np.float32)
        else:
            grad = loss_or_grad

        self.tracer.backward(grad)

    def zero_grad(self):
        """Clear all gradients."""
        self.model.zero_grad()
