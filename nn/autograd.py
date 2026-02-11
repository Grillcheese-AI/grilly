"""
Automatic Differentiation System (Autograd)

This module provides automatic differentiation capabilities similar to PyTorch's autograd.
It tracks operations during the forward pass and automatically computes gradients during backward.

Key Features:
- Computation graph construction during forward pass
- Automatic gradient computation via reverse-mode autodiff
- Support for higher-order gradients
- Memory-efficient gradient accumulation
- Operator overloading for natural syntax
"""

from collections.abc import Callable
from typing import Optional

import numpy as np

# Global flag to disable gradient computation
_grad_enabled = True


def is_grad_enabled() -> bool:
    """Check if gradient computation is enabled."""
    return _grad_enabled


class no_grad:
    """Context manager to disable gradient computation."""

    def __enter__(self):
        """Enter the runtime context."""

        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *args):
        """Exit the runtime context."""

        global _grad_enabled
        _grad_enabled = self._prev


class enable_grad:
    """Context manager to enable gradient computation."""

    def __enter__(self):
        """Enter the runtime context."""

        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = True
        return self

    def __exit__(self, *args):
        """Exit the runtime context."""

        global _grad_enabled
        _grad_enabled = self._prev


class GradFn:
    """
    Represents a node in the computation graph.
    Stores the backward function and references to input variables.

    Now supports GPU-accelerated backward pass via optional gpu_backward_fn.

    Note: We use strong references to inputs to prevent garbage collection
    of intermediate variables before backward() is called.
    """

    def __init__(
        self,
        name: str,
        backward_fn: Callable,
        inputs: list["Variable"],
        gpu_backward_fn: Callable | None = None,
    ):
        """Initialize the instance."""

        self.name = name
        self.backward_fn = backward_fn  # CPU fallback
        self.gpu_backward_fn = gpu_backward_fn  # GPU version (optional)
        # Use strong references to prevent GC of intermediate variables
        self._inputs = list(inputs)
        self._next_functions: list[GradFn | None] = []

        # Build the graph edges
        for v in inputs:
            if v is not None and v.grad_fn is not None:
                self._next_functions.append(v.grad_fn)
            else:
                self._next_functions.append(None)

    @property
    def inputs(self) -> list[Optional["Variable"]]:
        """Get input variables."""
        return self._inputs

    @property
    def has_gpu_backward(self) -> bool:
        """Check if GPU backward is available for this operation."""
        return self.gpu_backward_fn is not None

    def __repr__(self):
        """Return a debug representation."""

        gpu_str = " (GPU)" if self.has_gpu_backward else ""
        return f"<{self.name}Backward{gpu_str}>"


class Variable:
    """
    A variable that tracks computation history for automatic differentiation.

    Similar to PyTorch's Tensor with requires_grad=True. Supports operator
    overloading for natural mathematical expressions.

    Example:
        >>> x = Variable([1, 2, 3], requires_grad=True)
        >>> y = x * 2 + 1
        >>> z = y.sum()
        >>> z.backward()
        >>> print(x.grad)  # [2, 2, 2]
    """

    def __init__(
        self,
        data: np.ndarray | float | int | list,
        requires_grad: bool = False,
        grad_fn: GradFn | None = None,
    ):
        """
        Args:
            data: The actual data (numpy array, scalar, or list)
            requires_grad: Whether to track gradients for this variable
            grad_fn: Gradient function for non-leaf variables
        """
        if isinstance(data, Variable):
            data = data.data
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)

        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self.grad_fn = grad_fn
        self._is_leaf = grad_fn is None

    @property
    def is_leaf(self) -> bool:
        """A variable is a leaf if it was created by the user (not by an operation)."""
        return self._is_leaf

    @property
    def shape(self) -> tuple:
        """Shape of the data."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.data.ndim

    def item(self) -> float:
        """Get scalar value."""
        return float(self.data.item())

    def numpy(self) -> np.ndarray:
        """Get numpy array (detached from graph)."""
        return self.data.copy()

    def detach(self) -> "Variable":
        """Return a new Variable detached from the computation graph."""
        return Variable(self.data.copy(), requires_grad=False)

    def zero_grad(self):
        """Clear the gradient."""
        self.grad = None

    def backward(
        self,
        grad_output: np.ndarray | None = None,
        retain_graph: bool = False,
        use_gpu: bool = True,
    ):
        """
        Compute gradients via reverse-mode automatic differentiation.

        This traverses the computation graph in reverse topological order,
        computing and accumulating gradients for all variables with requires_grad=True.

        Args:
            grad_output: Gradient w.r.t. this variable. Default is ones_like for all outputs.
            retain_graph: If True, keep the computation graph for future backward passes.
            use_gpu: If True, use GPU-accelerated backward when available (default: True).
        """
        if not self.requires_grad:
            return

        if grad_output is None:
            # Default gradient is ones (identity gradient)
            grad_output = np.ones_like(self.data, dtype=np.float32)

        # Initialize GPU backend if requested
        gpu_ops = None
        if use_gpu:
            try:
                from grilly.nn.gpu_backward import get_gpu_backward_ops

                gpu_ops = get_gpu_backward_ops(use_gpu=True)
                if not gpu_ops.is_available():
                    gpu_ops = None
                    use_gpu = False
            except Exception:
                # Fall back to CPU if GPU not available
                gpu_ops = None
                use_gpu = False

        # Build topological order of the computation graph
        topo_order = []
        visited = set()

        def build_topo(var: Variable):
            """Execute build topo."""

            if var in visited:
                return
            visited.add(var)
            if var.grad_fn is not None:
                for input_var in var.grad_fn.inputs:
                    if input_var is not None and input_var.requires_grad:
                        build_topo(input_var)
            topo_order.append(var)

        build_topo(self)

        # Initialize gradient for the output
        grad_map = {id(self): grad_output.copy()}

        # Traverse in reverse topological order
        for var in reversed(topo_order):
            var_id = id(var)
            if var_id not in grad_map:
                continue

            grad = grad_map[var_id]

            # Accumulate gradient in leaf variables
            if var.is_leaf and var.requires_grad:
                if var.grad is None:
                    var.grad = grad.copy()
                else:
                    var.grad = var.grad + grad

            # Propagate gradients through the computation graph
            if var.grad_fn is not None:
                # Try GPU backward first if available
                if use_gpu and gpu_ops and var.grad_fn.has_gpu_backward:
                    try:
                        input_grads = var.grad_fn.gpu_backward_fn(gpu_ops, grad)
                    except Exception as e:
                        # Fall back to CPU on error
                        import logging

                        logging.getLogger(__name__).debug(
                            f"GPU backward failed for {var.grad_fn.name}: {e}. Using CPU fallback."
                        )
                        input_grads = var.grad_fn.backward_fn(grad)
                else:
                    # CPU backward
                    input_grads = var.grad_fn.backward_fn(grad)

                if not isinstance(input_grads, tuple):
                    input_grads = (input_grads,)

                # Distribute gradients to input variables
                for input_var, input_grad in zip(var.grad_fn.inputs, input_grads):
                    if input_var is not None and input_grad is not None and input_var.requires_grad:
                        input_id = id(input_var)
                        # Handle broadcasting: sum over broadcasted dimensions
                        input_grad = _unbroadcast(input_grad, input_var.shape)
                        if input_id in grad_map:
                            grad_map[input_id] = grad_map[input_id] + input_grad
                        else:
                            grad_map[input_id] = input_grad

    def __repr__(self):
        """Return a debug representation."""

        grad_str = f", grad_fn={self.grad_fn}" if self.grad_fn else ""
        return f"Variable({self.data}, requires_grad={self.requires_grad}{grad_str})"

    def __str__(self):
        """Return a human-readable representation."""

        return str(self.data)

    # ========================================================================
    # Operator Overloading
    # ========================================================================

    def __add__(self, other):
        """Execute add."""

        return add(self, other)

    def __radd__(self, other):
        """Execute radd."""

        return add(other, self)

    def __sub__(self, other):
        """Execute sub."""

        return sub(self, other)

    def __rsub__(self, other):
        """Execute rsub."""

        return sub(other, self)

    def __mul__(self, other):
        """Execute mul."""

        return mul(self, other)

    def __rmul__(self, other):
        """Execute rmul."""

        return mul(other, self)

    def __truediv__(self, other):
        """Execute truediv."""

        return div(self, other)

    def __rtruediv__(self, other):
        """Execute rtruediv."""

        return div(other, self)

    def __pow__(self, exponent):
        """Execute pow."""

        return pow(self, exponent)

    def __neg__(self):
        """Execute neg."""

        return neg(self)

    def __matmul__(self, other):
        """Execute matmul."""

        return matmul(self, other)

    def __getitem__(self, key):
        """Execute getitem."""

        return index(self, key)

    # Comparison operators (no gradients)
    def __lt__(self, other):
        """Execute lt."""

        other_data = other.data if isinstance(other, Variable) else other
        return self.data < other_data

    def __le__(self, other):
        """Execute le."""

        other_data = other.data if isinstance(other, Variable) else other
        return self.data <= other_data

    def __gt__(self, other):
        """Execute gt."""

        other_data = other.data if isinstance(other, Variable) else other
        return self.data > other_data

    def __ge__(self, other):
        """Execute ge."""

        other_data = other.data if isinstance(other, Variable) else other
        return self.data >= other_data

    # Reduction methods
    def sum(self, dim=None, keepdims=False):
        """Execute sum."""

        return sum(self, dim=dim, keepdims=keepdims)

    def mean(self, dim=None, keepdims=False):
        """Execute mean."""

        return mean(self, dim=dim, keepdims=keepdims)

    def max(self, dim=None, keepdims=False):
        """Execute max."""

        return max(self, dim=dim, keepdims=keepdims)

    def min(self, dim=None, keepdims=False):
        """Execute min."""

        return min(self, dim=dim, keepdims=keepdims)

    # Shape methods
    def reshape(self, *shape):
        """Execute reshape."""

        return reshape(self, shape)

    def transpose(self, *dims):
        """Execute transpose."""

        return transpose(self, dims)

    @property
    def T(self):
        """Execute t."""

        return transpose(self)

    def squeeze(self, dim=None):
        """Execute squeeze."""

        return squeeze(self, dim)

    def unsqueeze(self, dim):
        """Execute unsqueeze."""

        return unsqueeze(self, dim)

    # Activation methods
    def relu(self):
        """Execute relu."""

        return relu(self)

    def sigmoid(self):
        """Execute sigmoid."""

        return sigmoid(self)

    def tanh(self):
        """Execute tanh."""

        return tanh(self)

    def exp(self):
        """Execute exp."""

        return exp(self)

    def log(self):
        """Execute log."""

        return log(self)

    def sqrt(self):
        """Execute sqrt."""

        return sqrt(self)

    def abs(self):
        """Execute abs."""

        return abs(self)

    def clamp(self, min_val=None, max_val=None):
        """Execute clamp."""

        return clamp(self, min_val, max_val)

    def gelu(self):
        """Execute gelu."""

        return gelu(self)

    def silu(self):
        """Execute silu."""

        return silu(self)

    def leaky_relu(self, negative_slope=0.01):
        """Execute leaky relu."""

        return leaky_relu(self, negative_slope)

    def elu(self, alpha=1.0):
        """Execute elu."""

        return elu(self, alpha)

    def softplus(self, beta=1.0, threshold=20.0):
        """Execute softplus."""

        return softplus(self, beta, threshold)

    # Trigonometric methods
    def sin(self):
        """Execute sin."""

        return sin(self)

    def cos(self):
        """Execute cos."""

        return cos(self)

    def tan(self):
        """Execute tan."""

        return tan(self)

    def asin(self):
        """Execute asin."""

        return asin(self)

    def acos(self):
        """Execute acos."""

        return acos(self)

    def atan(self):
        """Execute atan."""

        return atan(self)

    # Statistical methods
    def var(self, dim=None, keepdims=False, unbiased=True):
        """Execute var."""

        return var(self, dim, keepdims, unbiased)

    def std(self, dim=None, keepdims=False, unbiased=True):
        """Execute std."""

        return std(self, dim, keepdims, unbiased)

    def norm(self, p=2, dim=None, keepdims=False):
        """Execute norm."""

        return norm(self, p, dim, keepdims)

    # Additional shape methods
    def flatten(self, start_dim=0, end_dim=-1):
        """Execute flatten."""

        return flatten(self, start_dim, end_dim)

    def view(self, *shape):
        """Execute view."""

        return view(self, *shape)

    def expand(self, *sizes):
        """Execute expand."""

        return expand(self, *sizes)

    def repeat(self, *repeats):
        """Execute repeat."""

        return repeat(self, *repeats)

    def permute(self, *dims):
        """Execute permute."""

        return permute(self, *dims)

    def contiguous(self):
        """Execute contiguous."""

        return contiguous(self)

    def clone(self):
        """Execute clone."""

        return clone(self)


# ============================================================================
# Custom Function Base Class (PyTorch-like)
# ============================================================================


class FunctionMeta(type):
    """Metaclass that provides the apply() classmethod for Function subclasses."""

    def __call__(cls, *args, **kwargs):
        # When called, use apply instead of __init__
        """Invoke the callable instance."""

        return cls.apply(*args, **kwargs)


class Function(metaclass=FunctionMeta):
    """
    Base class for creating custom autograd operations.

    Similar to torch.autograd.Function. Subclass this and implement static
    forward() and backward() methods.

    Example:
        >>> class MyReLU(Function):
        ...     @staticmethod
        ...     def forward(ctx, x):
        ...         ctx.save_for_backward(x)
        ...         return np.maximum(x.data, 0)
        ...
        ...     @staticmethod
        ...     def backward(ctx, grad_output):
        ...         x, = ctx.saved_tensors
        ...         return grad_output * (x.data > 0).astype(np.float32)
        >>>
        >>> x = Variable([1, -2, 3], requires_grad=True)
        >>> y = MyReLU.apply(x)
        >>> # Or equivalently: y = MyReLU(x)
    """

    @staticmethod
    def forward(ctx: "FunctionCtx", *args, **kwargs) -> np.ndarray:
        """
        Compute the forward pass.

        Args:
            ctx: Context object to save tensors for backward pass
            *args: Input Variables
            **kwargs: Additional keyword arguments

        Returns:
            numpy array result (will be wrapped in Variable automatically)
        """
        raise NotImplementedError("Subclasses must implement forward()")

    @staticmethod
    def backward(ctx: "FunctionCtx", grad_output: np.ndarray) -> tuple:
        """
        Compute gradients for inputs.

        Args:
            ctx: Context object with saved tensors
            grad_output: Gradient w.r.t. the output

        Returns:
            Tuple of gradients for each input, or single gradient if one input.
            Use None for inputs that don't need gradients.
        """
        raise NotImplementedError("Subclasses must implement backward()")

    @classmethod
    def apply(cls, *inputs, **kwargs) -> Variable:
        """
        Apply the custom function.

        This is the main entry point. It handles:
        1. Creating the context object
        2. Calling forward()
        3. Setting up the computation graph for backward()

        Args:
            *inputs: Input Variables or arrays
            **kwargs: Additional arguments passed to forward()

        Returns:
            Variable containing the result
        """
        # Convert inputs to Variables
        var_inputs = [_ensure_variable(inp) for inp in inputs]

        # Create context for saving tensors
        ctx = FunctionCtx()

        # Call forward
        result_data = cls.forward(ctx, *var_inputs, **kwargs)

        # Ensure result is numpy array
        if isinstance(result_data, Variable):
            result_data = result_data.data
        elif not isinstance(result_data, np.ndarray):
            result_data = np.array(result_data, dtype=np.float32)

        # Check if any input requires gradient
        requires_grad = any(v.requires_grad for v in var_inputs)

        # Create backward function
        def backward_fn(grad):
            """Execute backward fn."""

            result = cls.backward(ctx, grad)
            if not isinstance(result, tuple):
                result = (result,)
            return result

        # Create grad_fn if needed
        grad_fn = None
        if requires_grad and _grad_enabled:
            grad_fn = GradFn(cls.__name__, backward_fn, var_inputs)

        return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)


class FunctionCtx:
    """
    Context object for custom Functions.

    Used to save tensors during forward pass for use in backward pass.
    """

    def __init__(self):
        """Initialize the instance."""

        self.saved_tensors: tuple[Variable, ...] = ()
        self._saved_data: tuple[np.ndarray, ...] = ()

    def save_for_backward(self, *tensors):
        """
        Save tensors for backward computation.

        Args:
            *tensors: Variables or numpy arrays to save
        """
        saved = []
        data = []
        for t in tensors:
            if isinstance(t, Variable):
                saved.append(t)
                data.append(t.data.copy())
            else:
                saved.append(None)
                data.append(np.array(t, dtype=np.float32) if t is not None else None)
        self.saved_tensors = tuple(saved)
        self._saved_data = tuple(data)

    @property
    def needs_input_grad(self) -> tuple[bool, ...]:
        """
        Check which inputs need gradients.

        Returns:
            Tuple of bools indicating which saved tensors need gradients
        """
        return tuple(
            t.requires_grad if isinstance(t, Variable) else False for t in self.saved_tensors
        )


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Sum out broadcasted dimensions to match target shape.

    When a smaller tensor is broadcast to a larger tensor during forward pass,
    we need to sum the gradient over the broadcasted dimensions.
    """
    if grad.shape == shape:
        return grad

    # Handle scalar case
    if shape == () or shape == (1,):
        return np.sum(grad).reshape(shape)

    # Sum over leading dimensions that were broadcast
    ndim_diff = grad.ndim - len(shape)
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)))

    # Sum over dimensions that were broadcast (size 1 -> size n)
    for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, shape)):
        if target_dim == 1 and grad_dim > 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad.reshape(shape)


def _ensure_variable(x, copy=False) -> Variable:
    """Convert input to Variable if needed. Does NOT copy existing Variables."""
    if isinstance(x, Variable):
        return x  # Return the same object to preserve identity in computation graph
    return Variable(x, requires_grad=False)


def _make_backward(
    name: str,
    inputs: list[Variable],
    backward_fn: Callable,
    gpu_backward_fn: Callable | None = None,
) -> GradFn | None:
    """
    Create a GradFn if any input requires gradient.

    Args:
        name: Operation name
        inputs: Input variables
        backward_fn: CPU backward function
        gpu_backward_fn: Optional GPU backward function

    Returns:
        GradFn or None if gradients disabled
    """
    if not _grad_enabled:
        return None
    if not any(v.requires_grad for v in inputs if isinstance(v, Variable)):
        return None
    return GradFn(name, backward_fn, inputs, gpu_backward_fn=gpu_backward_fn)


# ============================================================================
# Arithmetic Operations
# ============================================================================


def add(a, b) -> Variable:
    """Element-wise addition: a + b"""
    a = _ensure_variable(a)
    b = _ensure_variable(b)

    result_data = a.data + b.data

    def backward(grad):
        """Run backward."""

        return grad, grad

    grad_fn = _make_backward("Add", [a, b], backward)
    requires_grad = a.requires_grad or b.requires_grad

    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)


def sub(a, b) -> Variable:
    """Element-wise subtraction: a - b"""
    a = _ensure_variable(a)
    b = _ensure_variable(b)

    result_data = a.data - b.data

    def backward(grad):
        """Run backward."""

        return grad, -grad

    grad_fn = _make_backward("Sub", [a, b], backward)
    requires_grad = a.requires_grad or b.requires_grad

    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)


def mul(a, b) -> Variable:
    """Element-wise multiplication: a * b"""
    a = _ensure_variable(a)
    b = _ensure_variable(b)

    result_data = a.data * b.data

    # Save for backward
    a_data, b_data = a.data, b.data

    def backward(grad):
        """Run backward."""

        return grad * b_data, grad * a_data

    grad_fn = _make_backward("Mul", [a, b], backward)
    requires_grad = a.requires_grad or b.requires_grad

    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)


def div(a, b) -> Variable:
    """Element-wise division: a / b"""
    a = _ensure_variable(a)
    b = _ensure_variable(b)

    result_data = a.data / b.data

    a_data, b_data = a.data, b.data

    def backward(grad):
        """Run backward."""

        grad_a = grad / b_data
        grad_b = -grad * a_data / (b_data**2)
        return grad_a, grad_b

    grad_fn = _make_backward("Div", [a, b], backward)
    requires_grad = a.requires_grad or b.requires_grad

    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)


def neg(a) -> Variable:
    """Negation: -a"""
    a = _ensure_variable(a)

    result_data = -a.data

    def backward(grad):
        """Run backward."""

        return (-grad,)

    grad_fn = _make_backward("Neg", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def pow(a, exponent) -> Variable:
    """Power: a ** exponent (exponent is a constant)"""
    a = _ensure_variable(a)

    if isinstance(exponent, Variable):
        exp_val = exponent.data
    else:
        exp_val = exponent

    result_data = np.power(a.data, exp_val)

    a_data = a.data

    def backward(grad):
        # d/da (a^n) = n * a^(n-1)
        """Run backward."""

        grad_a = grad * exp_val * np.power(a_data, exp_val - 1)
        return (grad_a,)

    grad_fn = _make_backward("Pow", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def matmul(a, b) -> Variable:
    """Matrix multiplication: a @ b (with GPU backward support)"""
    a = _ensure_variable(a)
    b = _ensure_variable(b)

    result_data = np.matmul(a.data, b.data)

    a_data, b_data = a.data, b.data

    # CPU backward (fallback)
    def backward(grad):
        """Run backward."""

        if a_data.ndim == 1 and b_data.ndim == 1:
            # Vector dot product
            grad_a = grad * b_data
            grad_b = grad * a_data
        elif a_data.ndim == 1:
            # Vector @ Matrix
            grad_a = np.matmul(grad, b_data.T)
            grad_b = np.outer(a_data, grad)
        elif b_data.ndim == 1:
            # Matrix @ Vector
            grad_a = np.outer(grad, b_data)
            grad_b = np.matmul(a_data.T, grad)
        else:
            # Matrix @ Matrix
            grad_a = np.matmul(grad, np.swapaxes(b_data, -2, -1))
            grad_b = np.matmul(np.swapaxes(a_data, -2, -1), grad)
        return grad_a, grad_b

    # GPU backward (NEW)
    def gpu_backward(gpu_ops, grad):
        # For matrix multiplication, use linear backward shader
        # This assumes b is the weight matrix (typical in neural networks)
        """Run gpu backward."""

        if a_data.ndim == 2 and b_data.ndim == 2:
            # Treat as: output = a @ b.T (linear layer convention)
            # grad_a = grad @ b, grad_b = grad.T @ a
            grad_a, grad_b, _ = gpu_ops.linear_backward(
                grad,
                a_data,
                b_data.T,
                compute_input_grad=True,
                compute_weight_grad=True,
                compute_bias_grad=False,
            )
            # Transpose grad_b back
            if grad_b is not None:
                grad_b = grad_b.T
            return grad_a, grad_b
        else:
            # Fall back to CPU for non-2D cases
            return backward(grad)

    grad_fn = _make_backward("MatMul", [a, b], backward, gpu_backward_fn=gpu_backward)
    requires_grad = a.requires_grad or b.requires_grad

    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)


# ============================================================================
# Reduction Operations
# ============================================================================


def sum(a, dim=None, keepdims=False) -> Variable:
    """Sum over dimensions."""
    a = _ensure_variable(a)

    result_data = np.sum(a.data, axis=dim, keepdims=keepdims)

    input_shape = a.data.shape

    def backward(grad):
        # Expand gradient to match input shape
        """Run backward."""

        if not keepdims and dim is not None:
            grad = np.expand_dims(grad, axis=dim)
        grad_expanded = np.broadcast_to(grad, input_shape).copy()
        return (grad_expanded,)

    grad_fn = _make_backward("Sum", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def mean(a, dim=None, keepdims=False) -> Variable:
    """Mean over dimensions."""
    a = _ensure_variable(a)

    result_data = np.mean(a.data, axis=dim, keepdims=keepdims)

    input_shape = a.data.shape
    if dim is None:
        n = a.data.size
    elif isinstance(dim, int):
        n = input_shape[dim]
    else:
        n = np.prod([input_shape[d] for d in dim])

    def backward(grad):
        """Run backward."""

        if not keepdims and dim is not None:
            grad = np.expand_dims(grad, axis=dim)
        grad_expanded = np.broadcast_to(grad, input_shape).copy() / n
        return (grad_expanded,)

    grad_fn = _make_backward("Mean", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def max(a, dim=None, keepdims=False) -> Variable:
    """Max over dimensions."""
    a = _ensure_variable(a)

    if dim is None:
        result_data = np.max(a.data)
        max_idx = np.unravel_index(np.argmax(a.data), a.data.shape)
    else:
        result_data = np.max(a.data, axis=dim, keepdims=keepdims)
        max_idx = np.argmax(a.data, axis=dim)

    a_data = a.data

    def backward(grad):
        """Run backward."""

        if dim is None:
            grad_input = np.zeros_like(a_data)
            grad_input[max_idx] = grad
        else:
            # Create mask for max values
            if not keepdims:
                grad = np.expand_dims(grad, axis=dim)
            max_expanded = np.max(a_data, axis=dim, keepdims=True)
            mask = (a_data == max_expanded).astype(np.float32)
            # Normalize mask in case of ties
            mask = mask / np.sum(mask, axis=dim, keepdims=True)
            grad_input = grad * mask
        return (grad_input,)

    grad_fn = _make_backward("Max", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def min(a, dim=None, keepdims=False) -> Variable:
    """Min over dimensions."""
    a = _ensure_variable(a)

    if dim is None:
        result_data = np.min(a.data)
        min_idx = np.unravel_index(np.argmin(a.data), a.data.shape)
    else:
        result_data = np.min(a.data, axis=dim, keepdims=keepdims)

    a_data = a.data

    def backward(grad):
        """Run backward."""

        if dim is None:
            grad_input = np.zeros_like(a_data)
            grad_input[min_idx] = grad
        else:
            if not keepdims:
                grad = np.expand_dims(grad, axis=dim)
            min_expanded = np.min(a_data, axis=dim, keepdims=True)
            mask = (a_data == min_expanded).astype(np.float32)
            mask = mask / np.sum(mask, axis=dim, keepdims=True)
            grad_input = grad * mask
        return (grad_input,)

    grad_fn = _make_backward("Min", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


# ============================================================================
# Activation Functions
# ============================================================================


def relu(a) -> Variable:
    """ReLU activation: max(0, a) (with GPU backward support)"""
    a = _ensure_variable(a)

    result_data = np.maximum(a.data, 0)

    a_data = a.data

    # CPU backward (fallback)
    def backward(grad):
        """Run backward."""

        return (grad * (a_data > 0).astype(np.float32),)

    # GPU backward (NEW)
    def gpu_backward(gpu_ops, grad):
        """Run gpu backward."""

        grad_input = gpu_ops.relu_backward(grad, a_data)
        return (grad_input,)

    grad_fn = _make_backward("ReLU", [a], backward, gpu_backward_fn=gpu_backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def sigmoid(a) -> Variable:
    """Sigmoid activation: 1 / (1 + exp(-a))"""
    a = _ensure_variable(a)

    result_data = 1.0 / (1.0 + np.exp(-a.data))

    result = result_data

    def backward(grad):
        """Run backward."""

        return (grad * result * (1 - result),)

    grad_fn = _make_backward("Sigmoid", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def tanh(a) -> Variable:
    """Tanh activation"""
    a = _ensure_variable(a)

    result_data = np.tanh(a.data)

    result = result_data

    def backward(grad):
        """Run backward."""

        return (grad * (1 - result**2),)

    grad_fn = _make_backward("Tanh", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def exp(a) -> Variable:
    """Exponential: exp(a)"""
    a = _ensure_variable(a)

    result_data = np.exp(a.data)

    result = result_data

    def backward(grad):
        """Run backward."""

        return (grad * result,)

    grad_fn = _make_backward("Exp", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def log(a) -> Variable:
    """Natural logarithm: log(a)"""
    a = _ensure_variable(a)

    result_data = np.log(a.data)

    a_data = a.data

    def backward(grad):
        """Run backward."""

        return (grad / a_data,)

    grad_fn = _make_backward("Log", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def sqrt(a) -> Variable:
    """Square root: sqrt(a)"""
    a = _ensure_variable(a)

    result_data = np.sqrt(a.data)

    result = result_data

    def backward(grad):
        """Run backward."""

        return (grad / (2 * result),)

    grad_fn = _make_backward("Sqrt", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def abs(a) -> Variable:
    """Return the elementwise absolute value."""
    a = _ensure_variable(a)

    result_data = np.abs(a.data)

    a_data = a.data

    def backward(grad):
        """Run backward."""

        return (grad * np.sign(a_data),)

    grad_fn = _make_backward("Abs", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def clamp(a, min_val=None, max_val=None) -> Variable:
    """Clamp values to [min_val, max_val]"""
    a = _ensure_variable(a)

    result_data = a.data.copy()
    if min_val is not None:
        result_data = np.maximum(result_data, min_val)
    if max_val is not None:
        result_data = np.minimum(result_data, max_val)

    a_data = a.data

    def backward(grad):
        """Run backward."""

        mask = np.ones_like(a_data)
        if min_val is not None:
            mask = mask * (a_data >= min_val)
        if max_val is not None:
            mask = mask * (a_data <= max_val)
        return (grad * mask,)

    grad_fn = _make_backward("Clamp", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


# ============================================================================
# Shape Operations
# ============================================================================


def reshape(a, shape) -> Variable:
    """Reshape tensor."""
    a = _ensure_variable(a)

    if isinstance(shape, tuple) and len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]

    result_data = a.data.reshape(shape)

    original_shape = a.data.shape

    def backward(grad):
        """Run backward."""

        return (grad.reshape(original_shape),)

    grad_fn = _make_backward("Reshape", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def transpose(a, dims=None) -> Variable:
    """Transpose tensor."""
    a = _ensure_variable(a)

    if dims is None or len(dims) == 0:
        # Default: reverse all dimensions
        result_data = a.data.T
        reverse_dims = None
    else:
        if isinstance(dims, tuple) and len(dims) == 1 and isinstance(dims[0], tuple):
            dims = dims[0]
        result_data = np.transpose(a.data, dims)
        # Compute inverse permutation
        reverse_dims = [0] * len(dims)
        for i, d in enumerate(dims):
            reverse_dims[d] = i
        reverse_dims = tuple(reverse_dims)

    def backward(grad):
        """Run backward."""

        if reverse_dims is None:
            return (grad.T,)
        return (np.transpose(grad, reverse_dims),)

    grad_fn = _make_backward("Transpose", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def squeeze(a, dim=None) -> Variable:
    """Remove dimensions of size 1."""
    a = _ensure_variable(a)

    result_data = np.squeeze(a.data, axis=dim)

    original_shape = a.data.shape

    def backward(grad):
        """Run backward."""

        return (grad.reshape(original_shape),)

    grad_fn = _make_backward("Squeeze", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def unsqueeze(a, dim) -> Variable:
    """Add a dimension of size 1."""
    a = _ensure_variable(a)

    result_data = np.expand_dims(a.data, axis=dim)

    def backward(grad):
        """Run backward."""

        return (np.squeeze(grad, axis=dim),)

    grad_fn = _make_backward("Unsqueeze", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def index(a, key) -> Variable:
    """Index/slice tensor."""
    a = _ensure_variable(a)

    result_data = a.data[key]

    original_shape = a.data.shape

    def backward(grad):
        """Run backward."""

        grad_input = np.zeros(original_shape, dtype=np.float32)
        grad_input[key] = grad
        return (grad_input,)

    grad_fn = _make_backward("Index", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def concat(tensors, dim=0) -> Variable:
    """Concatenate tensors along dimension."""
    tensors = [_ensure_variable(t) for t in tensors]

    result_data = np.concatenate([t.data for t in tensors], axis=dim)

    sizes = [t.data.shape[dim] for t in tensors]
    requires_grad = any(t.requires_grad for t in tensors)

    def backward(grad):
        """Run backward."""

        grads = np.split(grad, np.cumsum(sizes[:-1]), axis=dim)
        return tuple(grads)

    grad_fn = _make_backward("Concat", tensors, backward) if requires_grad else None

    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)


def stack(tensors, dim=0) -> Variable:
    """Stack tensors along new dimension."""
    tensors = [_ensure_variable(t) for t in tensors]

    result_data = np.stack([t.data for t in tensors], axis=dim)

    requires_grad = any(t.requires_grad for t in tensors)

    def backward(grad):
        """Run backward."""

        grads = [np.squeeze(g, axis=dim) for g in np.split(grad, len(tensors), axis=dim)]
        return tuple(grads)

    grad_fn = _make_backward("Stack", tensors, backward) if requires_grad else None

    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)


# ============================================================================
# Special Operations
# ============================================================================


def where(condition, x, y) -> Variable:
    """Element-wise conditional: where(condition, x, y)"""
    x = _ensure_variable(x)
    y = _ensure_variable(y)

    if isinstance(condition, Variable):
        cond = condition.data
    else:
        cond = condition

    result_data = np.where(cond, x.data, y.data)

    def backward(grad):
        """Run backward."""

        grad_x = np.where(cond, grad, 0)
        grad_y = np.where(cond, 0, grad)
        return grad_x, grad_y

    requires_grad = x.requires_grad or y.requires_grad
    grad_fn = _make_backward("Where", [x, y], backward) if requires_grad else None

    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)


def softmax(a, dim=-1) -> Variable:
    """Softmax along dimension."""
    a = _ensure_variable(a)

    # Numerically stable softmax
    shifted = a.data - np.max(a.data, axis=dim, keepdims=True)
    exp_x = np.exp(shifted)
    result_data = exp_x / np.sum(exp_x, axis=dim, keepdims=True)

    result = result_data

    def backward(grad):
        # Softmax gradient: grad_input = softmax * (grad - sum(grad * softmax))
        """Run backward."""

        s = result
        grad_input = s * (grad - np.sum(grad * s, axis=dim, keepdims=True))
        return (grad_input,)

    grad_fn = _make_backward("Softmax", [a], backward)

    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def cross_entropy(logits, targets, dim=-1) -> Variable:
    """Cross-entropy loss (combines log_softmax and nll_loss)."""
    logits = _ensure_variable(logits)

    # Numerically stable log-softmax
    shifted = logits.data - np.max(logits.data, axis=dim, keepdims=True)
    log_softmax = shifted - np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))

    # NLL loss
    if isinstance(targets, Variable):
        targets = targets.data

    if targets.ndim == logits.data.ndim:
        # Soft targets (one-hot or probabilities)
        loss = -np.sum(targets * log_softmax, axis=dim)
    else:
        # Hard targets (class indices)
        batch_size = logits.data.shape[0]
        loss = -log_softmax[np.arange(batch_size), targets.astype(int)]

    result_data = np.mean(loss)

    softmax_output = np.exp(log_softmax)
    logits_data = logits.data

    def backward(grad):
        """Run backward."""

        if targets.ndim == logits_data.ndim:
            grad_input = grad * (softmax_output - targets) / targets.shape[0]
        else:
            grad_input = softmax_output.copy()
            batch_size = logits_data.shape[0]
            grad_input[np.arange(batch_size), targets.astype(int)] -= 1
            grad_input = grad * grad_input / batch_size
        return (grad_input,)

    grad_fn = _make_backward("CrossEntropy", [logits], backward)

    return Variable(result_data, requires_grad=logits.requires_grad, grad_fn=grad_fn)


def mse_loss(pred, target) -> Variable:
    """Mean squared error loss."""
    pred = _ensure_variable(pred)
    target = _ensure_variable(target)

    diff = pred.data - target.data
    result_data = np.mean(diff**2)

    n = pred.data.size

    def backward(grad):
        """Run backward."""

        grad_pred = grad * 2 * diff / n
        grad_target = -grad_pred
        return grad_pred, grad_target

    requires_grad = pred.requires_grad or target.requires_grad
    grad_fn = _make_backward("MSELoss", [pred, target], backward) if requires_grad else None

    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)


# ============================================================================
# Convenience Functions for PyTorch Compatibility
# ============================================================================


def tensor(data, requires_grad=False) -> Variable:
    """Create a Variable (PyTorch-like alias)."""
    return Variable(data, requires_grad=requires_grad)


def zeros(shape, requires_grad=False) -> Variable:
    """Create a Variable filled with zeros."""
    return Variable(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)


def ones(shape, requires_grad=False) -> Variable:
    """Create a Variable filled with ones."""
    return Variable(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)


def randn(*shape, requires_grad=False) -> Variable:
    """Create a Variable with random normal values."""
    if len(shape) == 0:
        # Scalar
        return Variable(np.array(np.random.randn(), dtype=np.float32), requires_grad=requires_grad)
    return Variable(np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad)


def rand(*shape, requires_grad=False) -> Variable:
    """Create a Variable with random uniform values in [0, 1)."""
    if len(shape) == 0:
        # Scalar
        return Variable(np.array(np.random.rand(), dtype=np.float32), requires_grad=requires_grad)
    return Variable(np.random.rand(*shape).astype(np.float32), requires_grad=requires_grad)


def linspace(start, end, steps, requires_grad=False) -> Variable:
    """Create evenly spaced values."""
    return Variable(np.linspace(start, end, steps, dtype=np.float32), requires_grad=requires_grad)


def arange(start, end=None, step=1, requires_grad=False) -> Variable:
    """Create a range of values."""
    if end is None:
        end = start
        start = 0
    return Variable(np.arange(start, end, step, dtype=np.float32), requires_grad=requires_grad)


def eye(n, m=None, requires_grad=False) -> Variable:
    """Create an identity matrix."""
    if m is None:
        m = n
    return Variable(np.eye(n, m, dtype=np.float32), requires_grad=requires_grad)


def full(shape, fill_value, requires_grad=False) -> Variable:
    """Create a Variable filled with a constant value."""
    return Variable(np.full(shape, fill_value, dtype=np.float32), requires_grad=requires_grad)


# ============================================================================
# Additional Activation Functions
# ============================================================================


def gelu(a) -> Variable:
    """GELU activation: x * Phi(x) where Phi is the CDF of standard normal."""
    a = _ensure_variable(a)

    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x = a.data
    sqrt_2_pi = np.sqrt(2.0 / np.pi)
    cdf_approx = 0.5 * (1.0 + np.tanh(sqrt_2_pi * (x + 0.044715 * x**3)))
    result_data = x * cdf_approx

    def backward(grad):
        # d/dx GELU ≈ cdf + x * pdf * (1 - tanh^2(...)) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
        """Run backward."""

        inner = sqrt_2_pi * (x + 0.044715 * x**3)
        tanh_inner = np.tanh(inner)
        sech2 = 1 - tanh_inner**2
        dcdf = 0.5 * sech2 * sqrt_2_pi * (1 + 3 * 0.044715 * x**2)
        grad_x = grad * (cdf_approx + x * dcdf)
        return (grad_x,)

    grad_fn = _make_backward("GELU", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def silu(a) -> Variable:
    """SiLU/Swish activation: x * sigmoid(x) (with GPU backward support)"""
    a = _ensure_variable(a)

    sig = 1.0 / (1.0 + np.exp(-a.data))
    result_data = a.data * sig

    x = a.data
    sigmoid_x = sig

    # CPU backward (fallback)
    def backward(grad):
        # d/dx (x * sigmoid(x)) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        #                       = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        """Run backward."""

        grad_x = grad * sigmoid_x * (1 + x * (1 - sigmoid_x))
        return (grad_x,)

    # GPU backward (NEW)
    def gpu_backward(gpu_ops, grad):
        """Run gpu backward."""

        grad_input = gpu_ops.silu_backward(grad, x)
        return (grad_input,)

    grad_fn = _make_backward("SiLU", [a], backward, gpu_backward_fn=gpu_backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def leaky_relu(a, negative_slope=0.01) -> Variable:
    """Leaky ReLU: max(0, x) + negative_slope * min(0, x)"""
    a = _ensure_variable(a)

    result_data = np.where(a.data >= 0, a.data, negative_slope * a.data)

    a_data = a.data

    def backward(grad):
        """Run backward."""

        grad_x = np.where(a_data >= 0, grad, negative_slope * grad)
        return (grad_x,)

    grad_fn = _make_backward("LeakyReLU", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def elu(a, alpha=1.0) -> Variable:
    """ELU: max(0, x) + alpha * (exp(min(0, x)) - 1)"""
    a = _ensure_variable(a)

    x = a.data
    result_data = np.where(x >= 0, x, alpha * (np.exp(x) - 1))

    def backward(grad):
        """Run backward."""

        grad_x = np.where(x >= 0, grad, grad * alpha * np.exp(x))
        return (grad_x,)

    grad_fn = _make_backward("ELU", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def softplus(a, beta=1.0, threshold=20.0) -> Variable:
    """Softplus: (1/beta) * log(1 + exp(beta * x))"""
    a = _ensure_variable(a)

    x = a.data
    bx = beta * x
    # For numerical stability, use x when beta*x > threshold
    result_data = np.where(bx > threshold, x, np.log1p(np.exp(bx)) / beta)

    def backward(grad):
        # d/dx softplus = sigmoid(beta * x)
        """Run backward."""

        sigmoid_bx = 1.0 / (1.0 + np.exp(-bx))
        grad_x = np.where(bx > threshold, grad, grad * sigmoid_bx)
        return (grad_x,)

    grad_fn = _make_backward("Softplus", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


# ============================================================================
# Trigonometric Functions
# ============================================================================


def sin(a) -> Variable:
    """Sine function."""
    a = _ensure_variable(a)
    result_data = np.sin(a.data)

    a_data = a.data

    def backward(grad):
        """Run backward."""

        return (grad * np.cos(a_data),)

    grad_fn = _make_backward("Sin", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def cos(a) -> Variable:
    """Cosine function."""
    a = _ensure_variable(a)
    result_data = np.cos(a.data)

    a_data = a.data

    def backward(grad):
        """Run backward."""

        return (-grad * np.sin(a_data),)

    grad_fn = _make_backward("Cos", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def tan(a) -> Variable:
    """Tangent function."""
    a = _ensure_variable(a)
    result_data = np.tan(a.data)

    def backward(grad):
        # d/dx tan(x) = sec^2(x) = 1 + tan^2(x)
        """Run backward."""

        return (grad * (1 + result_data**2),)

    grad_fn = _make_backward("Tan", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def asin(a) -> Variable:
    """Arcsine function."""
    a = _ensure_variable(a)
    result_data = np.arcsin(a.data)

    a_data = a.data

    def backward(grad):
        """Run backward."""

        return (grad / np.sqrt(1 - a_data**2),)

    grad_fn = _make_backward("Asin", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def acos(a) -> Variable:
    """Arccosine function."""
    a = _ensure_variable(a)
    result_data = np.arccos(a.data)

    a_data = a.data

    def backward(grad):
        """Run backward."""

        return (-grad / np.sqrt(1 - a_data**2),)

    grad_fn = _make_backward("Acos", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def atan(a) -> Variable:
    """Arctangent function."""
    a = _ensure_variable(a)
    result_data = np.arctan(a.data)

    a_data = a.data

    def backward(grad):
        """Run backward."""

        return (grad / (1 + a_data**2),)

    grad_fn = _make_backward("Atan", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def atan2(y, x) -> Variable:
    """Arctangent of y/x with correct quadrant."""
    y = _ensure_variable(y)
    x = _ensure_variable(x)

    result_data = np.arctan2(y.data, x.data)

    y_data, x_data = y.data, x.data

    def backward(grad):
        """Run backward."""

        denom = x_data**2 + y_data**2
        grad_y = grad * x_data / denom
        grad_x = -grad * y_data / denom
        return grad_y, grad_x

    grad_fn = _make_backward("Atan2", [y, x], backward)
    requires_grad = y.requires_grad or x.requires_grad
    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)


# ============================================================================
# Statistical Functions
# ============================================================================


def var(a, dim=None, keepdims=False, unbiased=True) -> Variable:
    """Variance over dimensions."""
    a = _ensure_variable(a)

    ddof = 1 if unbiased else 0
    result_data = np.var(a.data, axis=dim, keepdims=keepdims, ddof=ddof)

    input_shape = a.data.shape
    mean_val = np.mean(a.data, axis=dim, keepdims=True)
    a_data = a.data

    if dim is None:
        n = a.data.size
    elif isinstance(dim, int):
        n = input_shape[dim]
    else:
        n = np.prod([input_shape[d] for d in dim])

    def backward(grad):
        """Run backward."""

        if not keepdims and dim is not None:
            grad = np.expand_dims(grad, axis=dim)
        # d/dx var = 2 * (x - mean) / (n - ddof)
        grad_input = grad * 2 * (a_data - mean_val) / (n - ddof)
        return (grad_input,)

    grad_fn = _make_backward("Var", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def std(a, dim=None, keepdims=False, unbiased=True) -> Variable:
    """Standard deviation over dimensions."""
    a = _ensure_variable(a)

    ddof = 1 if unbiased else 0
    result_data = np.std(a.data, axis=dim, keepdims=keepdims, ddof=ddof)

    input_shape = a.data.shape
    mean_val = np.mean(a.data, axis=dim, keepdims=True)
    var_val = np.var(a.data, axis=dim, keepdims=True, ddof=ddof)
    a_data = a.data

    if dim is None:
        n = a.data.size
    elif isinstance(dim, int):
        n = input_shape[dim]
    else:
        n = np.prod([input_shape[d] for d in dim])

    def backward(grad):
        """Run backward."""

        if not keepdims and dim is not None:
            grad = np.expand_dims(grad, axis=dim)
        # d/dx std = (x - mean) / (std * (n - ddof))
        std_expanded = np.sqrt(var_val)
        grad_input = grad * (a_data - mean_val) / (std_expanded * (n - ddof) + 1e-8)
        return (grad_input,)

    grad_fn = _make_backward("Std", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def norm(a, p=2, dim=None, keepdims=False) -> Variable:
    """Lp norm over dimensions."""
    a = _ensure_variable(a)

    if p == 2:
        result_data = np.sqrt(np.sum(a.data**2, axis=dim, keepdims=keepdims))
    elif p == 1:
        result_data = np.sum(np.abs(a.data), axis=dim, keepdims=keepdims)
    elif p == float("inf"):
        result_data = np.max(np.abs(a.data), axis=dim, keepdims=keepdims)
    else:
        result_data = np.sum(np.abs(a.data) ** p, axis=dim, keepdims=keepdims) ** (1 / p)

    a_data = a.data

    def backward(grad):
        """Run backward."""

        if not keepdims and dim is not None:
            grad = np.expand_dims(grad, axis=dim)
            result_expanded = np.expand_dims(result_data, axis=dim)
        else:
            result_expanded = result_data

        if p == 2:
            grad_input = grad * a_data / (result_expanded + 1e-8)
        elif p == 1:
            grad_input = grad * np.sign(a_data)
        elif p == float("inf"):
            # Gradient only flows to max elements
            max_val = np.max(np.abs(a_data), axis=dim, keepdims=True)
            mask = (np.abs(a_data) == max_val).astype(np.float32)
            mask = mask / (mask.sum(axis=dim, keepdims=True) + 1e-8)
            grad_input = grad * np.sign(a_data) * mask
        else:
            grad_input = (
                grad
                * (np.abs(a_data) ** (p - 1))
                * np.sign(a_data)
                / (result_expanded ** (p - 1) + 1e-8)
            )

        return (grad_input,)

    grad_fn = _make_backward("Norm", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


# ============================================================================
# Additional Shape Operations
# ============================================================================


def flatten(a, start_dim=0, end_dim=-1) -> Variable:
    """Flatten dimensions from start_dim to end_dim."""
    a = _ensure_variable(a)

    if end_dim < 0:
        end_dim = a.data.ndim + end_dim

    new_shape = list(a.data.shape[:start_dim])
    new_shape.append(-1)
    new_shape.extend(a.data.shape[end_dim + 1 :])

    result_data = a.data.reshape(new_shape)
    original_shape = a.data.shape

    def backward(grad):
        """Run backward."""

        return (grad.reshape(original_shape),)

    grad_fn = _make_backward("Flatten", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def view(a, *shape) -> Variable:
    """View tensor with new shape (alias for reshape)."""
    return reshape(a, shape)


def expand(a, *sizes) -> Variable:
    """Expand tensor to new sizes (broadcast)."""
    a = _ensure_variable(a)

    if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
        sizes = sizes[0]

    result_data = np.broadcast_to(a.data, sizes).copy()

    original_shape = a.data.shape

    def backward(grad):
        # Sum over expanded dimensions
        """Run backward."""

        return (_unbroadcast(grad, original_shape),)

    grad_fn = _make_backward("Expand", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def repeat(a, *repeats) -> Variable:
    """Repeat tensor along dimensions."""
    a = _ensure_variable(a)

    if len(repeats) == 1 and isinstance(repeats[0], (list, tuple)):
        repeats = repeats[0]

    result_data = np.tile(a.data, repeats)

    original_shape = a.data.shape

    def backward(grad):
        # Sum over repeated sections
        # This is a simplified version; full implementation would handle partial repeats
        """Run backward."""

        grad_input = np.zeros(original_shape, dtype=np.float32)
        # For each repeat, add the gradients
        [slice(0, s) for s in original_shape]
        for idx in np.ndindex(*repeats):
            start_idx = [i * s for i, s in zip(idx, original_shape)]
            src_slices = [
                slice(start, start + size) for start, size in zip(start_idx, original_shape)
            ]
            grad_input += grad[tuple(src_slices)]
        return (grad_input,)

    grad_fn = _make_backward("Repeat", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def permute(a, *dims) -> Variable:
    """Permute dimensions (alias for transpose with explicit dims)."""
    if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
        dims = dims[0]
    return transpose(a, dims)


def contiguous(a) -> Variable:
    """Return a contiguous tensor (no-op for numpy, but useful for compatibility)."""
    a = _ensure_variable(a)
    result_data = np.ascontiguousarray(a.data)

    def backward(grad):
        """Run backward."""

        return (grad,)

    grad_fn = _make_backward("Contiguous", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


def clone(a) -> Variable:
    """Clone a tensor, creating a copy that shares the computation graph."""
    a = _ensure_variable(a)
    result_data = a.data.copy()

    def backward(grad):
        """Run backward."""

        return (grad.copy(),)

    grad_fn = _make_backward("Clone", [a], backward)
    return Variable(result_data, requires_grad=a.requires_grad, grad_fn=grad_fn)


# ============================================================================
# Binary Comparisons (with gradients through STE)
# ============================================================================


def eq(a, b) -> Variable:
    """Element-wise equality (no gradient)."""
    a = _ensure_variable(a)
    b = _ensure_variable(b) if isinstance(b, Variable) else b
    b_data = b.data if isinstance(b, Variable) else b
    result_data = (a.data == b_data).astype(np.float32)
    return Variable(result_data, requires_grad=False)


def ne(a, b) -> Variable:
    """Element-wise inequality (no gradient)."""
    a = _ensure_variable(a)
    b_data = b.data if isinstance(b, Variable) else b
    result_data = (a.data != b_data).astype(np.float32)
    return Variable(result_data, requires_grad=False)


def lt(a, b) -> Variable:
    """Element-wise less than (no gradient)."""
    a = _ensure_variable(a)
    b_data = b.data if isinstance(b, Variable) else b
    result_data = (a.data < b_data).astype(np.float32)
    return Variable(result_data, requires_grad=False)


def le(a, b) -> Variable:
    """Element-wise less than or equal (no gradient)."""
    a = _ensure_variable(a)
    b_data = b.data if isinstance(b, Variable) else b
    result_data = (a.data <= b_data).astype(np.float32)
    return Variable(result_data, requires_grad=False)


def gt(a, b) -> Variable:
    """Element-wise greater than (no gradient)."""
    a = _ensure_variable(a)
    b_data = b.data if isinstance(b, Variable) else b
    result_data = (a.data > b_data).astype(np.float32)
    return Variable(result_data, requires_grad=False)


def ge(a, b) -> Variable:
    """Element-wise greater than or equal (no gradient)."""
    a = _ensure_variable(a)
    b_data = b.data if isinstance(b, Variable) else b
    result_data = (a.data >= b_data).astype(np.float32)
    return Variable(result_data, requires_grad=False)


# ============================================================================
# Additional Loss Functions
# ============================================================================


def l1_loss(pred, target, reduction="mean") -> Variable:
    """L1 loss (mean absolute error)."""
    pred = _ensure_variable(pred)
    target = _ensure_variable(target)

    diff = pred.data - target.data
    abs_diff = np.abs(diff)

    if reduction == "mean":
        result_data = np.mean(abs_diff)
        n = pred.data.size
    elif reduction == "sum":
        result_data = np.sum(abs_diff)
        n = 1
    else:  # none
        result_data = abs_diff
        n = 1

    def backward(grad):
        """Run backward."""

        sign = np.sign(diff)
        if reduction == "mean":
            grad_pred = grad * sign / n
        elif reduction == "sum":
            grad_pred = grad * sign
        else:
            grad_pred = grad * sign
        return grad_pred, -grad_pred

    requires_grad = pred.requires_grad or target.requires_grad
    grad_fn = _make_backward("L1Loss", [pred, target], backward) if requires_grad else None
    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)


def smooth_l1_loss(pred, target, beta=1.0, reduction="mean") -> Variable:
    """Smooth L1 loss (Huber loss)."""
    pred = _ensure_variable(pred)
    target = _ensure_variable(target)

    diff = pred.data - target.data
    abs_diff = np.abs(diff)

    # Smooth L1: 0.5 * x^2 / beta if |x| < beta, else |x| - 0.5 * beta
    loss = np.where(abs_diff < beta, 0.5 * diff**2 / beta, abs_diff - 0.5 * beta)

    if reduction == "mean":
        result_data = np.mean(loss)
        n = pred.data.size
    elif reduction == "sum":
        result_data = np.sum(loss)
        n = 1
    else:
        result_data = loss
        n = 1

    def backward(grad):
        # Gradient: x/beta if |x| < beta, else sign(x)
        """Run backward."""

        grad_diff = np.where(abs_diff < beta, diff / beta, np.sign(diff))
        if reduction == "mean":
            grad_pred = grad * grad_diff / n
        else:
            grad_pred = grad * grad_diff
        return grad_pred, -grad_pred

    requires_grad = pred.requires_grad or target.requires_grad
    grad_fn = _make_backward("SmoothL1Loss", [pred, target], backward) if requires_grad else None
    return Variable(result_data, requires_grad=requires_grad, grad_fn=grad_fn)


def bce_loss(pred, target, reduction="mean") -> Variable:
    """Binary cross-entropy loss (pred should be probabilities)."""
    pred = _ensure_variable(pred)
    target = _ensure_variable(target)

    # Clamp predictions for numerical stability
    eps = 1e-7
    pred_clamped = np.clip(pred.data, eps, 1 - eps)
    target_data = target.data

    loss = -target_data * np.log(pred_clamped) - (1 - target_data) * np.log(1 - pred_clamped)

    if reduction == "mean":
        result_data = np.mean(loss)
        n = pred.data.size
    elif reduction == "sum":
        result_data = np.sum(loss)
        n = 1
    else:
        result_data = loss
        n = 1

    def backward(grad):
        """Run backward."""

        grad_pred = -target_data / pred_clamped + (1 - target_data) / (1 - pred_clamped)
        if reduction == "mean":
            grad_pred = grad * grad_pred / n
        else:
            grad_pred = grad * grad_pred
        return (grad_pred,)

    grad_fn = _make_backward("BCELoss", [pred], backward) if pred.requires_grad else None
    return Variable(result_data, requires_grad=pred.requires_grad, grad_fn=grad_fn)


def bce_with_logits_loss(logits, target, reduction="mean") -> Variable:
    """Binary cross-entropy with logits (numerically stable)."""
    logits = _ensure_variable(logits)
    target = _ensure_variable(target)

    # max(0, x) - x*y + log(1 + exp(-|x|))
    x = logits.data
    y = target.data
    loss = np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))

    if reduction == "mean":
        result_data = np.mean(loss)
        n = logits.data.size
    elif reduction == "sum":
        result_data = np.sum(loss)
        n = 1
    else:
        result_data = loss
        n = 1

    def backward(grad):
        # d/dx = sigmoid(x) - y
        """Run backward."""

        sigmoid_x = 1.0 / (1.0 + np.exp(-x))
        grad_logits = sigmoid_x - y
        if reduction == "mean":
            grad_logits = grad * grad_logits / n
        else:
            grad_logits = grad * grad_logits
        return (grad_logits,)

    grad_fn = (
        _make_backward("BCEWithLogitsLoss", [logits], backward) if logits.requires_grad else None
    )
    return Variable(result_data, requires_grad=logits.requires_grad, grad_fn=grad_fn)


def nll_loss(log_probs, target, reduction="mean") -> Variable:
    """Negative log-likelihood loss."""
    log_probs = _ensure_variable(log_probs)

    if isinstance(target, Variable):
        target = target.data

    target_int = target.astype(int)
    batch_size = log_probs.data.shape[0]
    loss = -log_probs.data[np.arange(batch_size), target_int]

    if reduction == "mean":
        result_data = np.mean(loss)
    elif reduction == "sum":
        result_data = np.sum(loss)
    else:
        result_data = loss

    log_probs_data = log_probs.data

    def backward(grad):
        """Run backward."""

        grad_input = np.zeros_like(log_probs_data)
        if reduction == "mean":
            grad_input[np.arange(batch_size), target_int] = -grad / batch_size
        elif reduction == "sum":
            grad_input[np.arange(batch_size), target_int] = -grad
        else:
            grad_input[np.arange(batch_size), target_int] = -grad
        return (grad_input,)

    grad_fn = _make_backward("NLLLoss", [log_probs], backward) if log_probs.requires_grad else None
    return Variable(result_data, requires_grad=log_probs.requires_grad, grad_fn=grad_fn)


def kl_div_loss(log_probs, target, reduction="mean") -> Variable:
    """KL divergence loss: target * (log(target) - log_probs)."""
    log_probs = _ensure_variable(log_probs)
    target = _ensure_variable(target)

    # KL(target || pred) = sum(target * (log(target) - log_probs))
    # Since log(target) is constant w.r.t. log_probs, we only need: -target * log_probs
    loss = -target.data * log_probs.data

    if reduction == "mean":
        result_data = np.mean(loss)
        n = log_probs.data.size
    elif reduction == "sum":
        result_data = np.sum(loss)
        n = 1
    elif reduction == "batchmean":
        result_data = np.sum(loss) / log_probs.data.shape[0]
        n = log_probs.data.shape[0]
    else:
        result_data = loss
        n = 1

    target_data = target.data

    def backward(grad):
        """Run backward."""

        if reduction == "mean":
            grad_input = -grad * target_data / n
        elif reduction == "batchmean":
            grad_input = -grad * target_data / n
        else:
            grad_input = -grad * target_data
        return (grad_input,)

    grad_fn = (
        _make_backward("KLDivLoss", [log_probs], backward) if log_probs.requires_grad else None
    )
    return Variable(result_data, requires_grad=log_probs.requires_grad, grad_fn=grad_fn)
