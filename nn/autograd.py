"""
Automatic Differentiation System (Autograd)

This module provides automatic differentiation capabilities similar to PyTorch's autograd.
It tracks operations during the forward pass and automatically computes gradients during backward.
"""

import numpy as np
from typing import Optional, List, Callable, Tuple, Any
from collections import deque


class Variable:
    """
    A variable that tracks computation history for automatic differentiation.
    
    Similar to PyTorch's Tensor with requires_grad=True.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        requires_grad: bool = False,
        grad_fn: Optional[Callable] = None,
        is_leaf: bool = True
    ):
        """
        Args:
            data: The actual data (numpy array)
            requires_grad: Whether to track gradients for this variable
            grad_fn: Function to call during backward pass
            is_leaf: Whether this is a leaf node (no parent operations)
        """
        self.data = np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data.astype(np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = grad_fn
        self.is_leaf = is_leaf
        self._saved_tensors = []  # Store intermediate values needed for backward
    
    def backward(self, grad_output: Optional[np.ndarray] = None):
        """
        Backward pass: compute gradients.
        
        Args:
            grad_output: Gradient w.r.t. this variable (default: 1.0 for loss)
        """
        if not self.requires_grad:
            return
        
        if grad_output is None:
            # If this is the final output (loss), start with gradient of 1.0
            grad_output = np.ones_like(self.data, dtype=np.float32)
        
        # Accumulate gradient
        if self.grad is None:
            self.grad = grad_output.copy()
        else:
            self.grad += grad_output
        
        # Call backward function if this is not a leaf
        if self.grad_fn is not None:
            self.grad_fn(grad_output)
    
    def zero_grad(self):
        """Clear gradients"""
        self.grad = None
    
    def detach(self):
        """Return a new Variable without gradient tracking"""
        return Variable(self.data, requires_grad=False, is_leaf=True)
    
    def __repr__(self):
        return f"Variable(shape={self.data.shape}, requires_grad={self.requires_grad}, grad={self.grad is not None})"


class Function:
    """
    Base class for operations that can be tracked for autograd.
    
    Similar to PyTorch's Function class.
    """
    
    @staticmethod
    def forward(ctx: Any, *args, **kwargs) -> Any:
        """
        Forward pass of the operation.
        
        Args:
            ctx: Context object to store intermediate values
            *args: Input arguments
            **kwargs: Keyword arguments
        
        Returns:
            Output of the operation
        """
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx: Any, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray], ...]:
        """
        Backward pass of the operation.
        
        Args:
            ctx: Context object with saved intermediate values
            grad_output: Gradient w.r.t. output
        
        Returns:
            Tuple of gradients w.r.t. each input (None for non-differentiable inputs)
        """
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *args, **kwargs):
        """
        Apply the function with automatic gradient tracking.
        
        Args:
            *args: Input arguments (can be Variables or numpy arrays)
            **kwargs: Keyword arguments
        
        Returns:
            Variable with gradient tracking if any input requires_grad
        """
        # Check if any input requires gradient
        requires_grad = any(
            isinstance(arg, Variable) and arg.requires_grad
            for arg in args
        )
        
        # Convert numpy arrays to Variables if needed
        args_vars = []
        for arg in args:
            if isinstance(arg, Variable):
                args_vars.append(arg)
            else:
                args_vars.append(Variable(arg, requires_grad=False, is_leaf=True))
        
        # Create context for storing intermediate values
        ctx = type('Context', (), {})()
        
        # Extract data for forward pass
        args_data = [arg.data for arg in args_vars]
        
        # Forward pass
        output_data = cls.forward(ctx, *args_data, **kwargs)
        
        # Create output Variable
        output = Variable(
            output_data,
            requires_grad=requires_grad,
            grad_fn=None if not requires_grad else lambda grad_out: cls._backward(ctx, args_vars, grad_out),
            is_leaf=not requires_grad
        )
        
        # Store context and input variables for backward
        if requires_grad:
            output._ctx = ctx
            output._input_vars = args_vars
        
        return output
    
    @classmethod
    def _backward(cls, ctx: Any, input_vars: List[Variable], grad_output: np.ndarray):
        """
        Internal backward pass that distributes gradients to input variables.
        
        Args:
            ctx: Context object with saved intermediate values
            input_vars: List of input Variables
            grad_output: Gradient w.r.t. output
        """
        # Compute gradients w.r.t. inputs
        grad_inputs = cls.backward(ctx, grad_output)
        
        # Distribute gradients to input variables
        if grad_inputs is None:
            grad_inputs = (None,) * len(input_vars)
        
        # Ensure grad_inputs is a tuple
        if not isinstance(grad_inputs, tuple):
            grad_inputs = (grad_inputs,)
        
        # Accumulate gradients in input variables
        for var, grad_in in zip(input_vars, grad_inputs):
            if var.requires_grad and grad_in is not None:
                if var.grad is None:
                    var.grad = grad_in.copy()
                else:
                    var.grad += grad_in


# ============================================================================
# Common Operations with Autograd Support
# ============================================================================

class Add(Function):
    """Addition operation: output = input1 + input2"""
    
    @staticmethod
    def forward(ctx, input1, input2):
        return input1 + input2
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class Mul(Function):
    """Multiplication operation: output = input1 * input2"""
    
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.saved = (input1, input2)
        return input1 * input2
    
    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved
        return grad_output * input2, grad_output * input1


class MatMul(Function):
    """Matrix multiplication: output = input1 @ input2"""
    
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.saved = (input1, input2)
        return np.matmul(input1, input2)
    
    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved
        grad_input1 = np.matmul(grad_output, input2.T) if input1.ndim > 1 else np.sum(grad_output * input2, axis=-1)
        grad_input2 = np.matmul(input1.T, grad_output) if input2.ndim > 1 else np.sum(input1 * grad_output, axis=0)
        return grad_input1, grad_input2


class Sum(Function):
    """Sum operation: output = sum(input)"""
    
    @staticmethod
    def forward(ctx, input, dim=None, keepdims=False):
        ctx.saved = (input.shape, dim, keepdims)
        return np.sum(input, axis=dim, keepdims=keepdims)
    
    @staticmethod
    def backward(ctx, grad_output):
        input_shape, dim, keepdims = ctx.saved
        # Expand gradient to match input shape
        if not keepdims and dim is not None:
            grad_output = np.expand_dims(grad_output, axis=dim)
        # Broadcast to input shape
        grad_input = np.broadcast_to(grad_output, input_shape)
        return grad_input


class Mean(Function):
    """Mean operation: output = mean(input)"""
    
    @staticmethod
    def forward(ctx, input, dim=None, keepdims=False):
        ctx.saved = (input.shape, dim, keepdims, input.size if dim is None else input.shape[dim])
        return np.mean(input, axis=dim, keepdims=keepdims)
    
    @staticmethod
    def backward(ctx, grad_output):
        input_shape, dim, keepdims, N = ctx.saved
        # Expand gradient to match input shape
        if not keepdims and dim is not None:
            grad_output = np.expand_dims(grad_output, axis=dim)
        # Broadcast to input shape and scale by 1/N
        grad_input = np.broadcast_to(grad_output, input_shape) / N
        return grad_input


# ============================================================================
# Convenience Functions
# ============================================================================

def add(input1, input2):
    """Add two variables: input1 + input2"""
    return Add.apply(input1, input2)


def mul(input1, input2):
    """Multiply two variables: input1 * input2"""
    return Mul.apply(input1, input2)


def matmul(input1, input2):
    """Matrix multiply: input1 @ input2"""
    return MatMul.apply(input1, input2)


def sum(input, dim=None, keepdims=False):
    """Sum: sum(input, dim=dim, keepdims=keepdims)"""
    return Sum.apply(input, dim=dim, keepdims=keepdims)


def mean(input, dim=None, keepdims=False):
    """Mean: mean(input, dim=dim, keepdims=keepdims)"""
    return Mean.apply(input, dim=dim, keepdims=keepdims)


# ============================================================================
# Module Integration
# ============================================================================

def enable_grad(variable: Variable):
    """Enable gradient tracking for a variable"""
    variable.requires_grad = True
    return variable


def no_grad():
    """Context manager to disable gradient tracking"""
    class NoGradContext:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return NoGradContext()
