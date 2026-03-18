"""
Shared helpers for nn modules — parameter wrappers, bridge utilities.
Not part of the public API; import from nn.modules or specific submodules.
"""

import numpy as np

# Try to import Parameter class
try:
    from .parameter import Parameter as ParameterClass

    _PARAMETER_AVAILABLE = True
except ImportError:
    _PARAMETER_AVAILABLE = False
    ParameterClass = None

# C++ bridge fast path (persistent-mapped VMA, zero vkMap/vkUnmap)
try:
    from ..backend import _bridge

    _USE_CPP_BRIDGE = _bridge.is_available()
except Exception:
    _bridge = None
    _USE_CPP_BRIDGE = False


def _bridge_to_numpy(result):
    """Ensure bridge result is numpy array (C++ core may return grilly_core.Tensor)."""
    if result is None:
        return None
    if isinstance(result, np.ndarray):
        return result
    # grilly_core.Tensor — convert to numpy via buffer protocol or .numpy()
    if hasattr(result, "numpy"):
        return result.numpy()
    return np.asarray(result, dtype=np.float32)


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
