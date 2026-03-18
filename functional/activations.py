"""Functional activation helpers backed by Grilly compute kernels."""

import numpy as np


def _to_numpy(result):
    """Convert bridge result to numpy if it's a C++ Tensor."""
    if result is None:
        return None
    if isinstance(result, np.ndarray):
        return result
    if hasattr(result, "numpy"):
        return result.numpy()
    return np.asarray(result)


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation: max(0, x)
    Uses: activation-relu.glsl
    """
    try:
        from grilly.backend import _bridge

        result = _bridge.relu(x)
        if result is not None:
            return _to_numpy(result)
    except (ImportError, Exception):
        pass
    # Fallback to legacy Compute() path
    from grilly import Compute

    return Compute().activation_relu(x)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU activation
    Uses: activation-gelu.glsl
    """
    try:
        from grilly.backend import _bridge

        result = _bridge.gelu(x)
        if result is not None:
            return _to_numpy(result)
    except (ImportError, Exception):
        pass
    # Fallback to legacy Compute() path
    from grilly import Compute

    return Compute().activation_gelu(x)


def silu(x: np.ndarray) -> np.ndarray:
    """
    SiLU (Swish) activation: x * sigmoid(x)
    Uses: activation-silu.glsl
    """
    try:
        from grilly.backend import _bridge

        result = _bridge.silu(x)
        if result is not None:
            return _to_numpy(result)
    except (ImportError, Exception):
        pass
    # Fallback to legacy Compute() path
    from grilly import Compute

    return Compute().activation_silu(x)


def softmax(x: np.ndarray, dim: int = -1) -> np.ndarray:
    """
    Softmax activation
    Uses: activation-softmax.glsl
    """
    try:
        from grilly.backend import _bridge

        result = _bridge.softmax(x, dim)
        if result is not None:
            return _to_numpy(result)
    except (ImportError, Exception):
        pass
    # Fallback to legacy Compute() path
    from grilly import Compute

    return Compute().activation_softmax(x, dim=dim)


def softplus(x: np.ndarray) -> np.ndarray:
    """
    Softplus activation: log(1 + exp(x))
    Uses: activation-softplus.glsl
    """
    # No bridge equivalent; CPU fallback
    return np.log(1 + np.exp(x))
