"""Functional activation helpers backed by Grilly compute kernels."""

import numpy as np


def _get_backend():
    """Get compute backend"""
    from grilly import Compute

    return Compute()


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation: max(0, x)
    Uses: activation-relu.glsl
    """
    backend = _get_backend()
    return backend.activation_relu(x)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU activation
    Uses: activation-gelu.glsl
    """
    backend = _get_backend()
    return backend.activation_gelu(x)


def silu(x: np.ndarray) -> np.ndarray:
    """
    SiLU (Swish) activation: x * sigmoid(x)
    Uses: activation-silu.glsl
    """
    backend = _get_backend()
    return backend.activation_silu(x)


def softmax(x: np.ndarray, dim: int = -1) -> np.ndarray:
    """
    Softmax activation
    Uses: activation-softmax.glsl
    """
    backend = _get_backend()
    return backend.activation_softmax(x, dim=dim)


def softplus(x: np.ndarray) -> np.ndarray:
    """
    Softplus activation: log(1 + exp(x))
    Uses: activation-softplus.glsl
    """
    _get_backend()
    # Note: May need to implement in backend if not already exposed
    # CPU fallback for now
    return np.log(1 + np.exp(x))
