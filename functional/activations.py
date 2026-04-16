"""Functional activation helpers backed by Grilly compute kernels."""

import numpy as np

from ._helpers import _to_numpy


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
    x = np.asarray(x, dtype=np.float32)
    return np.maximum(0.0, x).astype(np.float32)


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
    x = np.asarray(x, dtype=np.float32)
    return (
        0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    ).astype(np.float32)


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
    x = np.asarray(x, dtype=np.float32)
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


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
    x = np.asarray(x, dtype=np.float32)
    x_max = np.max(x, axis=dim, keepdims=True)
    exp_x = np.exp(x - x_max)
    return (exp_x / np.sum(exp_x, axis=dim, keepdims=True)).astype(np.float32)


def softplus(x: np.ndarray, beta: float = 1.0, threshold: float = 20.0) -> np.ndarray:
    """
    Softplus activation: (1/beta) * log(1 + exp(beta * x))

    Uses a linear approximation for large values to prevent overflow.
    Uses: activation-softplus.glsl
    """
    x = np.asarray(x, dtype=np.float32)
    bx = beta * x
    # For bx > threshold, softplus ≈ x (avoids exp overflow)
    return np.where(
        bx > threshold, x, (1.0 / beta) * np.log(1.0 + np.exp(np.minimum(bx, threshold)))
    ).astype(np.float32)
