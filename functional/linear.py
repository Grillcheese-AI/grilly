"""
Linear operations (functional API)
Uses: fnn-linear.glsl, fnn-linear-backward.glsl
"""

import numpy as np


def _get_backend():
    """Get compute backend"""
    from grilly import Compute

    return Compute()


def linear(input: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """
    Linear transformation: output = input @ weight.T + bias
    Uses: fnn-linear.glsl

    Args:
        input: Input tensor of shape (..., in_features)
        weight: Weight matrix of shape (out_features, in_features)
        bias: Optional bias vector of shape (out_features,)

    Returns:
        Output tensor of shape (..., out_features)
    """
    backend = _get_backend()
    return backend.fnn.linear(input, weight, bias)
