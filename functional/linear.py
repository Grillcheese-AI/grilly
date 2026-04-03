"""
Linear operations (functional API)
Uses: fnn-linear.glsl, fnn-linear-backward.glsl
"""

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
    try:
        from grilly.backend import _bridge

        result = _bridge.linear(input, weight, bias)
        if result is not None:
            return _to_numpy(result)
    except (ImportError, Exception):
        pass
    input_arr = np.asarray(input, dtype=np.float32)
    weight_arr = np.asarray(weight, dtype=np.float32)
    output = input_arr @ weight_arr.T
    if bias is not None:
        output = output + np.asarray(bias, dtype=np.float32)
    return np.asarray(output, dtype=np.float32)
