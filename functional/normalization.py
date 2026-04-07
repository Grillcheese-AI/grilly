"""
Normalization functions (functional API)
Uses: fnn-layernorm.glsl, snn-rmsnorm.glsl
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


def layer_norm(
    input: np.ndarray,
    normalized_shape: int,
    weight: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Layer normalization
    Uses: fnn-layernorm.glsl

    Args:
        input: Input tensor
        normalized_shape: Size of normalized dimension
        weight: Optional scale parameter
        bias: Optional shift parameter
        eps: Small value for numerical stability

    Returns:
        Normalized tensor
    """
    if weight is None:
        weight = np.ones(normalized_shape, dtype=np.float32)
    if bias is None:
        bias = np.zeros(normalized_shape, dtype=np.float32)

    try:
        from grilly.backend import _bridge

        result = _bridge.layernorm(input, weight, bias, eps)
        if result is not None:
            return _to_numpy(result)
    except (ImportError, Exception):
        pass
    input_arr = np.asarray(input, dtype=np.float32)
    weight_arr = np.asarray(weight, dtype=np.float32)
    bias_arr = np.asarray(bias, dtype=np.float32)
    mean = np.mean(input_arr, axis=-1, keepdims=True)
    var = np.var(input_arr, axis=-1, keepdims=True)
    normalized = (input_arr - mean) / np.sqrt(var + eps)
    return (normalized * weight_arr + bias_arr).astype(np.float32)
