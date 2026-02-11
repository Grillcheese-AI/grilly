"""
Normalization functions (functional API)
Uses: fnn-layernorm.glsl, snn-rmsnorm.glsl
"""

import numpy as np


def _get_backend():
    """Get compute backend"""
    from grilly import Compute

    return Compute()


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
    backend = _get_backend()

    if weight is None:
        weight = np.ones(normalized_shape, dtype=np.float32)
    if bias is None:
        bias = np.zeros(normalized_shape, dtype=np.float32)

    return backend.layernorm(input, weight, bias, eps=eps)
