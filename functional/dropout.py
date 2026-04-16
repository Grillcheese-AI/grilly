"""
Dropout functions (functional API)
Uses: fnn-dropout.glsl
"""

import numpy as np

from ._helpers import _to_numpy


def dropout(input: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    """
    Dropout regularization
    Uses: fnn-dropout.glsl

    Args:
        input: Input tensor
        p: Dropout probability (0.0 to 1.0)
        training: If False, returns input unchanged

    Returns:
        Output tensor with dropout applied (if training)
    """
    input_arr = np.asarray(input, dtype=np.float32)
    if not training or p == 0:
        return input_arr
    if p >= 1.0:
        return np.zeros_like(input_arr, dtype=np.float32)

    random_mask = (np.random.rand(*input_arr.shape) >= p).astype(np.float32)
    try:
        from grilly.backend import _bridge

        result = _bridge.dropout(input_arr, random_mask, p=p, training=training)
        if result is not None:
            return _to_numpy(result)
    except (ImportError, Exception):
        pass

    scale = 1.0 / (1.0 - p)
    return (input_arr * random_mask * scale).astype(np.float32)
