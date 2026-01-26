"""
Dropout functions (functional API)
Uses: fnn-dropout.glsl
"""
import numpy as np
from typing import Optional


def _get_backend():
    """Get compute backend"""
    from grilly import Compute
    return Compute()


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
    if not training or p == 0:
        return input
    
    backend = _get_backend()
    return backend.dropout(input, p=p)
