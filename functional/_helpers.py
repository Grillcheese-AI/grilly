"""Shared helpers for functional modules — bridge result conversion."""

import numpy as np


def _to_numpy(result):
    """Convert bridge result to numpy if it's a C++ Tensor.

    Handles:
    - None → None
    - numpy array → pass through
    - grilly_core.Tensor → .numpy()
    - anything else → np.asarray()
    """
    if result is None:
        return None
    if isinstance(result, np.ndarray):
        return result
    if hasattr(result, "numpy"):
        return result.numpy()
    return np.asarray(result)
