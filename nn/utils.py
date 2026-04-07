"""``torch.nn.utils`` — gradient clipping and helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np


def clip_grad_norm_(
    parameters: Iterable[Any],
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """
    Clip gradient norm of an iterable of parameters (in-place).

    Mirrors PyTorch: per-tensor norm, then global norm of those norms.
    """
    params = [p for p in parameters if p is not None]
    grads: list[np.ndarray] = []
    for p in params:
        g = getattr(p, "grad", None)
        if g is not None:
            grads.append(np.asarray(g, dtype=np.float64))

    if not grads:
        return 0.0

    if norm_type in (math.inf, float("inf")):
        total_norm = max(float(np.max(np.abs(g))) for g in grads)
    else:
        norms = []
        for g in grads:
            if norm_type in (2.0, 2):
                norms.append(math.sqrt(float(np.sum(g * g))))
            else:
                norms.append(float(np.sum(np.abs(g) ** norm_type) ** (1.0 / norm_type)))
        stacked = np.array(norms, dtype=np.float64)
        if norm_type in (2.0, 2):
            total_norm = float(np.sqrt(np.sum(stacked * stacked)))
        else:
            total_norm = float(np.sum(np.abs(stacked) ** norm_type) ** (1.0 / norm_type))

    clip_coef = max_norm / (total_norm + 1e-6) if max_norm > 0 else 1.0
    if clip_coef < 1.0:
        for p in params:
            g = getattr(p, "grad", None)
            if g is None:
                continue
            if isinstance(g, np.ndarray):
                g *= clip_coef
            else:
                try:
                    p.grad = np.asarray(g, dtype=np.float32) * clip_coef
                except Exception:
                    pass

    return float(total_norm)
