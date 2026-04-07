"""
Multiplication-free (or low-mul) activations for VSA / addition-style pipelines.

- **mf_softmax**: ReLU-normalized probabilities — no ``exp``; only ``max``, subtract,
  ``relu``, sum, divide (same spirit as ``addition-linear.glsl``: add/sub/abs in the core).
- **mf_softplus**: Algebraic softplus — ``0.5 * (x + sqrt(x^2 + c))`` with ``c = 4/beta²``.
  Smooth, no ``log``/``exp``; uses ``sqrt`` (and scalar scaling), not GEMM-style multiply.
- **mf_sigmoid**: Rational sigmoid — ``x / (1 + |x|)``, bounded in ``(-1, 1)``; scale to
  ``(0, 1)`` with ``0.5 * (1 + ...)`` if needed via :func:`mf_sigmoid_01`.
"""

from __future__ import annotations

import numpy as np


def mf_softmax(x: np.ndarray, dim: int = -1, eps: float = 1e-12) -> np.ndarray:
    """ReLU-normalized softmax: ``relu(x - max) / sum(relu(x - max))``.

    No exponential; suitable as a sparse, addition-heavy alternative to softmax.
    """
    x = np.asarray(x, dtype=np.float32)
    axis = dim if dim >= 0 else x.ndim + dim
    m = np.max(x, axis=axis, keepdims=True)
    z = np.maximum(x - m, 0.0).astype(np.float32)
    s = np.sum(z, axis=axis, keepdims=True, dtype=np.float64)
    denom = np.maximum(s, eps)
    y = (z / denom).astype(np.float32)
    # Degenerate: all logits tied → z is all zeros → use uniform distribution
    tot = np.sum(y, axis=axis, keepdims=True)
    nfeat = float(x.shape[axis])
    unif = np.ones_like(x, dtype=np.float32) / nfeat
    return np.where(tot > 1e-8, y, unif).astype(np.float32)


def mf_softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Algebraic (exp-free) softplus: ``(x + sqrt(x^2 + 4/beta^2)) / 2``.

    Matches ``softplus`` shape qualitatively; uses ``sqrt`` instead of ``log``/``exp``.
    """
    x = np.asarray(x, dtype=np.float32)
    b = float(beta)
    if b <= 0:
        raise ValueError("beta must be positive")
    c = 4.0 / (b * b)
    s = np.sqrt(x * x + c)
    return (0.5 * (x + s)).astype(np.float32)


def mf_sigmoid(x: np.ndarray) -> np.ndarray:
    """Rational sigmoid: ``x / (1 + |x|)``, range ``(-1, 1)``."""
    x = np.asarray(x, dtype=np.float32)
    ax = np.abs(x) + 1.0
    return (x / ax).astype(np.float32)


def mf_sigmoid_01(x: np.ndarray) -> np.ndarray:
    """Map :func:`mf_sigmoid` to ``(0, 1)``: ``0.5 * (1 + x/(1+|x|))``."""
    return (0.5 * (1.0 + mf_sigmoid(x))).astype(np.float32)


def mf_relu(x: np.ndarray) -> np.ndarray:
    """``max(0, x)`` — multiplication-free nonlinearity."""
    x = np.asarray(x, dtype=np.float32)
    return np.maximum(x, 0.0).astype(np.float32)
