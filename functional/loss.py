"""
Loss functions (functional API)
Uses: loss-cross-entropy.glsl, loss-fn-bce.glsl
"""

import numpy as np

from ._helpers import _to_numpy


def cross_entropy(
    input: np.ndarray, target: np.ndarray, weight: np.ndarray | None = None, reduction: str = "mean"
) -> np.ndarray:
    """
    Cross-entropy loss
    Uses: loss-cross-entropy.glsl

    Args:
        input: Logits tensor of shape (N, C) or (N, C, ...)
        target: Target class indices of shape (N,) or (N, ...)
        weight: Optional class weights
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss value(s)
    """
    try:
        from grilly.backend import _bridge

        if target.ndim < input.ndim and target.dtype.kind in {"i", "u"}:
            gpu_loss = _bridge.cross_entropy_loss(input, target)
            if gpu_loss is not None:
                loss = _to_numpy(gpu_loss)
                if reduction == "mean":
                    return np.mean(loss)
                if reduction == "sum":
                    return np.sum(loss)
                return loss
    except (ImportError, Exception):
        pass
    # CPU fallback
    input_softmax = np.exp(input - np.max(input, axis=-1, keepdims=True))
    input_softmax = input_softmax / np.sum(input_softmax, axis=-1, keepdims=True)

    # One-hot encode targets
    if target.ndim < input.ndim:
        target_onehot = np.zeros_like(input)
        target_onehot[np.arange(len(target)), target] = 1
    else:
        target_onehot = target

    loss = -np.sum(target_onehot * np.log(input_softmax + 1e-8), axis=-1)

    if weight is not None:
        loss = loss * weight[target]

    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        return loss


def binary_cross_entropy(
    input: np.ndarray, target: np.ndarray, weight: np.ndarray | None = None, reduction: str = "mean"
) -> np.ndarray:
    """
    Binary cross-entropy loss
    Uses: loss-fn-bce.glsl

    Args:
        input: Predicted probabilities of shape (N, ...)
        target: Target probabilities of shape (N, ...)
        weight: Optional sample weights
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss value(s)
    """
    # CPU fallback
    input_clamped = np.clip(input, 1e-8, 1 - 1e-8)
    loss = -(target * np.log(input_clamped) + (1 - target) * np.log(1 - input_clamped))

    if weight is not None:
        loss = loss * weight

    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        return loss
