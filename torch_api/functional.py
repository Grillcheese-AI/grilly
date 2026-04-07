"""``torch.nn.functional``-compatible helpers."""

from __future__ import annotations

import numpy as np

from grilly.nn import autograd as ag


def softplus(input, beta: float = 1.0, threshold: float = 20.0):
    return ag.softplus(input, beta, threshold)


def cross_entropy(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index: int = -100,
    reduce=None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
):
    del weight, size_average, reduce, label_smoothing
    logits = ag._ensure_variable(input)
    if isinstance(target, ag.Variable):
        t = target.data
    elif isinstance(target, np.ndarray):
        t = target
    elif hasattr(target, "data") and not isinstance(target, np.ndarray):
        t = target.data
    else:
        t = np.asarray(target)

    if ignore_index >= 0:
        pass

    ce = ag.cross_entropy(logits, t)

    if reduction == "mean":
        return ce
    if reduction == "sum":
        n = float(logits.data.shape[0])
        return ag.mul(ce, n)
    if reduction == "none":
        raise NotImplementedError("cross_entropy reduction='none' is not implemented")
    raise ValueError(f"Unknown reduction: {reduction}")
