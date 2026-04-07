"""Tensor factories and math ops (``torch.*`` facade)."""

from __future__ import annotations

import numpy as np

from grilly.nn import autograd as ag

from .dtypes_and_device import _dtype_to_numpy, _is_long_dtype
from .dtypes_and_device import device as _TorchDevice
from .tensor import LongTensor, Tensor, as_numpy

_DEFAULT_FLOAT = np.dtype(np.float32)


def tensor(
    data,
    dtype: object | None = None,
    device: _TorchDevice | str | None = None,
    requires_grad: bool = False,
):
    if _is_long_dtype(dtype):
        return LongTensor(data, device=device)
    np_dt = _dtype_to_numpy(dtype) if dtype is not None else _DEFAULT_FLOAT
    arr = np.asarray(data, dtype=np_dt)
    if np_dt.kind in "iu":
        return LongTensor(arr, device=device)
    return Tensor(arr.astype(np.float32), requires_grad=requires_grad)


def empty(*size: int, dtype=None, device=None, requires_grad: bool = False) -> Tensor:
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])  # type: ignore[assignment]
    if not size:
        raise TypeError("empty() requires at least one size dimension")
    np_dt = _dtype_to_numpy(dtype) if dtype is not None else _DEFAULT_FLOAT
    if np_dt.kind in "iu":
        return LongTensor(np.empty(size, dtype=np_dt), device=device)
    return Tensor(np.empty(size, dtype=np.float32), requires_grad=requires_grad)


def zeros(*size: int, dtype=None, device=None, requires_grad: bool = False) -> Tensor:
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])  # type: ignore[assignment]
    if not size:
        raise TypeError("zeros() requires size")
    np_dt = _dtype_to_numpy(dtype) if dtype is not None else _DEFAULT_FLOAT
    if np_dt.kind in "iu":
        return LongTensor(np.zeros(size, dtype=np_dt), device=device)
    return Tensor(np.zeros(size, dtype=np.float32), requires_grad=requires_grad)


def ones(*size: int, dtype=None, device=None, requires_grad: bool = False) -> Tensor:
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])  # type: ignore[assignment]
    np_dt = _dtype_to_numpy(dtype) if dtype is not None else _DEFAULT_FLOAT
    if np_dt.kind in "iu":
        return LongTensor(np.ones(size, dtype=np_dt), device=device)
    return Tensor(np.ones(size, dtype=np.float32), requires_grad=requires_grad)


def randn(*size: int, requires_grad: bool = False) -> Tensor:
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])  # type: ignore[assignment]
    if not size:
        return Tensor(np.array(np.random.randn(), dtype=np.float32), requires_grad=requires_grad)
    return Tensor(np.random.randn(*size).astype(np.float32), requires_grad=requires_grad)


def randperm(n: int, dtype=None, device=None, requires_grad: bool = False) -> LongTensor:
    """Random permutation of ``0 .. n-1`` (CPU/numpy)."""
    idx = np.random.permutation(n).astype(np.int64)
    return LongTensor(idx, device=device)


def cdist(x1, x2, p: float = 2.0, compute_mode=None) -> Tensor:
    """Pairwise distance (``p=1`` L1 / Manhattan)."""
    del compute_mode
    a = as_numpy(x1)
    b = as_numpy(x2)
    if a.ndim == 2:
        a = a[np.newaxis, ...]
    if b.ndim == 2:
        b = b[np.newaxis, ...]
    if p == 1:
        # (B, P, D), (B, R, D) -> (B, P, R)
        diff = np.abs(a[:, :, None, :] - b[:, None, :, :])
        dist = np.sum(diff, axis=-1)
    elif p == 2:
        diff = a[:, :, None, :] - b[:, None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
    else:
        diff = np.abs(a[:, :, None, :] - b[:, None, :, :]) ** p
        dist = np.sum(diff, axis=-1) ** (1.0 / p)
    return Tensor(dist.astype(np.float32), requires_grad=False)


def tanh(input) -> Tensor:
    return ag.tanh(input)  # type: ignore[return-value]


def clamp(input, min_val=None, max_val=None) -> Tensor:
    return ag.clamp(input, min_val, max_val)  # type: ignore[return-value]


def sign(input) -> Tensor:
    return ag.sign(input)  # type: ignore[return-value]


# Re-export no_grad from autograd
no_grad = ag.no_grad
