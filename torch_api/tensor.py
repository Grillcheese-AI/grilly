"""``torch.Tensor``-like float tensor (``Variable`` subclass) and ``LongTensor``."""

from __future__ import annotations

import numpy as np

from grilly.nn.autograd import Variable

from .dtypes_and_device import _dtype_to_numpy, device


class LongTensor(np.ndarray):
    """Integer index tensor (``int64``), ndarray subclass."""

    def __getitem__(self, key):
        r = super().__getitem__(key)
        if not isinstance(r, np.ndarray):
            return r
        if r.ndim == 0:
            return r
        if r.dtype == np.int64:
            return r.view(LongTensor)
        return r

    def __new__(cls, input_array, device: device | str | None = None):
        obj = np.asarray(input_array, dtype=np.int64).view(cls)
        obj._torch_device = device  # type: ignore[attr-defined]
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._torch_device = getattr(obj, "_torch_device", None)  # type: ignore[attr-defined]

    def to(self, device: device | str | None = None, dtype=None, non_blocking: bool = False):
        del non_blocking
        if dtype is not None:
            np_dt = _dtype_to_numpy(dtype)
            return Tensor(np.asarray(self, dtype=np.float32).astype(np_dt))
        return LongTensor(self.copy(), device=device)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.int64)

    def numel(self) -> int:
        return int(self.size)


class Tensor(Variable):
    """Float tensor with PyTorch-style helpers (numpy + autograd ``Variable``)."""

    def uniform_(self, a: float = 0.0, b: float = 1.0) -> Tensor:
        self.data[:] = np.random.uniform(a, b, self.shape).astype(np.float32)
        return self

    def zero_(self) -> Tensor:
        self.data.fill(np.float32(0.0))
        return self

    def to(
        self,
        device: device | str | None = None,
        dtype: object | None = None,
        non_blocking: bool = False,
    ) -> Tensor | LongTensor:
        del non_blocking  # numpy path: unused
        if dtype is not None:
            np_dt = _dtype_to_numpy(dtype)
            if np_dt.kind in "iu":
                return LongTensor(self.data.astype(np_dt))
            new_data = self.data.astype(np_dt)
            return Tensor(new_data, requires_grad=self.requires_grad)
        if device is not None:
            pass  # no separate device memory in numpy path
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype


def as_numpy(x: object) -> np.ndarray:
    """Unwrap ``Tensor`` / ``Variable`` / ``LongTensor`` to ndarray."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "data") and isinstance(getattr(x, "data", None), np.ndarray):
        return x.data  # type: ignore[union-attr]
    return np.asarray(x)
