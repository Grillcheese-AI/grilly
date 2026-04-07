"""Torch-style dtypes and ``device`` (Vulkan-only GPU path; no CUDA)."""

from __future__ import annotations

import numpy as np

# Aliases for ``dtype=torch.float32`` / ``torch.long`` style imports
float16 = np.dtype("float16")
float32 = np.dtype("float32")
float64 = np.dtype("float64")
int32 = np.dtype("int32")
int64 = np.dtype("int64")
long = int64


def _is_long_dtype(dtype: object) -> bool:
    if dtype is None:
        return False
    if dtype in (int64, long, np.int64, "int64", "long"):
        return True
    if isinstance(dtype, np.dtype) and dtype.kind in "iu":
        return dtype.itemsize >= 4
    return isinstance(dtype, str) and dtype.lower() in ("int64", "long")


def _dtype_to_numpy(dtype: object) -> np.dtype:
    if dtype is None:
        return np.dtype("float32")
    if isinstance(dtype, np.dtype):
        return dtype
    if dtype in (float16, float32, float64, int32, int64):
        return np.dtype(dtype)
    if dtype is np.float16:
        return np.dtype("float16")
    if dtype is np.float32:
        return np.dtype("float32")
    if dtype is np.float64:
        return np.dtype("float64")
    if dtype is np.int64:
        return np.dtype("int64")
    s = str(dtype).lower()
    if s in ("float16", "half"):
        return np.dtype("float16")
    if s in ("float32",):
        return np.dtype("float32")
    if s in ("float64", "double"):
        return np.dtype("float64")
    if s in ("int64", "long"):
        return np.dtype("int64")
    if s in ("int32",):
        return np.dtype("int32")
    return np.dtype("float32")


class device:
    """``torch.device``-like device tag (``cpu`` / ``vulkan``)."""

    __slots__ = ("type", "index")

    def __init__(self, type_: str, index: int | None = None) -> None:
        s = str(type_)
        if ":" in s:
            parts = s.split(":", 1)
            self.type = parts[0].strip()
            try:
                self.index = int(parts[1]) if parts[1] else None
            except ValueError:
                self.index = None
        else:
            self.type = s.strip()
            self.index = index

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, device):
            return NotImplemented
        return self.type == other.type and self.index == other.index

    def __str__(self) -> str:
        if self.index is not None:
            return f"{self.type}:{self.index}"
        return self.type

    def __repr__(self) -> str:
        return f"device(type='{self.type}'{'' if self.index is None else f', index={self.index}'})"


cpu = device("cpu")
