"""``torch.nn.init``-style weight initialization (numpy arrays / Parameters)."""

from __future__ import annotations

from grilly.utils.initialization import (
    kaiming_normal_,
    kaiming_uniform_,
    normal_,
    uniform_,
    xavier_normal_,
    xavier_uniform_,
)

__all__ = [
    "kaiming_normal_",
    "kaiming_uniform_",
    "normal_",
    "uniform_",
    "xavier_normal_",
    "xavier_uniform_",
]
