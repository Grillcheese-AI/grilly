"""``torch.nn.functional``-compatible API (cross-entropy, activations)."""

from grilly.functional.mf_activations import (
    mf_relu,
    mf_sigmoid,
    mf_sigmoid_01,
    mf_softmax,
    mf_softplus,
)
from grilly.nn.autograd import gelu
from grilly.torch_api.functional import cross_entropy, softplus

__all__ = [
    "cross_entropy",
    "softplus",
    "gelu",
    "mf_softmax",
    "mf_softplus",
    "mf_sigmoid",
    "mf_sigmoid_01",
    "mf_relu",
]
