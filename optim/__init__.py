"""
Optimizers Module (PyTorch-like)

GPU-accelerated optimizers using Vulkan compute shaders.
"""
from .base import Optimizer
from .adam import Adam, AffectAdam
from .sgd import SGD
from .nlms import NLMS
from .natural_gradient import NaturalGradient

__all__ = [
    'Optimizer',
    'Adam',
    'AffectAdam',
    'SGD',
    'NLMS',
    'NaturalGradient',
]
