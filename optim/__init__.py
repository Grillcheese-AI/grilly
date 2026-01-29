"""
Optimizers Module (PyTorch-like)

GPU-accelerated optimizers using Vulkan compute shaders.
"""
from .base import Optimizer
from .adam import Adam, AffectAdam
from .adamw import AdamW
from .sgd import SGD
from .nlms import NLMS
from .natural_gradient import NaturalGradient
from .lr_scheduler import (
    LRScheduler,
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR
)

__all__ = [
    'Optimizer',
    'Adam',
    'AdamW',
    'AffectAdam',
    'SGD',
    'NLMS',
    'NaturalGradient',
    'LRScheduler',
    'StepLR',
    'CosineAnnealingLR',
    'ReduceLROnPlateau',
    'OneCycleLR',
]
