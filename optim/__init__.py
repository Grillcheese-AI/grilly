"""
Optimizers Module (PyTorch-like)

GPU-accelerated optimizers using Vulkan compute shaders.
"""

from .adam import Adam, AffectAdam
from .adamw import AdamW
from .base import Optimizer
from .hypergradient import AutoHypergradientAdamW, HypergradientAdamW
from .lr_scheduler import CosineAnnealingLR, LRScheduler, OneCycleLR, ReduceLROnPlateau, StepLR
from .natural_gradient import NaturalGradient
from .nlms import NLMS
from .sgd import SGD

__all__ = [
    "Optimizer",
    "Adam",
    "AdamW",
    "AffectAdam",
    "SGD",
    "NLMS",
    "NaturalGradient",
    "HypergradientAdamW",
    "AutoHypergradientAdamW",
    "LRScheduler",
    "StepLR",
    "CosineAnnealingLR",
    "ReduceLROnPlateau",
    "OneCycleLR",
]
