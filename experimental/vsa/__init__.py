"""
grilly.experimental.vsa - Vector Symbolic Architecture operations.

Core VSA operations that form the foundation for all experimental features.
Provides both binary (bipolar) and holographic (continuous) vector operations.

Classes:
    BinaryOps: Operations for bipolar (+1/-1) vectors
    HolographicOps: Operations for continuous vectors using FFT-based binding
    ResonatorNetwork: Factorization of composite vectors
"""

from .ops import BinaryOps, HolographicOps
from .resonator import ResonatorNetwork

__all__ = [
    "BinaryOps",
    "HolographicOps",
    "ResonatorNetwork",
]
