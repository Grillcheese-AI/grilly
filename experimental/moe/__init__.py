"""
grilly.experimental.moe - Resonator-based Mixture of Experts.

Provides relational encoding and expert routing using VSA operations.

Submodules:
    - relational: RelationalEncoder for encoding entities and relations
    - routing: ResonatorMoE and RelationalMoE for expert selection
"""

from .relational import RelationalEncoder
from .routing import RelationalMoE, ResonatorMoE

__all__ = [
    "RelationalEncoder",
    "ResonatorMoE",
    "RelationalMoE",
]
