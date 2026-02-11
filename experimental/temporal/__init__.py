"""
grilly.experimental.temporal - Temporal reasoning with VSA.

Provides temporal state tracking, causal chains, counterfactual reasoning,
and decision validation across time.

Submodules:
    - encoder: TemporalEncoder for time encoding
    - causal: CausalChain for causal rules and propagation
    - state: TemporalState and TemporalWorldModel
    - counterfactual: CounterfactualReasoner
    - validator: TemporalDecisionValidator
"""

from .causal import CausalChain, CausalRule
from .counterfactual import CounterfactualQuery, CounterfactualReasoner, CounterfactualResult
from .encoder import TemporalEncoder
from .state import TemporalState, TemporalWorldModel
from .validator import TemporalDecisionValidator, ValidationResult

__all__ = [
    "TemporalEncoder",
    "CausalChain",
    "CausalRule",
    "TemporalState",
    "TemporalWorldModel",
    "CounterfactualReasoner",
    "CounterfactualQuery",
    "CounterfactualResult",
    "TemporalDecisionValidator",
    "ValidationResult",
]
