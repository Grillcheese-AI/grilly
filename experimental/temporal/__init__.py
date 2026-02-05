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

from .encoder import TemporalEncoder
from .causal import CausalChain, CausalRule
from .state import TemporalState, TemporalWorldModel
from .counterfactual import CounterfactualReasoner, CounterfactualQuery, CounterfactualResult
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
