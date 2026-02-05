"""
Example: Temporal Reasoning

Demonstrates temporal encoding, causal chains, counterfactual reasoning,
and decision validation across time.
"""

import numpy as np
from grilly.experimental.temporal import (
    TemporalEncoder, CausalChain, CounterfactualReasoner,
    TemporalDecisionValidator
)

print("=" * 60)
print("Temporal Reasoning Examples")
print("=" * 60)

dim = 2048

# Temporal Encoder
print("\n1. Temporal Encoding")
print("-" * 60)

encoder = TemporalEncoder(dim=dim)

time_vec_5 = encoder.encode_time(5)
time_vec_6 = encoder.encode_time(6)
print(f"Time vector t=5: shape={time_vec_5.shape}")
print(f"Time vector t=6: shape={time_vec_6.shape}")

from grilly.experimental.vsa import HolographicOps
similarity = HolographicOps.similarity(time_vec_5, time_vec_6)
print(f"Similarity between t=5 and t=6: {similarity:.4f}")

state = HolographicOps.random_vector(dim)
bound_state = encoder.bind_with_time(state, t=5)
recovered = encoder.unbind_time(bound_state, t=5)
recovery_sim = HolographicOps.similarity(state, recovered)
print(f"\nState binding recovery: {recovery_sim:.4f}")

# Causal Chain
print("\n2. Causal Chain")
print("-" * 60)

chain = CausalChain(dim=dim)

chain.add_rule(
    name="rain_wet",
    conditions={"raining": True},
    effects={"wet": True},
    probability=0.9
)

chain.add_rule(
    name="wet_sick",
    conditions={"wet": True, "cold": True},
    effects={"sick": True},
    probability=0.7
)

print(f"Added {len(chain.rules)} causal rules")

initial_state = chain.encode_state({"raining": True})
print(f"Initial state: raining=True")
print(f"State vector: shape={initial_state.shape}")

propagated = chain.propagate_forward(initial_state, steps=2)
print(f"Propagated state after 2 steps: shape={propagated.shape}")

# Counterfactual Reasoning
print("\n3. Counterfactual Reasoning")
print("-" * 60)

reasoner = CounterfactualReasoner(chain, encoder)

query = reasoner.intervene(
    variable="raining",
    counterfactual_value=False,
    at_time=1,
    initial_state={"raining": True}
)

print(f"Counterfactual: What if it wasn't raining at t=1?")
print(f"Query created: {query is not None}")

result = reasoner.query_counterfactual(
    query=query,
    query_variable="sick",
    query_time=3
)

print(f"\nActual outcome (raining=True): {result.actual_outcome}")
print(f"Counterfactual outcome (raining=False): {result.counterfactual_outcome}")
print(f"Difference: {result.difference}")

# Decision Validation
print("\n4. Decision Validation")
print("-" * 60)

validator = TemporalDecisionValidator(chain, encoder)

decision = {"has_umbrella": False, "is_outside": True}
validation = validator.validate_decision(
    decision=decision,
    current_state={"has_umbrella": True},
    past_states=[{"has_umbrella": True}],
    future_constraints=[{"is_sick": False}]
)

print(f"Decision: {decision}")
print(f"Valid: {validation.is_valid}")
print(f"Past consistent: {validation.past_consistent}")
print(f"Present consistent: {validation.present_consistent}")
print(f"Future consistent: {validation.future_consistent}")
print(f"Confidence: {validation.confidence:.4f}")

if validation.violations:
    print(f"Violations: {validation.violations}")
