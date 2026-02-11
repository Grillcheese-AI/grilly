"""
Example: Temporal Reasoning

Demonstrates temporal state tracking, causal propagation, counterfactual
reasoning, and decision validation across time.
"""

from grilly.experimental.temporal import (
    CounterfactualQuery,
    CounterfactualReasoner,
    TemporalDecisionValidator,
    TemporalWorldModel,
)
from grilly.experimental.vsa import HolographicOps

print("=" * 60)
print("Temporal Reasoning Examples")
print("=" * 60)

dim = 2048

# Temporal World Model
print("\n1. Temporal World Model")
print("-" * 60)

world = TemporalWorldModel(dim=dim)

world.causal_chain.add_rule(
    name="rain_wet", conditions={"raining": True}, effects={"wet": True}, probability=1.0
)

world.causal_chain.add_rule(
    name="wet_sick", conditions={"wet": True, "cold": True}, effects={"sick": True}, probability=1.0
)

world.set_state(0, {"raining": True, "wet": True, "cold": True})
print("Initial state set at t=0")

pred = world.predict_future(from_time=0, steps=2)
print(f"Predicted future states: {list(pred.keys())}")

# Counterfactual Reasoning
print("\n2. Counterfactual Reasoning")
print("-" * 60)

reasoner = CounterfactualReasoner(world)
query = CounterfactualQuery(
    intervention_time=0,
    variable="raining",
    actual_value=True,
    counterfactual_value=False,
    query_time=2,
    query_variable="wet",
)

cf_result = reasoner.query_counterfactual(query)

print(f"Actual outcome at t=2: {cf_result.actual_outcome}")
print(f"Counterfactual outcome at t=2: {cf_result.counterfactual_outcome}")
print(f"Difference: {cf_result.difference}")

# Decision Validation
print("\n3. Decision Validation")
print("-" * 60)

validator = TemporalDecisionValidator(world, reasoner)

decision = {"wet": False}
validation = validator.validate_decision(decision_time=1, decision=decision, check_horizon=3)

print(f"Decision at t=1: {decision}")
print(f"Valid: {validation.is_valid}")
print(f"Past consistent: {validation.past_consistent}")
print(f"Present consistent: {validation.present_consistent}")
print(f"Future consistent: {validation.future_consistent}")
print(f"Confidence: {validation.confidence:.4f}")

if validation.violations:
    print(f"Violations: {validation.violations}")

# Temporal Encoding sanity check
print("\n4. Temporal Encoding")
print("-" * 60)

time_vec_5 = world.temporal_encoder.encode_time(5)
time_vec_6 = world.temporal_encoder.encode_time(6)
similarity = HolographicOps.similarity(time_vec_5, time_vec_6)
print(f"Similarity between t=5 and t=6: {similarity:.4f}")
