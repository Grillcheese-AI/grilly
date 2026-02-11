"""
CausalChain - Manages causal relationships between temporal states.

Provides causal rules and forward/backward propagation.
"""

import numpy as np

# Stable hashing (BLAKE3) for deterministic temporal seeding
try:
    from utils.stable_hash import stable_u32
except ModuleNotFoundError:
    try:
        from grilly.utils.stable_hash import stable_u32  # type: ignore
    except Exception:
        stable_u32 = None  # type: ignore

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from grilly.experimental.vsa.ops import HolographicOps


@dataclass
class CausalRule:
    """
    A rule that transforms state at T to state at T+1.

    Format: condition -> effect
    Example: "if raining AND no_umbrella -> wet"
    """

    name: str
    conditions: dict[str, Any]  # Variables that must match
    effects: dict[str, Any]  # Variables that change
    vector: np.ndarray  # Holographic encoding of the rule
    probability: float = 1.0  # Deterministic by default


class CausalChain:
    """
    Manages causal relationships between temporal states.

    Key operations:
    - propagate_forward: Given state at T, compute state at T+1
    - trace_backward: Given state at T, find causes at T-1
    - intervene: Change a variable and recompute effects
    """

    DEFAULT_DIM = 4096

    def __init__(self, dim: int = DEFAULT_DIM):
        """Initialize the instance."""

        self.dim = dim
        self.rules: list[CausalRule] = []

        # Variable encodings
        self.variable_vectors: dict[str, np.ndarray] = {}
        self.value_vectors: dict[str, dict[Any, np.ndarray]] = defaultdict(dict)

    def encode_variable(self, name: str) -> np.ndarray:
        """Get/create encoding for a variable name."""
        if name not in self.variable_vectors:
            self.variable_vectors[name] = HolographicOps.random_vector(
                self.dim,
                seed=(
                    stable_u32("name", name, domain="grilly.temporal") % (2**31)
                    if stable_u32
                    else 0
                ),
            )
        return self.variable_vectors[name]

    def encode_value(self, variable: str, value: Any) -> np.ndarray:
        """Get/create encoding for a variable's value."""
        if value not in self.value_vectors[variable]:
            self.value_vectors[variable][value] = HolographicOps.random_vector(
                self.dim,
                seed=(
                    stable_u32("var", variable, "val", str(value), domain="grilly.temporal")
                    % (2**31)
                    if stable_u32
                    else 0
                ),
            )
        return self.value_vectors[variable][value]

    def encode_assignment(self, variable: str, value: Any) -> np.ndarray:
        """Encode variable=value as a bound vector."""
        var_vec = self.encode_variable(variable)
        val_vec = self.encode_value(variable, value)
        return HolographicOps.convolve(var_vec, val_vec)

    def encode_state(self, variables: dict[str, Any]) -> np.ndarray:
        """Encode a complete state as superposition of assignments."""
        if not variables:
            # Return zero vector for empty state
            return np.zeros(self.dim, dtype=np.float32)

        assignments = []
        for var, val in variables.items():
            assignments.append(self.encode_assignment(var, val))
        return HolographicOps.bundle(assignments)

    def add_rule(
        self,
        name: str,
        conditions: dict[str, Any],
        effects: dict[str, Any],
        probability: float = 1.0,
    ):
        """Add a causal rule."""
        # Encode rule as: condition_encoding -> effect_encoding
        cond_vec = self.encode_state(conditions)
        effect_vec = self.encode_state(effects)
        rule_vec = HolographicOps.convolve(cond_vec, effect_vec)

        rule = CausalRule(
            name=name,
            conditions=conditions,
            effects=effects,
            vector=rule_vec,
            probability=probability,
        )
        self.rules.append(rule)

    def check_condition(self, state: dict[str, Any], condition: dict[str, Any]) -> bool:
        """Check if state satisfies condition."""
        for var, required_val in condition.items():
            if var not in state or state[var] != required_val:
                return False
        return True

    def propagate_forward(self, state: dict[str, Any], steps: int = 1) -> dict[str, Any]:
        """
        Compute future state by applying causal rules.

        This is deterministic forward simulation.
        """
        current = state.copy()

        for _ in range(steps):
            next_state = current.copy()

            for rule in self.rules:
                if self.check_condition(current, rule.conditions):
                    # Apply effects
                    if np.random.random() < rule.probability:
                        next_state.update(rule.effects)

            current = next_state

        return current

    def trace_causes(
        self, effect_state: dict[str, Any], effect_var: str
    ) -> list[tuple[CausalRule, dict[str, Any]]]:
        """
        Find what rules could have caused a specific variable value.

        Backward reasoning: given effect, find possible causes.
        """
        target_value = effect_state.get(effect_var)
        possible_causes = []

        for rule in self.rules:
            # Check if this rule produces the target effect
            if effect_var in rule.effects and rule.effects[effect_var] == target_value:
                possible_causes.append((rule, rule.conditions))

        return possible_causes
