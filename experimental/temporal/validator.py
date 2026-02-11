"""
TemporalDecisionValidator - Validates decisions against temporal constraints.

Checks consistency across past, present, and future.
"""

from dataclasses import dataclass
from typing import Any

from grilly.experimental.temporal.counterfactual import CounterfactualReasoner
from grilly.experimental.temporal.state import TemporalWorldModel


@dataclass
class ValidationResult:
    """Result of temporal validation."""

    is_valid: bool
    past_consistent: bool
    present_consistent: bool
    future_consistent: bool
    violations: list[str]
    confidence: float


class TemporalDecisionValidator:
    """
    Validates decisions against past, present, and future states.

    A decision is valid if:
    1. It's consistent with established past (doesn't contradict history)
    2. It's achievable from present (preconditions met)
    3. It doesn't lead to invalid future states (no constraint violations)
    """

    def __init__(
        self, world_model: TemporalWorldModel, counterfactual_reasoner: CounterfactualReasoner
    ):
        """Initialize the instance."""

        self.world = world_model
        self.cf_reasoner = counterfactual_reasoner
        self.dim = world_model.dim

    def validate_decision(
        self,
        decision_time: int,
        decision: dict[str, Any],  # Variable changes
        check_horizon: int = 5,  # How far into future to check
    ) -> ValidationResult:
        """
        Validate a decision against temporal constraints.
        """
        violations = []

        # 1. Check past consistency
        past_ok = self._check_past_consistency(decision_time, decision)
        if not past_ok:
            violations.append("Decision contradicts established past")

        # 2. Check present preconditions
        present_ok = self._check_present_preconditions(decision_time, decision)
        if not present_ok:
            violations.append("Preconditions not met in present state")

        # 3. Check future consistency
        future_ok, future_violations = self._check_future_consistency(
            decision_time, decision, check_horizon
        )
        violations.extend(future_violations)

        # Compute confidence
        checks_passed = sum([past_ok, present_ok, future_ok])
        confidence = checks_passed / 3.0

        return ValidationResult(
            is_valid=len(violations) == 0,
            past_consistent=past_ok,
            present_consistent=present_ok,
            future_consistent=future_ok,
            violations=violations,
            confidence=confidence,
        )

    def _check_past_consistency(self, decision_time: int, decision: dict[str, Any]) -> bool:
        """
        Check that decision doesn't contradict past.
        """
        for var, new_val in decision.items():
            # Check if this contradicts any past state
            for t in range(0, decision_time):
                past_state = self.world.get_state(t)
                if past_state and var in past_state.variables:
                    past_val = past_state.variables[var]
                    # Check for logical contradiction
                    if self._is_contradiction(var, past_val, new_val):
                        return False

        return True

    def _is_contradiction(self, var: str, past_val: Any, new_val: Any) -> bool:
        """Check if values are contradictory (domain-specific)."""
        # Simple version: some values are mutually exclusive
        contradictions = {
            ("alive", "dead"),
            ("present", "absent"),
            ("open", "closed"),
            (True, False),
        }

        pair = (past_val, new_val)
        reverse_pair = (new_val, past_val)

        return pair in contradictions or reverse_pair in contradictions

    def _check_present_preconditions(self, decision_time: int, decision: dict[str, Any]) -> bool:
        """
        Check that preconditions for decision are met.
        """
        present_state = self.world.get_state(decision_time)
        if not present_state:
            return True  # No state to check against

        # For each decision variable, check if change is possible
        for var, new_val in decision.items():
            present_state.variables.get(var)

            # Check causal rules: is there a rule that allows this transition?
            transition_possible = False
            for rule in self.world.causal_chain.rules:
                if var in rule.effects and rule.effects[var] == new_val:
                    # Check if rule conditions can be met
                    if self.world.causal_chain.check_condition(
                        present_state.variables, rule.conditions
                    ):
                        transition_possible = True
                        break

            # Also allow direct setting if no rules govern this variable
            rules_govern_var = any(var in r.effects for r in self.world.causal_chain.rules)
            if not rules_govern_var:
                transition_possible = True

            if not transition_possible:
                return False

        return True

    def _check_future_consistency(
        self, decision_time: int, decision: dict[str, Any], horizon: int
    ) -> tuple[bool, list[str]]:
        """
        Check that decision doesn't lead to invalid future states.
        """
        violations = []

        # Get current state at decision time
        current_state = self.world.get_state(decision_time)
        if not current_state:
            return True, []

        # Apply decision
        new_vars = current_state.variables.copy()
        new_vars.update(decision)

        # Propagate forward
        for step in range(1, horizon + 1):
            future_time = decision_time + step
            future_vars = self.world.causal_chain.propagate_forward(new_vars, steps=1)

            # Check constraints
            is_valid, constraint_violations = self.world.validate_state(future_time, future_vars)

            if not is_valid:
                for v in constraint_violations:
                    violations.append(f"t={future_time}: {v}")

            new_vars = future_vars

        return len(violations) == 0, violations

    def validate_with_counterfactual(
        self,
        decision_time: int,
        decision: dict[str, Any],
        counterfactual_var: str,
        counterfactual_val: Any,
        counterfactual_time: int,
    ) -> tuple[ValidationResult, ValidationResult]:
        """
        Compare decision validity in actual vs counterfactual world.
        """
        # Validate in actual world
        actual_result = self.validate_decision(decision_time, decision)

        # Create counterfactual branch
        branch_id = self.cf_reasoner.intervene(
            counterfactual_time, counterfactual_var, counterfactual_val
        )

        # Temporarily swap timelines
        actual_timeline = self.world.timeline
        self.world.timeline = self.world.counterfactual_branches[branch_id]

        # Validate in counterfactual world
        cf_result = self.validate_decision(decision_time, decision)

        # Restore actual timeline
        self.world.timeline = actual_timeline

        return actual_result, cf_result
