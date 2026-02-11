"""
CounterfactualReasoner - Performs counterfactual reasoning.

Answers "what if" questions by creating alternative timelines.
"""

from dataclasses import dataclass
from typing import Any

from grilly.experimental.temporal.state import TemporalState, TemporalWorldModel


@dataclass
class CounterfactualQuery:
    """A counterfactual query: what if X had been different at time T?"""

    intervention_time: int
    variable: str
    actual_value: Any
    counterfactual_value: Any
    query_time: int  # What time to check the effect
    query_variable: str | None = None  # What variable to check


@dataclass
class CounterfactualResult:
    """Result of counterfactual reasoning."""

    query: CounterfactualQuery
    actual_outcome: dict[str, Any]
    counterfactual_outcome: dict[str, Any]
    difference: dict[str, tuple[Any, Any]]  # var -> (actual, counterfactual)
    causal_path: list[str]  # Chain of causation


class CounterfactualReasoner:
    """
    Performs counterfactual reasoning.

    Process:
    1. Take actual timeline
    2. Intervene: change variable X at time T
    3. Re-propagate forward from T
    4. Compare actual vs counterfactual outcomes
    """

    def __init__(self, world_model: TemporalWorldModel):
        """Initialize the instance."""

        self.world = world_model
        self.dim = world_model.dim

    def intervene(self, time: int, variable: str, new_value: Any) -> str:
        """
        Create a counterfactual branch by intervening on a variable.

        Returns branch_id for this counterfactual world.
        """
        branch_id = f"cf_{time}_{variable}_{new_value}"

        # Copy timeline up to intervention point
        cf_timeline = {}
        for t, state in self.world.timeline.items():
            if t < time:
                # Before intervention: same as actual
                cf_timeline[t] = TemporalState(
                    time=t,
                    variables=state.variables.copy(),
                    vector=state.vector.copy(),
                    is_counterfactual=True,
                )

        # At intervention: apply the change
        if time in self.world.timeline:
            actual_vars = self.world.timeline[time].variables.copy()
            actual_vars[variable] = new_value  # THE INTERVENTION

            state_vec = self.world.causal_chain.encode_state(actual_vars)
            temporal_vec = self.world.temporal_encoder.bind_with_time(state_vec, time)

            cf_timeline[time] = TemporalState(
                time=time, variables=actual_vars, vector=temporal_vec, is_counterfactual=True
            )

        # After intervention: propagate forward
        if time in cf_timeline:
            current_vars = cf_timeline[time].variables
            max_time = max(self.world.timeline.keys()) if self.world.timeline else time

            for t in range(time + 1, max_time + 1):
                future_vars = self.world.causal_chain.propagate_forward(current_vars)

                state_vec = self.world.causal_chain.encode_state(future_vars)
                temporal_vec = self.world.temporal_encoder.bind_with_time(state_vec, t)

                cf_timeline[t] = TemporalState(
                    time=t, variables=future_vars, vector=temporal_vec, is_counterfactual=True
                )

                current_vars = future_vars

        self.world.counterfactual_branches[branch_id] = cf_timeline
        return branch_id

    def query_counterfactual(self, query: CounterfactualQuery) -> CounterfactualResult:
        """
        Answer a counterfactual query.
        """
        # Create intervention
        branch_id = self.intervene(
            query.intervention_time, query.variable, query.counterfactual_value
        )

        cf_timeline = self.world.counterfactual_branches[branch_id]

        # Get actual outcome at query time
        actual_state = self.world.timeline.get(query.query_time)
        actual_outcome = actual_state.variables if actual_state else {}

        # Get counterfactual outcome at query time
        cf_state = cf_timeline.get(query.query_time)
        cf_outcome = cf_state.variables if cf_state else {}

        # Compute differences
        differences = {}
        all_vars = set(actual_outcome.keys()) | set(cf_outcome.keys())

        for var in all_vars:
            actual_val = actual_outcome.get(var)
            cf_val = cf_outcome.get(var)
            if actual_val != cf_val:
                differences[var] = (actual_val, cf_val)

        # Trace causal path
        causal_path = self._trace_causal_path(
            query.intervention_time,
            query.variable,
            query.query_time,
            query.query_variable,
            cf_timeline,
        )

        return CounterfactualResult(
            query=query,
            actual_outcome=actual_outcome,
            counterfactual_outcome=cf_outcome,
            difference=differences,
            causal_path=causal_path,
        )

    def _trace_causal_path(
        self,
        start_time: int,
        start_var: str,
        end_time: int,
        end_var: str | None,
        cf_timeline: dict[int, TemporalState],
    ) -> list[str]:
        """Trace the causal chain from intervention to outcome."""
        path = [f"t={start_time}: intervene on {start_var}"]

        for t in range(start_time, end_time + 1):
            if t in cf_timeline:
                # Find which rules fired
                state = cf_timeline[t]
                for rule in self.world.causal_chain.rules:
                    if self.world.causal_chain.check_condition(state.variables, rule.conditions):
                        effects_str = ", ".join(f"{k}={v}" for k, v in rule.effects.items())
                        path.append(f"t={t}: {rule.name} -> {effects_str}")

        if end_var:
            final_state = cf_timeline.get(end_time)
            if final_state:
                final_val = final_state.variables.get(end_var)
                path.append(f"t={end_time}: {end_var} = {final_val}")

        return path
