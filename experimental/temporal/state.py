"""
TemporalState and TemporalWorldModel - Full timeline management.

Provides state tracking across time with causal propagation.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
from grilly.experimental.temporal.encoder import TemporalEncoder
from grilly.experimental.temporal.causal import CausalChain
from grilly.experimental.vsa.ops import HolographicOps


@dataclass
class TemporalState:
    """
    A state at a specific time point.
    
    Contains:
    - Variables and their values
    - The holographic encoding
    - Links to causes (past states that caused this)
    - Links to effects (future states this causes)
    """
    time: int
    variables: Dict[str, Any]
    vector: np.ndarray
    causes: List['TemporalState'] = field(default_factory=list)
    effects: List['TemporalState'] = field(default_factory=list)
    is_counterfactual: bool = False
    
    def __hash__(self):
        """Execute hash."""

        return hash((self.time, tuple(sorted(self.variables.items()))))


class TemporalWorldModel:
    """
    Complete temporal world model with:
    - State history (past)
    - Current state (present)
    - Predicted states (future)
    - Counterfactual branches
    
    Supports validation across any time point.
    """
    
    DEFAULT_DIM = 4096
    
    def __init__(self, dim: int = DEFAULT_DIM):
        """Initialize the instance."""

        self.dim = dim
        self.temporal_encoder = TemporalEncoder(dim=dim)
        self.causal_chain = CausalChain(dim=dim)
        
        # Timeline: time -> state
        self.timeline: Dict[int, TemporalState] = {}
        self.current_time: int = 0
        
        # Counterfactual branches: (branch_id, time) -> state
        self.counterfactual_branches: Dict[str, Dict[int, TemporalState]] = {}
        
        # Constraints that must hold
        self.temporal_constraints: List[Callable[[int, Dict], Tuple[bool, str]]] = []
        
        # Holographic timeline representation
        self.timeline_vector: np.ndarray = np.zeros(dim, dtype=np.float32)
    
    def set_state(
        self,
        time: int,
        variables: Dict[str, Any],
        is_observation: bool = True
    ) -> TemporalState:
        """Set state at a specific time."""
        # Encode state
        state_vec = self.causal_chain.encode_state(variables)
        temporal_vec = self.temporal_encoder.bind_with_time(state_vec, time)
        
        state = TemporalState(
            time=time,
            variables=variables.copy(),
            vector=temporal_vec
        )
        
        # Link to previous state if exists
        if time - 1 in self.timeline:
            prev_state = self.timeline[time - 1]
            state.causes.append(prev_state)
            prev_state.effects.append(state)
        
        self.timeline[time] = state
        
        # Update holographic timeline
        self.timeline_vector = HolographicOps.bundle([
            self.timeline_vector * 0.9,  # Decay old
            temporal_vec
        ], normalize=False)
        
        # Update current time
        if time > self.current_time:
            self.current_time = time
        
        return state
    
    def get_state(self, time: int) -> Optional[TemporalState]:
        """Get state at a specific time."""
        return self.timeline.get(time)
    
    def query_variable_at_time(
        self,
        variable: str,
        time: int
    ) -> Optional[Any]:
        """Query a specific variable at a specific time."""
        state = self.get_state(time)
        if state:
            return state.variables.get(variable)
        return None
    
    def predict_future(
        self,
        from_time: int,
        steps: int
    ) -> Dict[int, TemporalState]:
        """
        Predict future states using causal rules.
        """
        if from_time not in self.timeline:
            return {}
        
        predictions = {}
        current_vars = self.timeline[from_time].variables.copy()
        
        for step in range(1, steps + 1):
            future_time = from_time + step
            future_vars = self.causal_chain.propagate_forward(current_vars)
            
            # Create predicted state
            state_vec = self.causal_chain.encode_state(future_vars)
            temporal_vec = self.temporal_encoder.bind_with_time(state_vec, future_time)
            
            predicted_state = TemporalState(
                time=future_time,
                variables=future_vars,
                vector=temporal_vec
            )
            
            predictions[future_time] = predicted_state
            current_vars = future_vars
        
        return predictions
    
    def add_constraint(
        self,
        constraint_fn: Callable[[int, Dict], Tuple[bool, str]],
        name: str = "unnamed"
    ):
        """
        Add a temporal constraint.
        
        constraint_fn(time, variables) -> (is_valid, reason)
        """
        self.temporal_constraints.append(constraint_fn)
    
    def validate_state(
        self,
        time: int,
        variables: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a state against all constraints.
        """
        violations = []
        
        for constraint_fn in self.temporal_constraints:
            is_valid, reason = constraint_fn(time, variables)
            if not is_valid:
                violations.append(reason)
        
        return len(violations) == 0, violations
