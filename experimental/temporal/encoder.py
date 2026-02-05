"""
TemporalEncoder - Encodes time points as hypervectors.

Supports discrete and continuous time, temporal binding/unbinding.
"""

import numpy as np
from typing import Dict
from grilly.experimental.vsa.ops import HolographicOps


class TemporalEncoder:
    """
    Encodes time points as hypervectors.
    
    Key properties:
    - Adjacent times have similar vectors (enables temporal smoothing)
    - Can bind any state with a time marker
    - Supports both discrete and continuous time
    - Enables temporal unbinding (retrieve state at time T)
    """
    
    DEFAULT_DIM = 4096
    DEFAULT_MAX_TIME = 1000
    
    def __init__(self, dim: int = DEFAULT_DIM, max_time: int = DEFAULT_MAX_TIME):
        self.dim = dim
        self.max_time = max_time
        
        # Base time vector (T=0)
        self.t0 = HolographicOps.random_vector(dim, seed=8000)
        
        # Time increment vector (add this to get next time)
        self.dt = HolographicOps.random_vector(dim, seed=8001)
        
        # Pre-compute time vectors for efficiency
        self.time_vectors: Dict[int, np.ndarray] = {}
        self._precompute_times()
        
        # Special temporal markers
        self.markers = {
            "past": HolographicOps.random_vector(dim, seed=8100),
            "present": HolographicOps.random_vector(dim, seed=8101),
            "future": HolographicOps.random_vector(dim, seed=8102),
            "always": HolographicOps.random_vector(dim, seed=8103),
            "never": HolographicOps.random_vector(dim, seed=8104),
            "before": HolographicOps.random_vector(dim, seed=8105),
            "after": HolographicOps.random_vector(dim, seed=8106),
            "during": HolographicOps.random_vector(dim, seed=8107),
        }
    
    def _precompute_times(self):
        """Pre-compute time vectors using power-of-dt method."""
        self.time_vectors[0] = self.t0.copy()
        
        current = self.t0.copy()
        for t in range(1, self.max_time):
            # T(n) = T(n-1) ⊗ dt
            current = HolographicOps.convolve(current, self.dt)
            self.time_vectors[t] = current.copy()
    
    def encode_time(self, t: int) -> np.ndarray:
        """Get time vector for discrete time t."""
        if t in self.time_vectors:
            return self.time_vectors[t]
        
        # Compute on the fly for times beyond precomputed range
        if t >= self.max_time:
            current = self.time_vectors[self.max_time - 1]
            for _ in range(t - self.max_time + 1):
                current = HolographicOps.convolve(current, self.dt)
            return current
        
        return self.time_vectors.get(t, self.t0)
    
    def encode_time_continuous(self, t: float) -> np.ndarray:
        """
        Encode continuous time via interpolation.
        
        Uses fractional binding: T(2.5) ≈ interpolate(T(2), T(3))
        """
        t_floor = int(np.floor(t))
        t_ceil = int(np.ceil(t))
        frac = t - t_floor
        
        v_floor = self.encode_time(t_floor)
        v_ceil = self.encode_time(t_ceil)
        
        # Linear interpolation in vector space
        return HolographicOps.bundle([
            v_floor * (1 - frac),
            v_ceil * frac
        ], normalize=True)
    
    def bind_with_time(self, state: np.ndarray, t: int) -> np.ndarray:
        """Bind a state vector with a time marker."""
        time_vec = self.encode_time(t)
        return HolographicOps.convolve(state, time_vec)
    
    def unbind_time(self, temporal_state: np.ndarray, t: int) -> np.ndarray:
        """Retrieve state from a temporal binding."""
        time_vec = self.encode_time(t)
        return HolographicOps.correlate(temporal_state, time_vec)
    
    def get_temporal_relation(self, t1: int, t2: int) -> str:
        """Determine temporal relation between two times."""
        if t1 < t2:
            return "before"
        elif t1 > t2:
            return "after"
        else:
            return "simultaneous"
