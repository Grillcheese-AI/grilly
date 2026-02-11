"""
Functional Bridge Operations (Continuous ↔ Spike)

Uses: bridge-continuous-to-spike.glsl, bridge-spike-to-continuous.glsl,
      bridge-temporal-weights.glsl
"""

import numpy as np


def _get_backend():
    """Get backend instance"""
    try:
        from ..backend.compute import Compute

        return Compute()
    except Exception:
        return None


def continuous_to_spikes(
    continuous: np.ndarray,
    num_timesteps: int = 10,
    encoding_type: int = 0,
    projection_weights: np.ndarray = None,
    projection_bias: np.ndarray = None,
) -> np.ndarray:
    """
    Convert continuous features to spike trains.

    Uses: bridge-continuous-to-spike.glsl

    Args:
        continuous: Continuous values (batch, features)
        num_timesteps: Number of time steps for spike train
        encoding_type: Encoding type (0=rate, 1=temporal, 2=phase)
        projection_weights: Optional projection weights (spike_dim, features)
        projection_bias: Optional projection bias (spike_dim,)

    Returns:
        Spike trains (batch, num_timesteps, spike_dim)
    """
    from grilly import Compute

    backend = Compute()
    return backend.continuous_to_spikes(
        continuous,
        num_timesteps=num_timesteps,
        encoding_type=encoding_type,
        projection_weights=projection_weights,
        projection_bias=projection_bias,
    )


def spikes_to_continuous(
    spikes: np.ndarray,
    encoding_type: int = 0,
    time_window: int = 5,
    temporal_weights: np.ndarray = None,
    projection_weights: np.ndarray = None,
    projection_bias: np.ndarray = None,
) -> np.ndarray:
    """
    Convert spike trains to continuous features.

    Uses: bridge-spike-to-continuous.glsl

    Args:
        spikes: Spike trains (batch, time, features)
        encoding_type: 0=rate, 1=temporal, 2=phase
        time_window: Window for rate encoding
        temporal_weights: Weights for temporal encoding (time,)
        projection_weights: Optional projection weights (output_dim, spike_dim)
        projection_bias: Optional projection bias (output_dim,)

    Returns:
        Continuous values (batch, output_dim)
    """
    from grilly import Compute

    backend = Compute()
    return backend.spikes_to_continuous(
        spikes,
        encoding_type=encoding_type,
        time_window=time_window,
        temporal_weights=temporal_weights,
        projection_weights=projection_weights,
        projection_bias=projection_bias,
    )


def bridge_temporal_weights(weights: np.ndarray, temporal_window: int = 10) -> np.ndarray:
    """
    Apply temporal weighting to bridge operations.

    Uses: bridge-temporal-weights.glsl

    Args:
        weights: Weight matrix (out_features, in_features)
        temporal_window: Temporal window size

    Returns:
        Temporally weighted weights
    """
    backend = _get_backend()

    # Try GPU shader if available
    if backend and hasattr(backend, "shaders") and "bridge-temporal-weights" in backend.shaders:
        try:
            # GPU temporal weights would go here
            # For now, use CPU fallback
            pass
        except Exception:
            pass  # Fall back to CPU

    # CPU fallback - Apply temporal decay to weights
    temporal_decay = np.exp(-np.arange(temporal_window) / temporal_window)
    # Simple implementation: return weights with temporal scaling
    return weights * temporal_decay[-1]  # Simplified
