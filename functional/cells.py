"""
Functional Cell Operations

Uses: place-cell.glsl, time-cell.glsl, theta-gamma-encoding.glsl
"""

import numpy as np

from ._helpers import _to_numpy


def place_cell(
    agent_position: np.ndarray,
    field_centers: np.ndarray,
    field_width: float = 1.0,
    max_rate: float = 20.0,
    baseline_rate: float = 0.1,
) -> np.ndarray:
    """
    Compute place cell firing rates.

    Uses: place-cell.glsl

    Args:
        agent_position: Agent position (spatial_dims,) or (batch, spatial_dims)
        field_centers: Place field centers (n_neurons, spatial_dims)
        field_width: Width of place fields
        max_rate: Maximum firing rate
        baseline_rate: Baseline firing rate

    Returns:
        Firing rates (n_neurons,) or (batch, n_neurons)
    """
    try:
        from grilly.backend import _bridge

        result = _bridge.place_cell(
            agent_position,
            field_centers,
            field_width=field_width,
            max_rate=max_rate,
            baseline_rate=baseline_rate,
        )
        if result is not None:
            return _to_numpy(result)
    except (ImportError, Exception):
        pass
    pos = np.asarray(agent_position, dtype=np.float32)
    centers = np.asarray(field_centers, dtype=np.float32)
    if pos.ndim == 1:
        pos = pos[None, :]
    diff = pos[:, None, :] - centers[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    rates = baseline_rate + (max_rate - baseline_rate) * np.exp(
        -d2 / (2.0 * field_width * field_width + 1e-8)
    )
    return rates.astype(np.float32).squeeze(0) if np.asarray(agent_position).ndim == 1 else rates.astype(np.float32)


def time_cell(
    current_time: float,
    preferred_times: np.ndarray,
    temporal_width: float = 1.0,
    max_rate: float = 15.0,
    baseline_rate: float = 0.1,
    membrane_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute time cell firing rates.

    Uses: time-cell.glsl

    Args:
        current_time: Current time value
        preferred_times: Preferred firing times (n_neurons,)
        temporal_width: Width of time fields
        max_rate: Maximum firing rate
        baseline_rate: Baseline firing rate
        membrane_state: Optional membrane state (n_neurons,)

    Returns:
        (firing_rates, updated_membrane_state)
    """
    try:
        from grilly.backend import _bridge

        result = _bridge.time_cell(
            current_time,
            preferred_times,
            time_constant=temporal_width,
            max_rate=max_rate,
            baseline_rate=baseline_rate,
            membrane_state=membrane_state,
        )
        if result is not None:
            rates, mem = result
            return _to_numpy(rates), _to_numpy(mem)
    except (ImportError, Exception):
        pass
    pref = np.asarray(preferred_times, dtype=np.float32)
    rates = baseline_rate + (max_rate - baseline_rate) * np.exp(
        -((float(current_time) - pref) ** 2) / (2.0 * temporal_width * temporal_width + 1e-8)
    )
    if membrane_state is None:
        mem = rates.astype(np.float32)
    else:
        mem = (0.9 * np.asarray(membrane_state, dtype=np.float32) + 0.1 * rates).astype(np.float32)
    return rates.astype(np.float32), mem


def theta_gamma_encoding(
    time: float,
    n_theta: int = 8,
    n_gamma: int = 40,
    theta_freq: float = 8.0,
    gamma_freq: float = 40.0,
    coupling_strength: float = 0.5,
) -> np.ndarray:
    """
    Compute theta-gamma encoding.

    Uses: theta-gamma-encoding.glsl

    Args:
        time: Current time value
        n_theta: Number of theta phase bins
        n_gamma: Number of gamma amplitude bins
        theta_freq: Theta frequency (Hz)
        gamma_freq: Gamma frequency (Hz)
        coupling_strength: Coupling strength between theta and gamma

    Returns:
        Theta-gamma encoding (n_theta * n_gamma,)
    """
    # CPU fallback
    theta_phase = 2 * np.pi * theta_freq * time
    gamma_phase = 2 * np.pi * gamma_freq * time

    theta_encoding = np.sin(theta_phase + np.linspace(0, 2 * np.pi, n_theta))
    gamma_encoding = np.sin(gamma_phase + np.linspace(0, 2 * np.pi, n_gamma))

    # Phase-amplitude coupling
    coupling = coupling_strength * np.outer(theta_encoding, gamma_encoding)
    return coupling.flatten().astype(np.float32)
