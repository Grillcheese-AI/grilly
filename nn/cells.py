"""
Cell Layers - Place cells, Time cells, Theta-Gamma encoding

Uses: place-cell.glsl, time-cell.glsl, theta-gamma-encoding.glsl

Reference: ref/brain/gpu_brain.py
"""

import numpy as np

from .module import Module


class PlaceCell(Module):
    """
    Place Cell layer - Spatial encoding.

    Uses: place-cell.glsl

    Reference: ref/brain/gpu_brain.py compute_place_cells
    """

    def __init__(
        self,
        n_neurons: int,
        spatial_dims: int = 2,
        field_width: float = 1.0,
        max_rate: float = 20.0,
        baseline_rate: float = 0.1,
    ):
        """
        Initialize PlaceCell layer.

        Args:
            n_neurons: Number of place cells
            spatial_dims: Number of spatial dimensions (2D or 3D)
            field_width: Width of place fields
            max_rate: Maximum firing rate
            baseline_rate: Baseline firing rate
        """
        super().__init__()
        self.n_neurons = n_neurons
        self.spatial_dims = spatial_dims
        self.field_width = field_width
        self.max_rate = max_rate
        self.baseline_rate = baseline_rate

        # Place field centers (randomly initialized)
        self.field_centers = np.random.randn(n_neurons, spatial_dims).astype(np.float32)
        self._parameters["field_centers"] = self.field_centers

    def forward(self, agent_position: np.ndarray) -> np.ndarray:
        """
        Forward pass - compute place cell firing rates.

        Args:
            agent_position: Agent position (spatial_dims,) or (batch, spatial_dims)

        Returns:
            Firing rates (n_neurons,) or (batch, n_neurons)
        """
        backend = self._get_backend()
        return backend.place_cell(
            agent_position,
            self.field_centers,
            field_width=self.field_width,
            max_rate=self.max_rate,
            baseline_rate=self.baseline_rate,
        )

    def __repr__(self):
        """Return a debug representation."""

        return f"PlaceCell(n_neurons={self.n_neurons}, spatial_dims={self.spatial_dims})"


class TimeCell(Module):
    """
    Time Cell layer - Temporal encoding.

    Uses: time-cell.glsl

    Reference: ref/brain/gpu_brain.py compute_time_cells
    """

    def __init__(
        self,
        n_neurons: int,
        temporal_width: float = 1.0,
        max_rate: float = 15.0,
        baseline_rate: float = 0.1,
    ):
        """
        Initialize TimeCell layer.

        Args:
            n_neurons: Number of time cells
            temporal_width: Width of time fields
            max_rate: Maximum firing rate
            baseline_rate: Baseline firing rate
        """
        super().__init__()
        self.n_neurons = n_neurons
        self.temporal_width = temporal_width
        self.max_rate = max_rate
        self.baseline_rate = baseline_rate

        # Preferred times (randomly initialized)
        self.preferred_times = np.random.uniform(0, 10, n_neurons).astype(np.float32)
        self._parameters["preferred_times"] = self.preferred_times

        # Membrane state for dynamics
        self.membrane = np.zeros(n_neurons, dtype=np.float32)
        self._buffers["membrane"] = self.membrane

    def forward(self, current_time: float) -> np.ndarray:
        """
        Forward pass - compute time cell firing rates.

        Args:
            current_time: Current time value

        Returns:
            Firing rates (n_neurons,)
        """
        backend = self._get_backend()
        rates, self.membrane = backend.time_cell(
            current_time,
            self.preferred_times,
            temporal_width=self.temporal_width,
            max_rate=self.max_rate,
            baseline_rate=self.baseline_rate,
            membrane_state=self.membrane,
        )
        return rates

    def reset(self):
        """Reset membrane state"""
        self.membrane.fill(0)

    def __repr__(self):
        """Return a debug representation."""

        return f"TimeCell(n_neurons={self.n_neurons})"


class ThetaGammaEncoder(Module):
    """
    Theta-Gamma Encoder - Phase-amplitude coupling.

    Uses: theta-gamma-encoding.glsl

    Reference: ref/brain/gpu_brain.py compute_theta_gamma
    """

    def __init__(
        self,
        n_theta: int = 8,
        n_gamma: int = 40,
        theta_freq: float = 8.0,
        gamma_freq: float = 40.0,
        coupling_strength: float = 0.5,
    ):
        """
        Initialize ThetaGammaEncoder layer.

        Args:
            n_theta: Number of theta phase bins
            n_gamma: Number of gamma amplitude bins
            theta_freq: Theta frequency (Hz)
            gamma_freq: Gamma frequency (Hz)
            coupling_strength: Coupling strength between theta and gamma
        """
        super().__init__()
        self.n_theta = n_theta
        self.n_gamma = n_gamma
        self.theta_freq = theta_freq
        self.gamma_freq = gamma_freq
        self.coupling_strength = coupling_strength

    def forward(self, time: float) -> np.ndarray:
        """
        Forward pass - compute theta-gamma encoding.

        Args:
            time: Current time value

        Returns:
            Theta-gamma encoding (n_theta * n_gamma,)
        """
        backend = self._get_backend()

        # Try GPU shader if available
        if hasattr(backend, "shaders") and "theta-gamma-encoding" in backend.shaders:
            try:
                # GPU theta-gamma encoding would go here
                # For now, use CPU fallback
                pass
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        theta_phase = 2 * np.pi * self.theta_freq * time
        gamma_phase = 2 * np.pi * self.gamma_freq * time

        theta_encoding = np.sin(theta_phase + np.linspace(0, 2 * np.pi, self.n_theta))
        gamma_encoding = np.sin(gamma_phase + np.linspace(0, 2 * np.pi, self.n_gamma))

        # Phase-amplitude coupling
        coupling = self.coupling_strength * np.outer(theta_encoding, gamma_encoding)
        return coupling.flatten().astype(np.float32)

    def __repr__(self):
        """Return a debug representation."""

        return f"ThetaGammaEncoder(n_theta={self.n_theta}, n_gamma={self.n_gamma})"
