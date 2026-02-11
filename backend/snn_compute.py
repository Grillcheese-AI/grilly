"""
High-level SNN interface for spike computation.
"""

import logging

import numpy as np

from .base import VULKAN_AVAILABLE
from .compute import VulkanCompute

# Import config for SNN parameters (optional - for integration with main project)
try:
    import sys
    from pathlib import Path

    # Try to import from parent project if available
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from core.config import LogConfig, SNNConfig

    _config_available = True
except ImportError:
    _config_available = False

logging.basicConfig(level="INFO")
_snn_logger = logging.getLogger(__name__)


class SNNCompute:
    """
    High-level SNN interface for spike computation

    Provides a clean API for processing embeddings through a spiking neural network.
    Uses Vulkan GPU acceleration when available, with CPU fallback.
    """

    # Default parameters (overridden by config if available)
    DEFAULT_N_NEURONS = 1000
    DEFAULT_DT = 0.01
    DEFAULT_TAU_MEM = 5.0
    DEFAULT_V_THRESH = 0.5
    DEFAULT_TIMESTEPS = 50
    DEFAULT_INPUT_SCALE = 20.0

    def __init__(self, n_neurons: int = None, use_vulkan: bool = True):
        """
        Initialize SNN compute engine

        Args:
            n_neurons: Number of neurons (default from config or 1000)
            use_vulkan: Whether to use GPU acceleration
        """
        # Get parameters from config if available
        if _config_available:
            self.n_neurons = n_neurons or SNNConfig.N_NEURONS
            self.dt = SNNConfig.DT
            self.tau_mem = SNNConfig.TAU_MEM
            self.v_thresh = SNNConfig.V_THRESH
            self.timesteps = SNNConfig.TIMESTEPS
            self.input_scale = SNNConfig.INPUT_SCALE
        else:
            self.n_neurons = n_neurons or self.DEFAULT_N_NEURONS
            self.dt = self.DEFAULT_DT
            self.tau_mem = self.DEFAULT_TAU_MEM
            self.v_thresh = self.DEFAULT_V_THRESH
            self.timesteps = self.DEFAULT_TIMESTEPS
            self.input_scale = self.DEFAULT_INPUT_SCALE

        # Initialize GPU backend
        self.use_vulkan = False
        self.backend = None

        if use_vulkan and VULKAN_AVAILABLE:
            try:
                self.backend = VulkanCompute()
                self.use_vulkan = True
                _snn_logger.info("[OK] GPU compute enabled")
            except Exception as e:
                _snn_logger.warning(f"[!] GPU init failed: {e}, using CPU fallback")

        # Neuron state
        self.membrane = np.zeros(self.n_neurons, dtype=np.float32)
        self.refractory = np.zeros(self.n_neurons, dtype=np.float32)

    def reset(self) -> None:
        """Reset neuron state (membrane potential and refractory periods)"""
        self.membrane.fill(0)
        self.refractory.fill(0)

    def forward(self, input_current: np.ndarray) -> np.ndarray:
        """
        Run one timestep of LIF dynamics

        Args:
            input_current: Input current for each neuron (n_neurons,)

        Returns:
            Spike output array (n_neurons,) - 1.0 where spike occurred, 0.0 otherwise
        """
        if self.use_vulkan and self.backend is not None:
            self.membrane, self.refractory, spikes = self.backend.lif_step(
                input_current,
                self.membrane,
                self.refractory,
                dt=self.dt,
                tau_mem=self.tau_mem,
                v_thresh=self.v_thresh,
            )
            return spikes
        else:
            return self._cpu_forward(input_current)

    def _cpu_forward(self, input_current: np.ndarray) -> np.ndarray:
        """CPU fallback for LIF dynamics"""
        decay = np.exp(-self.dt / self.tau_mem)
        self.membrane = self.membrane * decay + input_current * self.dt
        spikes = (self.membrane >= self.v_thresh).astype(np.float32)
        self.membrane = self.membrane * (1 - spikes)  # Reset after spike
        return spikes

    def process(self, embedding: np.ndarray) -> dict:
        """
        Process embedding through SNN pipeline

        Converts embedding to input current, runs through LIF neurons for
        multiple timesteps, and returns spike metrics for visualization.

        Args:
            embedding: Input embedding vector (any length, will be padded/truncated)

        Returns:
            Dictionary with:
                - 'spike_activity': Total spike count across all timesteps
                - 'spikes': Binary spike pattern (n_neurons,)
                - 'firing_rate': Average firing rate (spikes per neuron per timestep)
        """
        # Prepare input current
        input_current = self._prepare_input(embedding)

        # Run simulation
        total_spikes, spike_pattern = self._run_simulation(input_current)

        # Compute firing rate
        firing_rate = total_spikes / (self.n_neurons * self.timesteps)

        return {"spike_activity": total_spikes, "spikes": spike_pattern, "firing_rate": firing_rate}

    def _prepare_input(self, embedding: np.ndarray) -> np.ndarray:
        """Prepare embedding as input current for neurons"""
        # Pad or truncate to n_neurons
        input_current = embedding.astype(np.float32)[: self.n_neurons]
        if len(input_current) < self.n_neurons:
            input_current = np.pad(input_current, (0, self.n_neurons - len(input_current)))

        # Scale to appropriate range for LIF neurons
        # Use absolute value and scale for consistent activity
        return np.abs(input_current) * self.input_scale

    def _run_simulation(self, input_current: np.ndarray) -> tuple:
        """
        Run LIF simulation for multiple timesteps

        Returns:
            Tuple of (total_spike_count, spike_pattern)
        """
        # Reset membrane for fresh simulation
        self.reset()

        # Scale down for accumulation (LIF integrates over time)
        scaled_input = input_current / 4.0

        total_spikes = 0.0
        spike_pattern = np.zeros(self.n_neurons, dtype=np.float32)

        for _ in range(self.timesteps):
            spikes = self.forward(scaled_input)
            total_spikes += float(spikes.sum())
            spike_pattern = np.maximum(spike_pattern, spikes)

        # If no spikes from LIF, use threshold-based fallback for visualization
        if total_spikes == 0:
            threshold_spikes = (input_current >= self.v_thresh).astype(np.float32)
            total_spikes = float(threshold_spikes.sum())
            spike_pattern = threshold_spikes

        return total_spikes, spike_pattern
