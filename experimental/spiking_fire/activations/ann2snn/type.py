from dataclasses import dataclass

@dataclass
class ANN2SNNConfig:
    """Configuration for ANN to SNN conversion."""
    init_scale: float = 1.0  # Initial scaling factor for ANN outputs
    max_rate: float = 100.0  # Maximum firing rate in Hz
    time_step: float = 1e-3   # Time step for simulation in seconds
    threshold: float = 1.0    # Neuron firing threshold
    scale_factor: float = 1.0  # Scaling factor for ANN outputs to spike rates
    momentum: float = 0.9     # Momentum for rate coding updates
    mode: str = "rate"  # Coding mode TBD
