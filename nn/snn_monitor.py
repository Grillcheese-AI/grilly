"""
SNN Monitoring and Recording utilities.

Monitor: Records neuron state variables over time for visualization.
"""

import numpy as np


class Monitor:
    """Records neuron state variables over time.

    Attaches to a MemoryModule and records specified variables
    at each timestep for analysis and visualization.

    Args:
        module: MemoryModule to monitor
        var_names: List of variable names to record (default: ['v', 'spike'])

    Example:
        >>> lif = LIFNode(tau=2.0)
        >>> mon = Monitor(lif, var_names=['v'])
        >>> for t in range(100):
        ...     spike = lif(input_current[t])
        ...     mon.record()
        >>> v_trace = mon.get('v')  # (100, *neuron_shape)
    """

    def __init__(self, module, var_names=None):
        self.module = module
        self.var_names = var_names or ["v"]
        self._records = {name: [] for name in self.var_names}
        self._enabled = True

    def record(self):
        """Record current values of monitored variables.

        Call after each forward pass to capture the state.
        """
        if not self._enabled:
            return

        for name in self.var_names:
            val = getattr(self.module, name, None)
            if val is not None:
                if isinstance(val, np.ndarray):
                    self._records[name].append(val.copy())
                else:
                    self._records[name].append(np.array(val, dtype=np.float32))

    def get(self, var_name):
        """Get recorded values for a variable.

        Args:
            var_name: Name of the variable

        Returns:
            np.ndarray of shape (T, *var_shape) or empty array if no recordings
        """
        if var_name not in self._records:
            raise KeyError(f"Variable '{var_name}' not monitored. Available: {self.var_names}")

        records = self._records[var_name]
        if not records:
            return np.array([], dtype=np.float32)

        return np.stack(records, axis=0)

    def reset(self):
        """Clear all recordings."""
        self._records = {name: [] for name in self.var_names}

    def enable(self):
        """Enable recording."""
        self._enabled = True

    def disable(self):
        """Disable recording (skip record() calls)."""
        self._enabled = False

    @property
    def num_recordings(self):
        """Number of timesteps recorded."""
        for records in self._records.values():
            return len(records)
        return 0

    def __repr__(self):
        return (
            f"Monitor(module={self.module.__class__.__name__}, "
            f"var_names={self.var_names}, "
            f"recordings={self.num_recordings})"
        )


def plot_membrane_potential(monitor, neuron_indices=None, var_name="v"):
    """Plot membrane potential traces (requires matplotlib).

    Args:
        monitor: Monitor instance with recorded 'v' data
        neuron_indices: Which neurons to plot (default: first 5)
        var_name: Variable name to plot (default: 'v')

    Returns:
        matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")

    data = monitor.get(var_name)
    if data.size == 0:
        raise ValueError("No recordings to plot")

    T = data.shape[0]
    flat = data.reshape(T, -1)

    if neuron_indices is None:
        neuron_indices = list(range(min(5, flat.shape[1])))

    fig, ax = plt.subplots(figsize=(10, 4))
    for idx in neuron_indices:
        ax.plot(flat[:, idx], label=f"Neuron {idx}")

    ax.set_xlabel("Timestep")
    ax.set_ylabel(var_name)
    ax.set_title(f"{var_name} over time")
    ax.legend()
    return fig


def plot_spike_raster(monitor, neuron_indices=None, var_name="spike"):
    """Plot spike raster (requires matplotlib).

    Args:
        monitor: Monitor instance with recorded spike data
        neuron_indices: Which neurons to plot (default: first 20)
        var_name: Variable name containing spike data

    Returns:
        matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")

    data = monitor.get(var_name)
    if data.size == 0:
        raise ValueError("No recordings to plot")

    T = data.shape[0]
    flat = data.reshape(T, -1)

    if neuron_indices is None:
        neuron_indices = list(range(min(20, flat.shape[1])))

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, idx in enumerate(neuron_indices):
        spike_times = np.where(flat[:, idx] > 0.5)[0]
        ax.scatter(spike_times, [i] * len(spike_times), s=2, c="black")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Neuron index")
    ax.set_title("Spike Raster")
    ax.set_yticks(range(len(neuron_indices)))
    ax.set_yticklabels(neuron_indices)
    return fig
