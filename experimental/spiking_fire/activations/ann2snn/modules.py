import grilly.nn as nn
import grilly.snn as snn
import numpy as np


class VoltageHook(nn.Module, snn.SpikingModule):
    """Hook to capture voltage dynamics during SNN simulation."""

    def __init__(self):
        super().__init__()
        self.voltages = []
        self.buffer_size = 1000  # Limit buffer to last 1000 time steps


    def forward(self, x):
        self.voltages.append(x.copy())
        if len(self.voltages) > self.buffer_size:
            self.voltages.pop(0)
        return x


class ANN2SNN(nn.Module):
    """Convert an ANN to an SNN using rate coding."""

    def __init__(self, ann_model, n_steps=100):
        super().__init__()
        self.ann_model = ann_model
        self.n_steps = n_steps

    def forward(self, x):

        err_msg = "Input must be a 2D tensor of shape (batch_size, n_features)"
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            raise ValueError(err_msg)

        # Get ANN output
        ann_output = self.ann_model(x)

        # Rate coding: convert ANN output to spike counts
        spike_counts = np.round(ann_output * self.n_steps).astype(int)

        # Generate spike trains
        batch_size, n_neurons = spike_counts.shape
        spike_trains = np.zeros((batch_size, n_neurons, self.n_steps), dtype=np.float32)
        for i in range(batch_size):
            for j in range(n_neurons):
                count = spike_counts[i, j]
                if count > 0:
                    spike_times = np.random.choice(self.n_steps, size=count, replace=False)
                    spike_trains[i, j, spike_times] = 1.0

        return spike_trains
