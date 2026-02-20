from abc import ABC, abstractmethod
from collections.abc import Callable

import nn
from build.lib.grilly.backend import snn


class BaseNode(ABC, nn.Module):
    """Abstract base class for nodes in the spiking fire framework."""

    def __init__(self, voltage_threshold: float =1.0, reset_voltage: float | None=0.0,
                     surrogate_function: Callable = snn.surrogate.ATan()):
            super().__init__()
            self.voltage_threshold = voltage_threshold
            self.reset_voltage = reset_voltage
            self.surrogate_function = surrogate_function


    @abstractmethod
    def forward(self, x):
        """Process input data and return output."""
        pass
