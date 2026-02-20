"""
Spiking Neuron Node implementations.

IFNode: Integrate-and-Fire (simplest, no leak)
LIFNode: Leaky Integrate-and-Fire (standard SNN neuron)
ParametricLIFNode: LIF with learnable time constant tau
"""

import numpy as np

from .parameter import Parameter
from .snn_base import BaseNode


class IFNode(BaseNode):
    """Integrate-and-Fire neuron node.

    Charge: H[t] = V[t-1] + X[t]
    No leak — membrane accumulates until threshold.

    Args:
        v_threshold: Spike threshold (default: 1.0)
        v_reset: Reset voltage (default: 0.0, None for soft reset)
        surrogate_function: Surrogate gradient function
        detach_reset: Detach reset from grad computation
        step_mode: 's' (single) or 'm' (multi-step)
    """

    def neuronal_charge(self, x):
        self.h = self.v + x

    def _gpu_neuron_type(self):
        return 0  # IF

    def _gpu_params(self):
        return 2.0, False, None


class LIFNode(BaseNode):
    """Leaky Integrate-and-Fire neuron node.

    With decay_input=True (physics-accurate):
        H[t] = V[t-1] * (1 - 1/tau) + X[t] / tau
        (Both voltage and input are scaled by tau)

    With decay_input=False (default, practical for deep SNNs):
        H[t] = V[t-1] * (1 - 1/tau) + X[t]
        (Voltage leaks but input is added at full strength)

    Args:
        tau: Membrane time constant (default: 2.0). Higher = slower leak.
        decay_input: If True, divide input by tau (physics-accurate but
            neurons fire less). If False (default), add input at full
            strength — much better for deep SNN training.
        v_threshold: Spike threshold (default: 1.0)
        v_reset: Reset voltage (default: 0.0, None for soft reset)
        surrogate_function: Surrogate gradient function
        detach_reset: Detach reset from grad computation
        step_mode: 's' (single) or 'm' (multi-step)
    """

    def __init__(self, tau=2.0, decay_input=False, **kwargs):
        super().__init__(**kwargs)
        if tau < 1.0:
            raise ValueError(f"tau must be >= 1.0, got {tau}")
        self.tau = tau
        self.decay_input = decay_input

    def neuronal_charge(self, x):
        decay = 1.0 - 1.0 / self.tau
        if self.decay_input:
            # Physics-accurate: H = V*(1-1/tau) + X/tau
            if self.v_reset is not None:
                self.h = self.v + (x - (self.v - self.v_reset)) / self.tau
            else:
                self.h = self.v + (x - self.v) / self.tau
        else:
            # Practical: H = V*(1-1/tau) + X (input at full strength)
            if self.v_reset is not None:
                self.h = decay * (self.v - self.v_reset) + self.v_reset + x
            else:
                self.h = decay * self.v + x

    def _gpu_neuron_type(self):
        return 1  # LIF

    def _gpu_params(self):
        return self.tau, self.decay_input, None

    def _dh_dv_prev(self):
        """LIF: dH/dV_prev = 1 - 1/tau (leak factor)."""
        return 1.0 - 1.0 / self.tau

    def _dh_dx(self):
        """LIF: dH/dX = 1/tau if decay_input else 1.0."""
        if self.decay_input:
            return 1.0 / self.tau
        return 1.0

    def __repr__(self):
        return (
            f"LIFNode(tau={self.tau}, decay_input={self.decay_input}, "
            f"v_threshold={self.v_threshold}, "
            f"v_reset={self.v_reset}, "
            f"surrogate_function={self.surrogate_function}, "
            f"step_mode='{self.step_mode}')"
        )


class ParametricLIFNode(BaseNode):
    """Parametric Leaky Integrate-and-Fire neuron node.

    Same as LIFNode but tau is a learnable Parameter.

    Args:
        init_tau: Initial time constant (default: 2.0)
        decay_input: If True, divide input by tau. If False (default),
            add input at full strength.
        v_threshold: Spike threshold (default: 1.0)
        v_reset: Reset voltage (default: 0.0)
        surrogate_function: Surrogate gradient function
        detach_reset: Detach reset from grad computation
        step_mode: 's' (single) or 'm' (multi-step)
    """

    def __init__(self, init_tau=2.0, decay_input=False, **kwargs):
        super().__init__(**kwargs)
        self.decay_input = decay_input
        self.tau = Parameter(
            np.array([init_tau], dtype=np.float32), requires_grad=True
        )
        self.register_parameter("tau", self.tau)

    def _get_tau(self):
        """Get clamped tau value (>= 1.0)."""
        return np.maximum(np.asarray(self.tau, dtype=np.float32), 1.0)

    def neuronal_charge(self, x):
        tau = self._get_tau()
        decay = 1.0 - 1.0 / tau
        if self.decay_input:
            if self.v_reset is not None:
                self.h = self.v + (x - (self.v - self.v_reset)) / tau
            else:
                self.h = self.v + (x - self.v) / tau
        else:
            if self.v_reset is not None:
                self.h = decay * (self.v - self.v_reset) + self.v_reset + x
            else:
                self.h = decay * self.v + x

    def _gpu_neuron_type(self):
        return 2  # PLIF

    def _gpu_params(self):
        tau = self._get_tau()
        tau_val = float(np.asarray(tau).item())
        # For PLIF, pass per-neuron tau array (broadcast to element count by shader)
        return tau_val, self.decay_input, np.asarray(tau, dtype=np.float32).flatten()

    def _dh_dv_prev(self):
        """PLIF: dH/dV_prev = 1 - 1/tau."""
        tau = self._get_tau()
        return 1.0 - 1.0 / tau

    def _dh_dx(self):
        """PLIF: dH/dX = 1/tau if decay_input else 1.0."""
        if self.decay_input:
            return 1.0 / self._get_tau()
        return 1.0

    def __repr__(self):
        tau_val = float(np.asarray(self.tau).item())
        return (
            f"ParametricLIFNode(tau={tau_val:.2f}, decay_input={self.decay_input}, "
            f"v_threshold={self.v_threshold}, "
            f"v_reset={self.v_reset}, "
            f"surrogate_function={self.surrogate_function}, "
            f"step_mode='{self.step_mode}')"
        )
