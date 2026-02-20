"""
Base classes for Spiking Neural Network modules.

MemoryModule: Stateful module with register_memory/reset/detach support.
BaseNode: Abstract spiking neuron with charge-fire-reset dynamics.
             Supports GPU acceleration via snn-node-forward/backward shaders.
"""

import numpy as np

from .module import Module
from .snn_surrogate import ATan


def _get_snn_gpu():
    """Get shared Vulkan SNN backend (cached at module level)."""
    try:
        from .snn_synapses import _get_gpu_compute

        compute = _get_gpu_compute()
        if compute is not None:
            return compute.snn
    except Exception:
        pass
    return None


class MemoryModule(Module):
    """Stateful module that tracks internal memory variables.

    Memory variables (like membrane potential) persist across timesteps
    but can be reset between sequences. Subclasses register memory
    via register_memory().
    """

    def __init__(self):
        super().__init__()
        self._memories = {}

    def register_memory(self, name, default_value):
        """Register a memory variable with its default (reset) value.

        Args:
            name: Name of the memory variable (becomes self.name)
            default_value: Value to reset to (scalar or None for lazy init)
        """
        self._memories[name] = default_value
        setattr(self, name, default_value)

    def reset(self):
        """Reset all memory variables to defaults. Call between sequences."""
        for name, default in self._memories.items():
            if default is None:
                setattr(self, name, None)
            elif isinstance(default, np.ndarray):
                setattr(self, name, default.copy())
            else:
                setattr(self, name, default)
        # Recursively reset child modules
        for module in self._modules.values():
            if isinstance(module, MemoryModule):
                module.reset()

    def detach(self):
        """Detach memory from computation graph (stop gradients through time)."""
        for name in self._memories:
            val = getattr(self, name)
            if isinstance(val, np.ndarray):
                setattr(self, name, val.copy())


class BaseNode(MemoryModule):
    """Abstract spiking neuron base class.

    Implements the charge-fire-reset dynamics common to all
    spiking neuron models. Subclasses implement neuronal_charge().

    Args:
        v_threshold: Spike threshold voltage (default: 1.0)
        v_reset: Reset voltage after spike (default: 0.0, None for soft reset)
        surrogate_function: Surrogate gradient function for backward pass
        detach_reset: If True, detach reset from gradient computation
        step_mode: 's' for single-step, 'm' for multi-step (temporal)

    Shape:
        single step: (N, *) -> (N, *)
        multi step:  (T, N, *) -> (T, N, *)
    """

    def __init__(
        self,
        v_threshold=1.0,
        v_reset=0.0,
        surrogate_function=None,
        detach_reset=False,
        step_mode="s",
        use_gpu=False,
    ):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function or ATan()
        self.detach_reset = detach_reset
        self.step_mode = step_mode
        self.use_gpu = use_gpu
        self._gpu_snn = None

        # Membrane potential — lazily initialized from input shape
        self.register_memory("v", None)
        # Cached pre-spike membrane (for surrogate gradient backward)
        self._h_cache = None
        # Per-timestep caches for multi-step BPTT backward
        self._h_seq_cache = []
        self._spike_seq_cache = []

    def _init_v(self, x):
        """Initialize membrane potential to match input shape."""
        if self.v_reset is not None:
            self.v = np.full(x.shape, self.v_reset, dtype=np.float32)
        else:
            self.v = np.zeros(x.shape, dtype=np.float32)

    def neuronal_charge(self, x):
        """Charge the neuron (neuron-type specific). Must set self.h.

        Args:
            x: Input tensor (N, *) or per-element current
        """
        raise NotImplementedError

    def neuronal_fire(self):
        """Generate spikes using surrogate function.

        Returns:
            Binary spike tensor (same shape as self.h)
        """
        return self.surrogate_function(self.h - self.v_threshold)

    def neuronal_reset(self, spike):
        """Reset membrane potential based on spikes.

        Hard reset: V = H * (1 - S) + v_reset * S
        Soft reset: V = H - v_threshold * S
        """
        if self.v_reset is not None:
            # Hard reset
            if self.detach_reset:
                spike_d = spike.copy()
            else:
                spike_d = spike
            self.v = self.h * (1.0 - spike_d) + self.v_reset * spike_d
        else:
            # Soft reset
            if self.detach_reset:
                spike_d = spike.copy()
            else:
                spike_d = spike
            self.v = self.h - self.v_threshold * spike_d

    def _try_gpu(self):
        """Lazy-init GPU backend."""
        if self._gpu_snn is None and self.use_gpu:
            self._gpu_snn = _get_snn_gpu()
        return self._gpu_snn

    def _gpu_neuron_type(self):
        """Return GPU shader neuron type. Override in subclasses.

        0=IF, 1=LIF, 2=PLIF
        """
        return 0

    def _gpu_params(self):
        """Return (tau, decay_input, tau_param) for GPU shader.

        Override in subclasses. tau_param is per-neuron tau array for PLIF.
        """
        return 2.0, False, None

    def single_step_forward(self, x):
        """One timestep: charge -> fire -> reset.

        Uses GPU shader when use_gpu=True and backend available.

        Args:
            x: Input (N, *)

        Returns:
            spike: Binary spike output (N, *)
        """
        if self.v is None:
            self._init_v(x)

        gpu = self._try_gpu()
        if gpu is not None:
            try:
                return self._gpu_single_step(gpu, x)
            except Exception:
                pass

        # CPU path
        self.neuronal_charge(x)
        self._h_cache = self.h.copy()
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def _gpu_single_step(self, gpu, x):
        """Run charge-fire-reset on GPU via snn-node-forward shader."""
        tau, decay_input, tau_param = self._gpu_params()
        reset_mode = 0 if self.v_reset is not None else 1
        v_reset = self.v_reset if self.v_reset is not None else 0.0

        spikes, v_out, h_out = gpu.snn_node_forward(
            x, self.v,
            neuron_type=self._gpu_neuron_type(),
            tau=tau,
            v_threshold=self.v_threshold,
            v_reset=v_reset,
            reset_mode=reset_mode,
            decay_input=decay_input,
            tau_param=tau_param,
        )
        self.v = v_out
        self.h = h_out
        self._h_cache = h_out.copy()
        return spikes

    def multi_step_forward(self, x_seq):
        """Process temporal sequence: loop over T timesteps.

        Args:
            x_seq: Input (T, N, *)

        Returns:
            spike_seq: Spikes over time (T, N, *)
        """
        T = x_seq.shape[0]
        spikes = []
        self._h_seq_cache = []
        self._spike_seq_cache = []
        for t in range(T):
            spike = self.single_step_forward(x_seq[t])
            self._h_seq_cache.append(self._h_cache.copy())
            self._spike_seq_cache.append(spike.copy())
            spikes.append(spike)
        return np.stack(spikes, axis=0)

    def forward(self, x):
        """Dispatch to single or multi step based on step_mode."""
        if self.step_mode == "s":
            return self.single_step_forward(x)
        elif self.step_mode == "m":
            return self.multi_step_forward(x)
        else:
            raise ValueError(f"Invalid step_mode '{self.step_mode}', expected 's' or 'm'")

    def _surrogate_type_id(self):
        """Map surrogate function to GPU shader type ID."""
        from .snn_surrogate import FastSigmoid, Sigmoid

        if isinstance(self.surrogate_function, Sigmoid):
            return 1
        elif isinstance(self.surrogate_function, FastSigmoid):
            return 2
        return 0  # ATan (default)

    def _dh_dv_prev(self):
        """Return dH[t]/dV[t-1] for BPTT. Override in subclasses.

        For IF: dH/dV_prev = 1.0
        For LIF: dH/dV_prev = 1 - 1/tau
        """
        return 1.0

    def _dh_dx(self):
        """Return dH[t]/dX[t] for BPTT. Override in subclasses.

        For IF: dH/dX = 1.0
        For LIF: dH/dX = 1/tau
        """
        return 1.0

    def backward(self, grad_output, x=None):
        """Backward pass using BPTT with surrogate gradients.

        Implements proper Backpropagation Through Time:
        - Gradients flow backward through time via membrane potential
        - dL/dH[t] = dL/dS[t]*sg(H-Vth) + dL/dV[t]*(1-S[t])
        - dL/dX[t] = dL/dH[t] * dH/dX
        - dL/dV[t-1] = dL/dH[t] * dH/dV_prev

        Args:
            grad_output: Gradient w.r.t. spike output
            x: Input from forward pass (unused, kept for API compat)

        Returns:
            Gradient w.r.t. input
        """
        if self.step_mode == "s":
            if self._h_cache is None:
                return grad_output

            gpu = self._try_gpu()
            if gpu is not None:
                try:
                    sg_type = self._surrogate_type_id()
                    grad_x = gpu.snn_node_backward(
                        grad_output, self._h_cache,
                        alpha=self.surrogate_function.alpha,
                        surrogate_type=sg_type,
                        v_threshold=self.v_threshold,
                    )
                    return (grad_x * self._dh_dx()).astype(np.float32)
                except Exception:
                    pass

            sg = self.surrogate_function.gradient(self._h_cache - self.v_threshold)
            return (grad_output * sg * self._dh_dx()).astype(np.float32)
        else:
            # Multi-step BPTT: grad_output is (T, N, *)
            if not self._h_seq_cache:
                return grad_output
            T = grad_output.shape[0]
            dh_dv = self._dh_dv_prev()
            dh_dx = self._dh_dx()
            grad_input = []

            # Initialize future gradient on V to zero
            dL_dV = np.zeros_like(grad_output[0])

            # Backward through time
            for t in range(T - 1, -1, -1):
                h_t = self._h_seq_cache[t] if t < len(self._h_seq_cache) else 0.0
                s_t = self._spike_seq_cache[t] if t < len(self._spike_seq_cache) else 0.0
                sg = self.surrogate_function.gradient(h_t - self.v_threshold)

                # dL/dH[t] = dL/dS[t] * sg + dL/dV[t] * (1 - S[t])
                dL_dH = grad_output[t] * sg + dL_dV * (1.0 - s_t)

                # dL/dX[t] = dL/dH[t] * dH/dX
                grad_input.append(dL_dH * dh_dx)

                # dL/dV[t-1] = dL/dH[t] * dH/dV_prev
                dL_dV = dL_dH * dh_dv

            # Reverse to get chronological order
            grad_input.reverse()
            return np.stack(grad_input, axis=0).astype(np.float32)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"v_threshold={self.v_threshold}, "
            f"v_reset={self.v_reset}, "
            f"surrogate_function={self.surrogate_function}, "
            f"step_mode='{self.step_mode}')"
        )
