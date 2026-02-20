"""
SNN Synapse modules for temporal filtering and recurrent connections.

ElementWiseRecurrentContainer: Feeds output back as input (e.g., recurrent SNN).
SynapseFilter: Exponential decay temporal smoothing (GPU-accelerated).
DualTimescaleSynapse: Fast + slow synaptic dynamics (AMPA + NMDA-like).
STDPSynapse: Spike-timing dependent plasticity with eligibility traces.
"""

import numpy as np

from .snn_base import MemoryModule

_GPU_COMPUTE = None


def _get_gpu_compute():
    """Get or create the shared VulkanCompute instance.

    Returns the full VulkanCompute so callers can access .snn, .fnn, etc.
    Caches at module level — Vulkan devices can only be initialized once.
    """
    global _GPU_COMPUTE
    if _GPU_COMPUTE is not None:
        return _GPU_COMPUTE
    try:
        from grilly.backend.compute import VulkanCompute

        _GPU_COMPUTE = VulkanCompute()
        return _GPU_COMPUTE
    except Exception:
        return None


def set_gpu_compute(compute):
    """Set the shared VulkanCompute instance (for use by test fixtures etc.)."""
    global _GPU_COMPUTE
    _GPU_COMPUTE = compute


class ElementWiseRecurrentContainer(MemoryModule):
    """Element-wise recurrent container for spiking neurons.

    Feeds the previous timestep's output back and combines it with
    the current input: i[t] = f(x[t], y[t-1]).

    By default f is addition. Wrap a spiking neuron (e.g., IFNode)
    as the sub_module.

    Args:
        sub_module: The spiking neuron module to wrap
        element_wise_function: Callable(x, y_prev) -> combined input
            Defaults to addition: lambda x, y: x + y
    """

    def __init__(self, sub_module, element_wise_function=None):
        super().__init__()
        self._modules["sub_module"] = sub_module
        self.sub_module = sub_module
        self.element_wise_function = element_wise_function or (lambda x, y: x + y)
        self.register_memory("y_prev", None)

    def forward(self, x):
        """Single step with recurrent feedback.

        Args:
            x: Input (N, *)

        Returns:
            Output spikes (N, *)
        """
        if self.y_prev is None:
            self.y_prev = np.zeros_like(x, dtype=np.float32)

        combined = self.element_wise_function(x, self.y_prev)
        y = self.sub_module(combined)
        self.y_prev = y.copy()
        return y

    def __repr__(self):
        return f"ElementWiseRecurrentContainer(sub_module={self.sub_module})"


class SynapseFilter(MemoryModule):
    """Exponential decay synaptic filter with GPU acceleration.

    Smooths input temporally: y[t] = y[t-1] * exp(-1/tau) + x[t]

    Placed after Conv/Linear layers to add temporal dynamics
    before spiking neurons. Uses the snn-synapse-filter compute
    shader on GPU when available, falls back to numpy on CPU.

    Args:
        tau: Time constant (default: 2.0). Higher = slower decay.
        learnable: If True, tau is a learnable parameter (default: False)
        use_gpu: If True, attempt GPU dispatch (default: True)
    """

    def __init__(self, tau=2.0, learnable=False, use_gpu=True):
        super().__init__()
        self.learnable = learnable
        self.use_gpu = use_gpu
        self._gpu_snn = None

        if learnable:
            from .parameter import Parameter

            self.tau = Parameter(
                np.array([tau], dtype=np.float32), requires_grad=True
            )
            self.register_parameter("tau", self.tau)
        else:
            self.tau = tau

        self.register_memory("y", None)
        self._x_cache = None  # For backward pass

    def _get_decay(self):
        """Compute decay factor exp(-1/tau)."""
        tau_val = float(np.asarray(self.tau).item()) if self.learnable else self.tau
        tau_val = max(tau_val, 0.1)
        return float(np.exp(-1.0 / tau_val))

    def _try_gpu(self):
        """Lazy-init GPU backend."""
        if self._gpu_snn is None and self.use_gpu:
            compute = _get_gpu_compute()
            if compute is not None:
                self._gpu_snn = compute.snn
        return self._gpu_snn

    def forward(self, x):
        """Apply exponential filter to input.

        Args:
            x: Input (N, *)

        Returns:
            Filtered output (N, *)
        """
        if self.y is None:
            self.y = np.zeros_like(x, dtype=np.float32)

        decay = self._get_decay()
        self._x_cache = x.copy()

        gpu = self._try_gpu()
        if gpu is not None:
            try:
                self.y = gpu.synapse_filter(x, self.y, decay)
                return self.y.copy()
            except Exception:
                pass

        # CPU fallback
        self.y = self.y * decay + x
        return self.y.copy()

    def backward(self, grad_output, x=None):
        """Backward pass through synapse filter.

        The filter y[t] = decay * y[t-1] + x[t] has gradient:
            dL/dx[t] = dL/dy[t]  (direct path)

        For multi-step, gradients also flow through the decay chain:
            dL/dx[t-1] += dL/dy[t] * decay

        For single-step backward, grad passes straight through.
        """
        return grad_output.astype(np.float32)

    def __repr__(self):
        tau_val = float(np.asarray(self.tau).item()) if self.learnable else self.tau
        gpu_str = "gpu" if self._gpu_snn is not None else "cpu"
        return f"SynapseFilter(tau={tau_val:.2f}, learnable={self.learnable}, {gpu_str})"


class DualTimescaleSynapse(MemoryModule):
    """Dual-timescale synaptic filter (fast + slow components).

    Models biological AMPA (fast, ~1-5ms) and NMDA (slow, ~50-150ms)
    receptor dynamics:

        y_fast[t] = y_fast[t-1] * decay_fast + x[t]
        y_slow[t] = y_slow[t-1] * decay_slow + x[t]
        output[t] = w_fast * y_fast[t] + w_slow * y_slow[t]

    This gives neurons access to both fast transient signals and
    slow integrated context, improving temporal processing.

    Args:
        tau_fast: Fast component time constant (default: 2.0)
        tau_slow: Slow component time constant (default: 20.0)
        w_fast: Weight for fast component (default: 0.7)
        w_slow: Weight for slow component (default: 0.3)
        learnable: If True, make tau and weights learnable (default: False)
        use_gpu: If True, attempt GPU dispatch (default: True)
    """

    def __init__(self, tau_fast=2.0, tau_slow=20.0, w_fast=0.7, w_slow=0.3,
                 learnable=False, use_gpu=True):
        super().__init__()
        self.learnable = learnable
        self.use_gpu = use_gpu
        self._gpu_snn = None

        if learnable:
            from .parameter import Parameter

            self.tau_fast = Parameter(
                np.array([tau_fast], dtype=np.float32), requires_grad=True
            )
            self.tau_slow = Parameter(
                np.array([tau_slow], dtype=np.float32), requires_grad=True
            )
            self.w_fast = Parameter(
                np.array([w_fast], dtype=np.float32), requires_grad=True
            )
            self.w_slow = Parameter(
                np.array([w_slow], dtype=np.float32), requires_grad=True
            )
            self.register_parameter("tau_fast", self.tau_fast)
            self.register_parameter("tau_slow", self.tau_slow)
            self.register_parameter("w_fast", self.w_fast)
            self.register_parameter("w_slow", self.w_slow)
        else:
            self.tau_fast = tau_fast
            self.tau_slow = tau_slow
            self.w_fast = w_fast
            self.w_slow = w_slow

        self.register_memory("y_fast", None)
        self.register_memory("y_slow", None)

    def _get_decay(self, tau):
        """Compute decay factor from tau."""
        tau_val = float(np.asarray(tau).item()) if self.learnable else float(tau)
        tau_val = max(tau_val, 0.1)
        return float(np.exp(-1.0 / tau_val))

    def _get_weight(self, w):
        """Get weight value."""
        return float(np.asarray(w).item()) if self.learnable else float(w)

    def _try_gpu(self):
        """Lazy-init GPU backend."""
        if self._gpu_snn is None and self.use_gpu:
            compute = _get_gpu_compute()
            if compute is not None:
                self._gpu_snn = compute.snn
        return self._gpu_snn

    def forward(self, x):
        """Apply dual-timescale filtering.

        Args:
            x: Input (N, *)

        Returns:
            Weighted sum of fast and slow filtered output (N, *)
        """
        if self.y_fast is None:
            self.y_fast = np.zeros_like(x, dtype=np.float32)
        if self.y_slow is None:
            self.y_slow = np.zeros_like(x, dtype=np.float32)

        decay_fast = self._get_decay(self.tau_fast)
        decay_slow = self._get_decay(self.tau_slow)
        wf = self._get_weight(self.w_fast)
        ws = self._get_weight(self.w_slow)

        gpu = self._try_gpu()
        if gpu is not None:
            try:
                # Two GPU dispatches — fast and slow filters in sequence
                self.y_fast = gpu.synapse_filter(x, self.y_fast, decay_fast)
                self.y_slow = gpu.synapse_filter(x, self.y_slow, decay_slow)
                return (wf * self.y_fast + ws * self.y_slow).astype(np.float32)
            except Exception:
                pass

        # CPU fallback
        self.y_fast = self.y_fast * decay_fast + x
        self.y_slow = self.y_slow * decay_slow + x
        return (wf * self.y_fast + ws * self.y_slow).astype(np.float32)

    def backward(self, grad_output, x=None):
        """Backward pass — gradient flows through both pathways."""
        wf = self._get_weight(self.w_fast)
        ws = self._get_weight(self.w_slow)
        return (grad_output * (wf + ws)).astype(np.float32)

    def __repr__(self):
        tf = float(np.asarray(self.tau_fast).item()) if self.learnable else self.tau_fast
        ts = float(np.asarray(self.tau_slow).item()) if self.learnable else self.tau_slow
        wf = float(np.asarray(self.w_fast).item()) if self.learnable else self.w_fast
        ws = float(np.asarray(self.w_slow).item()) if self.learnable else self.w_slow
        return (
            f"DualTimescaleSynapse(tau_fast={tf:.1f}, tau_slow={ts:.1f}, "
            f"w_fast={wf:.2f}, w_slow={ws:.2f}, learnable={self.learnable})"
        )


class STPSynapse(MemoryModule):
    """Short-Term Plasticity synapse (facilitation + depression).

    Models use-dependent synaptic dynamics:
        u[t] = u[t-1] * (1 - dt/tau_f) + U * (1 - u[t-1]) * spike[t]
        r[t] = r[t-1] * (1 - dt/tau_d) + (1 - r[t-1]) * u[t-1] * spike[t]  (missing dt)
        output = x * u * r

    u = utilization (facilitation): how much of available resources are released
    r = recovery (depression): fraction of resources available

    Args:
        U: Base release probability (default: 0.2)
        tau_f: Facilitation time constant (default: 20.0)
        tau_d: Depression time constant (default: 200.0)
    """

    def __init__(self, U=0.2, tau_f=20.0, tau_d=200.0):
        super().__init__()
        self.U = U
        self.tau_f = tau_f
        self.tau_d = tau_d
        self.register_memory("u", None)  # Facilitation variable
        self.register_memory("r", None)  # Depression variable (recovery)

    def forward(self, x, spike=None):
        """Apply short-term plasticity.

        Args:
            x: Input signal (N, *)
            spike: Binary spike indicators. If None, treats any |x| > 0 as spike.

        Returns:
            Modulated output (N, *)
        """
        if self.u is None:
            self.u = np.full_like(x, self.U, dtype=np.float32)
        if self.r is None:
            self.r = np.ones_like(x, dtype=np.float32)

        if spike is None:
            spike = (np.abs(x) > 0).astype(np.float32)

        # Facilitation: u increases with each spike, decays between
        du = -(self.u - self.U) / self.tau_f + self.U * (1.0 - self.u) * spike
        self.u = np.clip(self.u + du, 0.0, 1.0).astype(np.float32)

        # Depression: r decreases with each spike (resources used), recovers between
        dr = (1.0 - self.r) / self.tau_d - self.u * self.r * spike
        self.r = np.clip(self.r + dr, 0.0, 1.0).astype(np.float32)

        return (x * self.u * self.r).astype(np.float32)

    def backward(self, grad_output, x=None):
        """Backward pass — approximate gradient (STP is not differentiable w.r.t. spikes)."""
        if self.u is not None and self.r is not None:
            return (grad_output * self.u * self.r).astype(np.float32)
        return grad_output.astype(np.float32)

    def __repr__(self):
        return f"STPSynapse(U={self.U}, tau_f={self.tau_f:.1f}, tau_d={self.tau_d:.1f})"
