"""
Spiking Neural Network Modules
Uses: lif-neuron.glsl, hebbian-learning.glsl, stdp-learning.glsl, gif-neuron.glsl
"""

import numpy as np

from .module import Module


class LIFNeuron(Module):
    """
    Leaky Integrate-and-Fire neuron
    Uses: lif-neuron.glsl
    """

    def __init__(
        self, n_neurons: int, dt: float = 0.001, tau_mem: float = 20.0, v_thresh: float = 1.0
    ):
        """Initialize the instance."""

        super().__init__()
        self.n_neurons = n_neurons
        self.dt = dt
        self.tau_mem = tau_mem
        self.v_thresh = v_thresh

        # Neuron state
        self.membrane = np.zeros(n_neurons, dtype=np.float32)
        self.refractory = np.zeros(n_neurons, dtype=np.float32)
        self._buffers["membrane"] = self.membrane
        self._buffers["refractory"] = self.refractory

    def forward(self, input_current: np.ndarray) -> np.ndarray:
        """Forward pass using lif-neuron.glsl"""
        backend = self._get_backend()
        self.membrane, self.refractory, spikes = backend.lif_step(
            input_current,
            self.membrane,
            self.refractory,
            dt=self.dt,
            tau_mem=self.tau_mem,
            v_thresh=self.v_thresh,
        )
        return spikes

    def reset(self):
        """Reset neuron state"""
        self.membrane.fill(0)
        self.refractory.fill(0)

    def __repr__(self):
        """Return a debug representation."""

        return f"LIFNeuron(n_neurons={self.n_neurons}, dt={self.dt}, tau_mem={self.tau_mem}, v_thresh={self.v_thresh})"


class SNNLayer(Module):
    """
    Spiking Neural Network layer with LIF neurons
    Uses: lif-neuron.glsl, snn-matmul.glsl, snn-softmax.glsl, snn-rmsnorm.glsl
    """

    def __init__(self, in_features: int, out_features: int, n_neurons: int = None):
        """Initialize the instance."""

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_neurons = n_neurons or out_features

        # Weight matrix
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = np.random.uniform(-limit, limit, (out_features, in_features)).astype(
            np.float32
        )
        self._parameters["weight"] = self.weight

        # LIF neurons
        self.neurons = LIFNeuron(self.n_neurons)
        self._modules["neurons"] = self.neurons

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self._get_backend()

        # Linear transformation (snn-matmul.glsl)
        # Note: May need to implement snn_matmul in backend
        output = x @ self.weight.T

        # Pass through LIF neurons
        spikes = self.neurons(output)
        return spikes

    def __repr__(self):
        """Return a debug representation."""

        return f"SNNLayer(in_features={self.in_features}, out_features={self.out_features}, n_neurons={self.n_neurons})"


class HebbianLayer(Module):
    """
    Hebbian learning layer
    Uses: hebbian-learning.glsl
    """

    def __init__(self, in_features: int, out_features: int):
        """Initialize the instance."""

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights
        self.weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.1
        self._parameters["weight"] = self.weight

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        return x @ self.weight.T

    def update_weights(self, pre: np.ndarray, post: np.ndarray, learning_rate: float = 0.01):
        """Update weights using Hebbian learning (hebbian-learning.glsl)"""
        backend = self._get_backend()
        self.weight = backend.hebbian_learning(self.weight, pre, post, learning_rate=learning_rate)

    def __repr__(self):
        """Return a debug representation."""

        return f"HebbianLayer(in_features={self.in_features}, out_features={self.out_features})"


class STDPLayer(Module):
    """
    Spike-Timing-Dependent Plasticity layer
    Uses: stdp-learning.glsl, synapsis-stdp-trace.glsl, synapsis-stdp-update.glsl
    """

    def __init__(self, in_features: int, out_features: int):
        """Initialize the instance."""

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights
        self.weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.1
        self._parameters["weight"] = self.weight

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        return x @ self.weight.T

    def update_weights(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        a_plus: float = 0.01,
        a_minus: float = 0.01,
    ):
        """Update weights using STDP (stdp-learning.glsl)"""
        backend = self._get_backend()
        self.weight = backend.stdp_learning(
            self.weight,
            pre_spikes,
            post_spikes,
            tau_plus=tau_plus,
            tau_minus=tau_minus,
            a_plus=a_plus,
            a_minus=a_minus,
        )

    def __repr__(self):
        """Return a debug representation."""

        return f"STDPLayer(in_features={self.in_features}, out_features={self.out_features})"


class GIFNeuron(Module):
    """
    Generalized Integrate-and-Fire neuron.

    Uses: gif-neuron.glsl
    """

    def __init__(
        self,
        n_neurons: int,
        dt: float = 0.001,
        tau_mem: float = 20.0,
        tau_adapt: float = 100.0,
        v_thresh: float = 1.0,
        adaptation_strength: float = 0.1,
    ):
        """
        Initialize GIFNeuron.

        Args:
            n_neurons: Number of neurons
            dt: Time step
            tau_mem: Membrane time constant
            tau_adapt: Adaptation time constant
            v_thresh: Firing threshold
            adaptation_strength: Strength of adaptation
        """
        super().__init__()
        self.n_neurons = n_neurons
        self.dt = dt
        self.tau_mem = tau_mem
        self.tau_adapt = tau_adapt
        self.v_thresh = v_thresh
        self.adaptation_strength = adaptation_strength

        # Neuron state
        self.membrane = np.zeros(n_neurons, dtype=np.float32)
        self.adaptation = np.zeros(n_neurons, dtype=np.float32)
        self.refractory = np.zeros(n_neurons, dtype=np.float32)
        self._buffers["membrane"] = self.membrane
        self._buffers["adaptation"] = self.adaptation
        self._buffers["refractory"] = self.refractory

    def forward(self, input_current: np.ndarray) -> np.ndarray:
        """Forward pass using gif-neuron.glsl"""
        backend = self._get_backend()
        if hasattr(backend, "gif_neuron_step"):
            self.membrane, self.adaptation, self.refractory, spikes = backend.gif_neuron_step(
                input_current,
                self.membrane,
                self.adaptation,
                self.refractory,
                dt=self.dt,
                tau_mem=self.tau_mem,
                tau_adapt=self.tau_adapt,
                v_thresh=self.v_thresh,
                adaptation_strength=self.adaptation_strength,
            )
        else:
            # CPU fallback
            spikes = np.zeros(self.n_neurons, dtype=np.float32)
            for i in range(self.n_neurons):
                # Simplified GIF dynamics
                self.membrane[i] += (
                    (input_current[i] - self.membrane[i] - self.adaptation[i])
                    / self.tau_mem
                    * self.dt
                )
                self.adaptation[i] += (-self.adaptation[i] / self.tau_adapt) * self.dt

                if self.membrane[i] >= self.v_thresh:
                    spikes[i] = 1.0
                    self.membrane[i] = 0.0
                    self.adaptation[i] += self.adaptation_strength
                    self.refractory[i] = 0.01  # 10ms refractory
                else:
                    spikes[i] = 0.0
                    self.refractory[i] = max(0, self.refractory[i] - self.dt)

        return spikes

    def reset(self):
        """Reset neuron state"""
        self.membrane.fill(0)
        self.adaptation.fill(0)
        self.refractory.fill(0)

    def __repr__(self):
        """Return a debug representation."""

        return f"GIFNeuron(n_neurons={self.n_neurons}, tau_mem={self.tau_mem}, tau_adapt={self.tau_adapt})"


class SNNMatMul(Module):
    """
    SNN Matrix Multiplication layer.

    Uses: snn-matmul.glsl
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Initialize SNNMatMul layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight matrix
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = np.random.uniform(-limit, limit, (out_features, in_features)).astype(
            np.float32
        )
        self._parameters["weight"] = self.weight

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass - matrix multiplication for SNN.

        Args:
            x: Input spikes or rates (batch, in_features)

        Returns:
            Output (batch, out_features)
        """
        backend = self._get_backend()

        # Try GPU shader if available
        if hasattr(backend, "shaders") and "snn-matmul" in backend.shaders:
            try:
                # GPU SNN matmul would go here
                # For now, use CPU fallback
                pass
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        return x @ self.weight.T

    def __repr__(self):
        """Return a debug representation."""

        return f"SNNMatMul(in_features={self.in_features}, out_features={self.out_features})"


class SNNSoftmax(Module):
    """
    SNN Softmax layer.

    Uses: snn-softmax.glsl
    """

    def __init__(self, dim: int = -1, temperature: float = 1.0):
        """
        Initialize SNNSoftmax layer.

        Args:
            dim: Dimension to apply softmax
            temperature: Temperature parameter
        """
        super().__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass - softmax for SNN.

        Args:
            x: Input tensor

        Returns:
            Softmax output
        """
        backend = self._get_backend()

        # Try GPU shader if available
        if hasattr(backend, "shaders") and "snn-softmax" in backend.shaders:
            try:
                # GPU SNN softmax would go here
                # For now, use CPU fallback
                pass
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        x_scaled = x / self.temperature
        x_max = np.max(x_scaled, axis=self.dim, keepdims=True)
        x_exp = np.exp(x_scaled - x_max)
        return x_exp / np.sum(x_exp, axis=self.dim, keepdims=True)

    def __repr__(self):
        """Return a debug representation."""

        return f"SNNSoftmax(dim={self.dim}, temperature={self.temperature})"


class SNNRMSNorm(Module):
    """
    SNN RMS Normalization layer.

    Uses: snn-rmsnorm.glsl
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """
        Initialize SNNRMSNorm layer.

        Args:
            normalized_shape: Shape to normalize
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.weight = np.ones(normalized_shape, dtype=np.float32)
        self._parameters["weight"] = self.weight

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass - RMS normalization for SNN.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        backend = self._get_backend()

        # Try GPU shader if available
        if hasattr(backend, "shaders") and "snn-rmsnorm" in backend.shaders:
            try:
                # GPU SNN RMSNorm would go here
                # For now, use CPU fallback
                pass
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight

    def __repr__(self):
        """Return a debug representation."""

        return f"SNNRMSNorm(normalized_shape={self.normalized_shape}, eps={self.eps})"


class SNNReadout(Module):
    """
    SNN Readout layer.

    Uses: snn-readout.glsl
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Initialize SNNReadout layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Readout weights
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = np.random.uniform(-limit, limit, (out_features, in_features)).astype(
            np.float32
        )
        self._parameters["weight"] = self.weight

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass - readout from SNN.

        Args:
            x: Input spikes or rates (batch, time, in_features) or (batch, in_features)

        Returns:
            Readout output (batch, out_features)
        """
        backend = self._get_backend()

        # Try GPU shader if available
        if hasattr(backend, "shaders") and "snn-readout" in backend.shaders:
            try:
                # GPU SNN readout would go here
                # For now, use CPU fallback
                pass
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        if x.ndim == 3:
            # Average over time dimension
            x = x.mean(axis=1)
        return x @ self.weight.T

    def __repr__(self):
        """Return a debug representation."""

        return f"SNNReadout(in_features={self.in_features}, out_features={self.out_features})"


class Synapse(Module):
    """
    Synaptic connection layer.

    Uses: synapsis-forward.glsl
    """

    def __init__(self, in_features: int, out_features: int, delay: int = 1):
        """
        Initialize Synapse layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            delay: Synaptic delay (timesteps)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.delay = delay

        # Synaptic weights
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = np.random.uniform(-limit, limit, (out_features, in_features)).astype(
            np.float32
        )
        self._parameters["weight"] = self.weight

        # Delay buffer
        self.delay_buffer = [np.zeros((out_features,), dtype=np.float32) for _ in range(delay)]
        self._buffers["delay_buffer"] = np.array(self.delay_buffer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass - synaptic transmission with delay.

        Args:
            x: Input spikes (batch, in_features)

        Returns:
            Output (batch, out_features)
        """
        backend = self._get_backend()

        # Try GPU shader if available
        if hasattr(backend, "shaders") and "synapsis-forward" in backend.shaders:
            try:
                # GPU synapse forward would go here
                # For now, use CPU fallback
                pass
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        output = x @ self.weight.T

        # Apply delay (simplified - full implementation would use delay buffer)
        if self.delay > 1:
            # Store in delay buffer and return delayed output
            self.delay_buffer.append(output)
            if len(self.delay_buffer) > self.delay:
                output = self.delay_buffer.pop(0)

        return output

    def __repr__(self):
        """Return a debug representation."""

        return f"Synapse(in_features={self.in_features}, out_features={self.out_features}, delay={self.delay})"
