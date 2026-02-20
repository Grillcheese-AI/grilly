"""
SNN Container modules for temporal dimension handling.

MultiStepContainer: Wraps single-step module for [T,N,...] input.
SeqToANNContainer: Reshapes [T,N,...] to [T*N,...] for ANN modules.
Flatten: Flatten spatial dimensions (for use in Sequential).

GPU-resident mode: When gpu_mode(True) is set, ANN modules inside
SeqToANNContainer keep intermediate activations on GPU (as VulkanTensor)
between layers, avoiding PCIe round-trips. Data is materialized to
numpy at container boundaries for SNN neuron processing.
"""

import numpy as np

from .module import Module


def _to_numpy(x):
    """Convert VulkanTensor to numpy if needed, pass numpy through."""
    from ..utils.tensor_conversion import VulkanTensor

    if isinstance(x, VulkanTensor):
        return x.numpy()
    return x


def multi_step_forward(x_seq, module):
    """Process temporal sequence through a single-step module.

    Args:
        x_seq: Input sequence (T, N, *)
        module: Single-step module

    Returns:
        Output sequence (T, N, *)
    """
    T = x_seq.shape[0]
    outputs = []
    for t in range(T):
        outputs.append(module(x_seq[t]))
    return np.stack(outputs, axis=0)


class MultiStepContainer(Module):
    """Wraps a single-step module to process [T, N, ...] input.

    Iterates over the time dimension T, calling the wrapped module
    at each timestep. Use for modules that don't natively support
    temporal sequences (e.g., MaxPool2d, Flatten).

    Args:
        module: Single-step module to wrap
    """

    def __init__(self, module):
        super().__init__()
        self._modules["module"] = module
        self.module = module
        self._cached_inputs = []

    def forward(self, x_seq):
        """Process [T, N, ...] by looping over T.

        Accepts numpy array. If inner module returns VulkanTensor
        (gpu_mode), converts back to numpy for stacking.
        """
        T = x_seq.shape[0]
        self._cached_inputs = []
        outputs = []
        for t in range(T):
            x_t = x_seq[t]
            self._cached_inputs.append(np.asarray(_to_numpy(x_t)).copy())
            out_t = self.module(x_t)
            outputs.append(_to_numpy(out_t))
        return np.stack(outputs, axis=0)

    def backward(self, grad_output, x=None):
        """Backward through each timestep."""
        T = grad_output.shape[0]
        grad_inputs = []
        for t in range(T):
            inp_t = self._cached_inputs[t] if t < len(self._cached_inputs) else None
            if hasattr(self.module, "backward"):
                try:
                    g = self.module.backward(grad_output[t], inp_t)
                except TypeError:
                    g = self.module.backward(grad_output[t])
            else:
                g = grad_output[t]
            grad_inputs.append(g if g is not None else grad_output[t])
        return np.stack(grad_inputs, axis=0)

    def __repr__(self):
        return f"MultiStepContainer({self.module})"


class SeqToANNContainer(Module):
    """Wraps ANN modules to handle [T, N, ...] temporal sequences.

    Reshapes [T, N, ...] to [T*N, ...], runs the ANN module(s),
    then reshapes back to [T, N, ...].

    This is the key building block for using Conv2d, BatchNorm2d, Linear
    in SNN pipelines — these layers don't need temporal awareness.

    GPU-resident mode: When gpu_mode(True) is set on the parent model,
    the ANN modules inside this container keep activations on GPU
    between layers (e.g., Conv2d output stays as VulkanTensor for
    BatchNorm2d input). Data is only downloaded at the container boundary
    for the [T*N,...] → [T,N,...] reshape.

    Args:
        *modules: One or more ANN modules to apply sequentially
    """

    def __init__(self, *modules):
        super().__init__()
        self.ann_modules = list(modules)
        for i, mod in enumerate(modules):
            self._modules[str(i)] = mod
        self._T = None
        self._N = None
        self._cached_intermediates = []

    def forward(self, x_seq):
        """Process [T, N, ...] through ANN modules.

        When gpu_mode is on, intermediate activations between ANN modules
        stay on GPU as VulkanTensors. The final output is materialized to
        numpy for the [T*N,...] → [T,N,...] reshape.
        """
        self._T, self._N = x_seq.shape[0], x_seq.shape[1]
        spatial_shape = x_seq.shape[2:]

        # Merge T and N: [T, N, ...] -> [T*N, ...]
        x_flat = x_seq.reshape(self._T * self._N, *spatial_shape)

        # Cache intermediates for backward (always numpy for backward pass)
        self._cached_intermediates = [np.asarray(_to_numpy(x_flat)).copy()]
        for mod in self.ann_modules:
            x_flat = mod(x_flat)
            # Cache numpy for backward, but pass through VulkanTensor to next layer
            self._cached_intermediates.append(np.asarray(_to_numpy(x_flat)).copy())

        # Materialize final output to numpy for reshape
        x_flat = _to_numpy(x_flat)

        # Split T and N back: [T*N, ...] -> [T, N, ...]
        out_spatial = x_flat.shape[1:]
        return x_flat.reshape(self._T, self._N, *out_spatial)

    def backward(self, grad_output, x=None):
        """Backward through ANN modules with reshape."""
        T, N = self._T, self._N
        out_spatial = grad_output.shape[2:]

        # Merge T and N for backward: [T, N, ...] -> [T*N, ...]
        grad = grad_output.reshape(T * N, *out_spatial)

        # Backward through ANN modules in reverse
        for i in range(len(self.ann_modules) - 1, -1, -1):
            mod = self.ann_modules[i]
            inp = self._cached_intermediates[i] if i < len(self._cached_intermediates) else None
            if hasattr(mod, "backward"):
                try:
                    grad = mod.backward(grad, inp)
                except TypeError:
                    grad = mod.backward(grad)
                if grad is None:
                    grad = np.zeros_like(self._cached_intermediates[i])
            # else: pass gradient through

        # Split back: [T*N, ...] -> [T, N, ...]
        in_spatial = grad.shape[1:]
        return grad.reshape(T, N, *in_spatial)

    def __repr__(self):
        mods = ", ".join(repr(m) for m in self.ann_modules)
        return f"SeqToANNContainer({mods})"


class Flatten(Module):
    """Flatten spatial dimensions in a tensor.

    For use in Sequential pipelines to transition from conv to linear layers.

    Args:
        start_dim: First dim to flatten (default: 1, preserves batch dim)
        end_dim: Last dim to flatten (default: -1)
    """

    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self._input_shape = None

    def forward(self, x):
        self._input_shape = x.shape
        shape = x.shape
        ndim = len(shape)

        start = self.start_dim if self.start_dim >= 0 else ndim + self.start_dim
        end = self.end_dim if self.end_dim >= 0 else ndim + self.end_dim

        # Compute new shape
        new_shape = shape[:start] + (int(np.prod(shape[start : end + 1])),)
        if end + 1 < ndim:
            new_shape = new_shape + shape[end + 1 :]

        return x.reshape(new_shape)

    def backward(self, grad_output, x=None):
        """Unflatten gradient back to original shape."""
        if self._input_shape is not None:
            return grad_output.reshape(self._input_shape)
        return grad_output

    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"
