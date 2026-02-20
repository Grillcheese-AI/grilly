"""
Functional API for Spiking Neural Networks.

Stateless functions for SNN operations, mirroring torch.nn.functional style.
"""

import numpy as np

from ..nn.snn_base import BaseNode, MemoryModule
from ..nn.snn_surrogate import ATan


def lif_step(x, v, tau=2.0, v_threshold=1.0, v_reset=0.0, surrogate_fn=None):
    """Single LIF neuron step (functional, stateless).

    Args:
        x: Input current (N, *)
        v: Current membrane potential (N, *)
        tau: Membrane time constant
        v_threshold: Spike threshold
        v_reset: Reset voltage (None for soft reset)
        surrogate_fn: Surrogate gradient function (default: ATan)

    Returns:
        spike: Binary spikes (N, *)
        v_new: Updated membrane potential (N, *)
    """
    if surrogate_fn is None:
        surrogate_fn = ATan()

    if v_reset is not None:
        h = v + (x - (v - v_reset)) / tau
    else:
        h = v + (x - v) / tau

    spike = surrogate_fn(h - v_threshold)

    if v_reset is not None:
        v_new = h * (1.0 - spike) + v_reset * spike
    else:
        v_new = h - v_threshold * spike

    return spike, v_new


def if_step(x, v, v_threshold=1.0, v_reset=0.0, surrogate_fn=None):
    """Single IF neuron step (functional, stateless).

    Args:
        x: Input current (N, *)
        v: Current membrane potential (N, *)
        v_threshold: Spike threshold
        v_reset: Reset voltage (None for soft reset)
        surrogate_fn: Surrogate gradient function (default: ATan)

    Returns:
        spike: Binary spikes (N, *)
        v_new: Updated membrane potential (N, *)
    """
    if surrogate_fn is None:
        surrogate_fn = ATan()

    h = v + x
    spike = surrogate_fn(h - v_threshold)

    if v_reset is not None:
        v_new = h * (1.0 - spike) + v_reset * spike
    else:
        v_new = h - v_threshold * spike

    return spike, v_new


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


def seq_to_ann_forward(x_seq, module):
    """Reshape [T,N,...] to [T*N,...], run ANN module, reshape back.

    Args:
        x_seq: Input (T, N, *spatial)
        module: ANN module (e.g., Conv2d, Linear)

    Returns:
        Output (T, N, *spatial_out)
    """
    T, N = x_seq.shape[0], x_seq.shape[1]
    spatial = x_seq.shape[2:]
    x_flat = x_seq.reshape(T * N, *spatial)
    out = module(x_flat)
    out_spatial = out.shape[1:]
    return out.reshape(T, N, *out_spatial)


def reset_net(net):
    """Recursively reset all MemoryModules in a network.

    Call between training batches to clear temporal state.

    Args:
        net: Module or nested structure to reset
    """
    if isinstance(net, MemoryModule):
        net.reset()
    elif hasattr(net, "_modules"):
        for module in net._modules.values():
            reset_net(module)


def set_step_mode(net, mode):
    """Set step_mode on all BaseNodes in a network.

    Args:
        net: Module or nested structure
        mode: 's' for single-step, 'm' for multi-step
    """
    if isinstance(net, BaseNode):
        net.step_mode = mode
    if hasattr(net, "_modules"):
        for module in net._modules.values():
            set_step_mode(module, mode)
