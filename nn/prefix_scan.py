"""Causal Linear-RNN prefix scan — autograd-wrapped Python frontend.

Wraps the C++ / Vulkan ``grilly_core.prefix_scan_causal`` and
``prefix_scan_causal_backward`` kernels into grilly's autograd system so
``loss.backward()`` flows gradients through the recurrence.

Math:
    Forward:   h_t  = a_t * h_{t-1} + x_t          (h_0 = 0)
    Backward:  dx_t = dh_t + a_{t+1} * dx_{t+1}    (anti-causal scan)
               da_t = dx_t * h_{t-1}

The shader runs one subgroup per (batch, hidden_dim) pair, one thread per
time step, and uses ``subgroupInclusiveAdd`` for O(log S) parallel depth.

Constraint: ``seq_len <= 32``. Longer sequences need a hierarchical scan
(chunk the sequence, carry state between chunks) — not implemented yet.

Example::

    from grilly.nn.prefix_scan import prefix_scan_causal
    h = prefix_scan_causal(x, a)   # x, a: (B, S, D) Variables
    loss = h.mean()
    loss.backward()                # grads flow to both x and a
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from grilly.nn.autograd import GradFn, Variable, _ensure_variable, _grad_enabled


def _get_bridge_device():
    """Return the grilly bridge device with shaders loaded."""
    from grilly.backend import _bridge

    dev = _bridge._get_device()
    if dev is None:
        raise RuntimeError(
            "grilly bridge device not initialized — "
            "import grilly (or grilly.torch_api) first so shaders load"
        )
    return dev


def prefix_scan_causal(x, a) -> Variable:
    """Causal Linear-RNN: ``h_t = a_t * h_{t-1} + x_t``.

    Args:
        x: Input sequence, shape ``(B, S, D)``. Any ``Variable`` / ``Tensor``
           / ndarray works.
        a: Decay gates in ``(0, 1]``, same shape as ``x``.

    Returns:
        ``Variable`` of shape ``(B, S, D)`` with the causal hidden states.
        If autograd is enabled and either input requires grad, the result
        is wired into the graph via a ``GradFn`` that calls the C++
        ``prefix_scan_causal_backward`` kernel on ``loss.backward()``.
    """
    import grilly_core as gc

    x_var = _ensure_variable(x)
    a_var = _ensure_variable(a)

    x_data = np.asarray(x_var.data, dtype=np.float32)
    a_data = np.asarray(a_var.data, dtype=np.float32)

    if x_data.shape != a_data.shape:
        raise ValueError(
            f"prefix_scan_causal: x shape {x_data.shape} != a shape {a_data.shape}"
        )
    if x_data.ndim != 3:
        raise ValueError(
            f"prefix_scan_causal: inputs must be 3D (B, S, D), got {x_data.ndim}D"
        )

    S = x_data.shape[1]
    if S > 32:
        raise ValueError(
            f"prefix_scan_causal: seq_len {S} > 32. The current shader runs "
            f"one subgroup per (batch, dim) pair with one thread per time "
            f"step — hierarchical multi-subgroup scan is a TODO. Either "
            f"chunk the sequence on the Python side or truncate to 32 for "
            f"the correctness run."
        )

    dev = _get_bridge_device()
    h_data = np.asarray(gc.prefix_scan_causal(dev, x_data, a_data), dtype=np.float32)

    # ── Autograd wiring ──
    requires_grad = (
        _grad_enabled
        and (x_var.requires_grad or a_var.requires_grad)
    )
    if not requires_grad:
        return Variable(h_data, requires_grad=False)

    # Capture tensors needed by the backward closure. We save the forward
    # x / a / h as immutable ndarrays so later in-place mutations on the
    # caller's buffers don't corrupt the backward pass.
    saved_x = x_data.copy()
    saved_a = a_data.copy()
    saved_h = h_data.copy()

    def backward_fn(grad_output):
        # grad_output is dh coming back from downstream. Dispatch the
        # anti-causal kernel and split the returned dict into (grad_x, grad_a).
        grad_h = np.asarray(grad_output, dtype=np.float32)
        result = gc.prefix_scan_causal_backward(
            dev, grad_h, saved_a, saved_h, saved_x
        )
        grad_x = np.asarray(result["grad_x"], dtype=np.float32)
        grad_a = np.asarray(result["grad_a"], dtype=np.float32)
        return (grad_x, grad_a)

    # GradFn inputs list order MUST match the backward return tuple order.
    grad_fn = GradFn("PrefixScanCausal", backward_fn, [x_var, a_var])
    return Variable(h_data, requires_grad=True, grad_fn=grad_fn)


class CausalSequenceMixer:
    """Subgroup-accelerated causal sequence mixer.

    Replaces the ``h.mean(dim=1)`` sequence pooling in the old LiquidCell
    path, which destroyed causality by letting any time step see the future.
    This module runs a proper Linear-RNN that strictly respects causal
    masking.

    Architecture:
        x_t = proj_x(x)              # input projection
        a_t = sigmoid(proj_a(x))     # decay gate in (0, 1)
        h_t = a_t * h_{t-1} + x_t    # causal prefix scan (GPU)

    Implemented as a regular grilly ``nn.Module``, not a subclass — kept
    minimal and self-contained so it's easy to drop into the v3c script.
    Use via ``mixer = CausalSequenceMixer(d); h = mixer(x)`` where ``x``
    is shape ``(B, S, D)``.

    NOTE: constructed with explicit ``grilly.nn`` imports at call time to
    avoid a circular import at module load.
    """

    def __init__(self, d: int):
        from grilly import nn

        self.d = d
        self.proj_x = nn.Linear(d, d, bias=False)
        self.proj_a = nn.Linear(d, d, bias=True)
        # Initialize the gate bias to +1 so sigmoid(1) ≈ 0.73 — the model
        # starts out "remembering" most of the hidden state at t=0, which
        # matches the LiquidCell behavior the old code defaulted to.
        try:
            b_arr = np.asarray(self.proj_a.bias.data if hasattr(self.proj_a.bias, "data")
                               else self.proj_a.bias)
            b_arr[:] = 1.0
        except Exception:
            pass

    def parameters(self):
        yield from self.proj_x.parameters()
        yield from self.proj_a.parameters()

    def __call__(self, x):
        # x: (B, S, D) — a Variable / Tensor from upstream.
        x_t = self.proj_x(x)

        # Sigmoid without importing torch_api: use the identity
        #   sigmoid(z) = 0.5 * (1 + tanh(z/2))
        # which is numerically stable and uses only tanh, which grilly's
        # autograd exposes via Variable.tanh().
        a_logits = self.proj_a(x)
        if hasattr(a_logits, "tanh"):
            a_t = 0.5 * (1.0 + (a_logits * 0.5).tanh())
        else:
            # Fallback for plain ndarray — upstream should be a Variable.
            import numpy as _np
            a_t = 0.5 * (1.0 + _np.tanh(_np.asarray(a_logits) * 0.5))

        h = prefix_scan_causal(x_t, a_t)
        return h
