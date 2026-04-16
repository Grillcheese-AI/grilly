"""Causal Linear-RNN prefix scan — autograd-wrapped Python frontend.

Wraps the C++ / Vulkan ``grilly_core.prefix_scan_causal`` and
``prefix_scan_causal_backward`` kernels into grilly's autograd system so
``loss.backward()`` flows gradients through the recurrence.

Math:
    Forward:   h_t  = a_t * h_{t-1} + x_t          (h_0 = 0)
    Backward:  dx_t = dh_t + a_{t+1} * dx_{t+1}    (anti-causal recurrence)
               da_t = dx_t * h_{t-1}

The shader runs one thread per (batch, hidden_dim) pair with an inner
sequential loop over time. Parallelism is across (batch × hidden_dim),
so there is **no seq_len cap** — any length that fits in GPU memory works.
Previous revision used a subgroup scan and capped at 32.

Fused MinGRU:
    Forward:   x_scan = sigmoid(g) * tanh(v)
               a      = 0.05 + 0.9 * sigmoid(d)
               h_t    = a_t * h_{t-1} + x_scan_t
               (Computed in a single fused GPU kernel)

Example::

    from grilly.nn.prefix_scan import prefix_scan_causal
    h = prefix_scan_causal(x, a)   # x, a: (B, S, D) Variables
    loss = h.mean()
    loss.backward()                # grads flow to both x and a
"""

from __future__ import annotations

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


def min_gru(g, v, d) -> Variable:
    """Fused MinGRU mixer: ``h_t = a_t * h_{t-1} + x_scan_t``.
    
    Logic:
        x_scan_t = sigmoid(g_t) * tanh(v_t)
        a_t      = 0.05 + 0.9 * sigmoid(d_t)
        h_t      = a_t * h_{t-1} + x_scan_t
        
    Args:
        g, v, d: Gate projections of shape ``(B, S, D)``.
    """
    import grilly_core as gc

    g_var = _ensure_variable(g)
    v_var = _ensure_variable(v)
    d_var = _ensure_variable(d)

    g_data = np.asarray(g_var.data, dtype=np.float32)
    v_data = np.asarray(v_var.data, dtype=np.float32)
    d_data = np.asarray(d_var.data, dtype=np.float32)

    dev = _get_bridge_device()
    h_data = np.asarray(gc.mingru_forward(dev, g_data, v_data, d_data), dtype=np.float32)

    requires_grad = (
        _grad_enabled
        and (g_var.requires_grad or v_var.requires_grad or d_var.requires_grad)
    )
    if not requires_grad:
        return Variable(h_data, requires_grad=False)

    saved_g = g_data.copy()
    saved_v = v_data.copy()
    saved_d = d_data.copy()
    saved_h = h_data.copy()

    def backward_fn(grad_output):
        grad_h = np.asarray(grad_output, dtype=np.float32)
        result = gc.mingru_backward(
            dev, grad_h, saved_g, saved_v, saved_d, saved_h
        )
        return (
            np.asarray(result["grad_g"], dtype=np.float32),
            np.asarray(result["grad_v"], dtype=np.float32),
            np.asarray(result["grad_d"], dtype=np.float32)
        )

    grad_fn = GradFn("MinGRUFused", backward_fn, [g_var, v_var, d_var])
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

    def __init__(self, d: int, proj_x_init_scale: float = 1.0):
        """Build the mixer.

        Args:
            d: hidden dimension.
            proj_x_init_scale: multiplicative scale on the default Xavier
                init of ``proj_x``. Use values <1 to tame the recurrence's
                accumulated gain over a long sequence — for a sequence of
                length S the effective amplification of x_t through the
                Linear-RNN can be up to ~S, so very deep stacks need
                ``proj_x_init_scale ≈ 1/sqrt(S)`` to keep the residual
                stream stable. Defaults to 1.0 to preserve old behavior.
        """
        from grilly import nn

        self.d = d
        self.proj_x = nn.Linear(d, d, bias=False)
        self.proj_a = nn.Linear(d, d, bias=True)

        # Optionally shrink proj_x weights so the recurrence's gain stays O(1)
        # over the full sequence length. The mixer accumulates x_t through
        # ``h_t = a_t * h_{t-1} + x_t``, so even with a_t < 1 the steady-state
        # ‖h‖ scales like ‖x_t‖ / (1 - a). At a_max = 0.95 that's a 20x
        # amplification — without this rescale the residual stream blows up
        # in deep stacks (12+ layers).
        if proj_x_init_scale != 1.0:
            try:
                w_arr = np.asarray(self.proj_x.weight.data if hasattr(self.proj_x.weight, "data")
                                   else self.proj_x.weight)
                w_arr *= float(proj_x_init_scale)
            except Exception:
                pass

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

        # Bounded gate in [0.05, 0.95] using tanh, NOT sigmoid.
        #
        # Why bounded: ``prefix_scan_causal`` runs the scan in log-space:
        # ``subgroupInclusiveAdd(log(a))``. If ``a_t`` ever lands very near
        # 0, ``log(a_t) -> -inf``, the partial sum overflows, and the
        # forward output of the scan is nan. Sigmoid (which the previous
        # version used) has range (0, 1), and as the projection weights grow
        # during training some hidden dims drift to a_logits ~ -20, where
        # ``sigmoid(-20) ~ 2e-9``. ``log(2e-9) ~ -20``, accumulated 32 times
        # over a sequence = -640, and ``exp(-640) = 0`` -> divide-by-zero
        # in the scan rescaling. Tanh-bounded gate (range [0.05, 0.95])
        # gives ``log(0.05) = -3.0`` even at saturation, which is safe.
        a_logits = self.proj_a(x)
        if hasattr(a_logits, "tanh"):
            # Use the same expression structure as before (Variable arithmetic
            # exposed by grilly autograd: ``var * scalar``, ``scalar + var``,
            # ``var.tanh()``). Build sigmoid via tanh, then squash into
            # ``[0.05, 0.95]`` so log(a_t) >= -3.0 and the prefix scan stays
            # numerically safe.
            sig = 0.5 * (1.0 + (a_logits * 0.5).tanh())
            a_t = 0.05 + sig * 0.9
        else:
            import numpy as _np
            sig = 0.5 * (1.0 + _np.tanh(_np.asarray(a_logits) * 0.5))
            a_t = 0.05 + sig * 0.9

        h = prefix_scan_causal(x_t, a_t)
        return h


class MinGRU:
    """Fused MinGRU layer using the specialized GPU kernel."""
    
    def __init__(self, d_model: int, bias_init: float = 1.0):
        from grilly import nn
        self.d_model = d_model
        self.proj_g = nn.Linear(d_model, d_model, bias=True)
        self.proj_v = nn.Linear(d_model, d_model, bias=True)
        self.proj_d = nn.Linear(d_model, d_model, bias=True)

        try:
            b = self.proj_d.bias
            b_arr = b.data if hasattr(b, "data") else b
            b_arr[:] = float(bias_init)
        except Exception:
            pass

    def parameters(self):
        yield from self.proj_g.parameters()
        yield from self.proj_v.parameters()
        yield from self.proj_d.parameters()

    def __call__(self, x):
        g = self.proj_g(x)
        v = self.proj_v(x)
        d = self.proj_d(x)
        return min_gru(g, v, d)
