"""
Bridge layer: routes Python nn module ops through grilly_core C++ extension.

Instead of the legacy Python ctypes Vulkan path (struct.pack -> vkMapMemory ->
ctypes.memmove -> dispatch -> fence wait -> repeat per op), the bridge calls
pybind11-bound C++ ops that use VMA persistent mapping (single memcpy, zero
vkMap/vkUnmap) and BufferPool bucketed allocation.

Higher-level modules (nn/modules.py, nn/conv.py, nn/normalization.py) call
bridge functions with a try/fallback pattern:
    result = _bridge.linear(x, weight, bias)
    if result is not None: return result
    # else fall through to legacy backend

Fused MoE (:func:`moe_upload`, :func:`moe_forward`, :func:`moe_backward`, etc.)
uses the same lazy :func:`_get_device` and ``shaders/spv`` load as other bridge
ops—prefer these over calling ``grilly_core.moe_*`` with a separate Device.
"""

import logging
import os

import numpy as np

logger = logging.getLogger("grilly.bridge")
_BRIDGE_STRICT = os.getenv("GRILLY_BRIDGE_STRICT", "0").strip().lower() in {"1", "true", "yes", "on"}
_FALLBACK_COUNTS = {}


def _maybe_trace(op_name, inputs, output, **kwargs):
    """Record an op if JIT tracing is active."""
    try:
        from .jit import Tracer
        tracer = Tracer.current()
        if tracer is not None:
            tracer.record_op(op_name, inputs, output, **kwargs)
    except ImportError:
        pass


def _record_fallback(op_name: str, reason: Exception | str | None = None):
    """Record fallback events and optionally raise in strict mode."""
    _FALLBACK_COUNTS[op_name] = _FALLBACK_COUNTS.get(op_name, 0) + 1
    if reason is not None:
        logger.debug("%s fallback triggered: %s", op_name, reason)
    if _BRIDGE_STRICT:
        raise RuntimeError(f"GRILLY_BRIDGE_STRICT=1: bridge op '{op_name}' failed: {reason}")


def get_fallback_stats() -> dict[str, int]:
    """Get fallback counts per bridge op."""
    return dict(_FALLBACK_COUNTS)


def reset_fallback_stats():
    """Reset bridge fallback counters."""
    _FALLBACK_COUNTS.clear()

try:
    import grilly_core as _core

    _NATIVE = True
except ImportError:
    _core = None
    _NATIVE = False

# ── Lazy singleton device ────────────────────────────────────────────────

_device = None


def _get_device():
    """Lazily initialize the C++ Vulkan device and load shaders."""
    global _device
    if _device is not None:
        return _device
    if not _NATIVE:
        return None
    try:
        _device = _core.Device()
        # Find shaders relative to this file: backend/_bridge.py -> ../../shaders/spv/
        shader_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shaders", "spv"
        )
        if os.path.isdir(shader_dir):
            _device.load_shaders(shader_dir)
        return _device
    except Exception:
        _device = None
        return None


def is_available():
    """Check if the C++ bridge is available and working."""
    return _get_device() is not None


# ── Helpers ──────────────────────────────────────────────────────────────


def _extract_cpp_tensor(obj):
    """Extract the C++ grilly_core.Tensor from a VulkanTensor. Returns None otherwise.

    Fix #4: Duck typing with try/except — avoids hasattr+isinstance+_NATIVE
    check overhead on every call (measurable at hundreds of calls per forward).
    """
    if obj is None:
        return None
    try:
        return obj._t
    except AttributeError:
        return None


def _ensure_f32_contiguous(arr):
    """Ensure array is float32 and C-contiguous for the C++ side."""
    if arr is None:
        return None
    # VulkanTensor: access C++ Tensor's numpy() directly (1 copy instead of 2)
    if hasattr(arr, '_t') and hasattr(arr._t, 'numpy'):
        arr = arr._t.numpy()
    else:
        arr = np.asarray(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


# ── Linear ───────────────────────────────────────────────────────────────


def linear(x, weight, bias=None):
    """GPU linear: output = x @ W^T + bias. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    # Fast path: Tensor I/O (zero numpy copies)
    if hasattr(_core, 'linear_t'):
        x_t = _extract_cpp_tensor(x)
        w_t = _extract_cpp_tensor(weight)
        if x_t is not None and w_t is not None:
            try:
                b_t = _extract_cpp_tensor(bias)
                result = _core.linear_t(dev, x_t, w_t, b_t)
                _maybe_trace("linear", [x, weight, bias], result)
                return result
            except Exception:
                pass  # fall through to numpy path
    # Standard path: numpy arrays
    try:
        x = _ensure_f32_contiguous(x)
        weight = _ensure_f32_contiguous(weight)
        bias = _ensure_f32_contiguous(bias)
        result = _core.linear(dev, x, weight, bias)
        _maybe_trace("linear", [x, weight, bias], result)
        return result
    except Exception as e:
        logger.debug("linear GPU failed (%s), caller will use CPU fallback", e)
        _record_fallback("linear", e)
        return None


# ── Activations ──────────────────────────────────────────────────────────


def relu(x):
    """GPU ReLU. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.relu(dev, _ensure_f32_contiguous(x))
    except Exception as e:
        _record_fallback("relu", e)
        return None


def gelu(x):
    """GPU GELU. Returns None on failure.

    Clamps input to [-10, 10] before dispatch — the GELU tanh approximation
    uses exp(2*inner) which overflows float32 for |x| > ~10. For |x| > 10:
      GELU(x) ≈ x (positive) or 0 (negative), so clamping loses no precision.
    """
    x = _ensure_f32_contiguous(x)
    # Pre-clamp: handle asymptotic values before GPU dispatch
    large_pos = x > 10.0
    large_neg = x < -10.0
    needs_clamp = np.any(large_pos) or np.any(large_neg)
    if needs_clamp:
        result = np.empty_like(x)
        result[large_pos] = x[large_pos]  # GELU → x for large positive
        result[large_neg] = 0.0            # GELU → 0 for large negative
        mask = ~(large_pos | large_neg)
        if np.any(mask):
            dev = _get_device()
            if dev is not None:
                try:
                    gpu_result = _core.gelu(dev, np.ascontiguousarray(x[mask]))
                    if gpu_result is not None:
                        result[mask] = np.asarray(gpu_result, dtype=np.float32)
                        return result
                except Exception:
                    pass
            # CPU fallback for middle range
            xm = x[mask]
            result[mask] = (0.5 * xm * (1.0 + np.tanh(
                np.sqrt(2.0 / np.pi) * (xm + 0.044715 * xm ** 3)
            ))).astype(np.float32)
        return result

    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.gelu(dev, x)
    except Exception as e:
        _record_fallback("gelu", e)
        return None


def silu(x):
    """GPU SiLU. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.silu(dev, _ensure_f32_contiguous(x))
    except Exception as e:
        _record_fallback("silu", e)
        return None


def tanh(x):
    """GPU tanh activation. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.tanh_act(dev, _ensure_f32_contiguous(x))
    except Exception as e:
        _record_fallback("tanh", e)
        return None


# ── Activation Backward ──────────────────────────────────────────────────


def relu_backward(grad_output, input):
    """GPU ReLU backward. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.relu_backward(
            dev, _ensure_f32_contiguous(grad_output), _ensure_f32_contiguous(input)
        )
    except Exception as e:
        _record_fallback("relu_backward", e)
        return None


def gelu_backward(grad_output, input):
    """GPU GELU backward. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.gelu_backward(
            dev, _ensure_f32_contiguous(grad_output), _ensure_f32_contiguous(input)
        )
    except Exception as e:
        _record_fallback("gelu_backward", e)
        return None


def silu_backward(grad_output, input):
    """GPU SiLU backward. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.silu_backward(
            dev, _ensure_f32_contiguous(grad_output), _ensure_f32_contiguous(input)
        )
    except Exception as e:
        _record_fallback("silu_backward", e)
        return None


# ── SNN Standalone Ops ───────────────────────────────────────────────────


def lif_step(
    input,
    v_mem,
    t_refrac,
    dt=1.0,
    tau_mem=20.0,
    v_rest=0.0,
    v_reset=0.0,
    v_thresh=1.0,
    r_mem=1.0,
    t_refrac_period=0.0,
):
    """GPU LIF neuron step. Returns dict with spikes, v_mem, t_refrac."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.lif_step(
            dev,
            _ensure_f32_contiguous(input),
            _ensure_f32_contiguous(v_mem),
            _ensure_f32_contiguous(t_refrac),
            dt,
            tau_mem,
            v_rest,
            v_reset,
            v_thresh,
            r_mem,
            t_refrac_period,
        )
    except Exception as e:
        _record_fallback("lif_step", e)
        return None


def snn_node_forward(
    x_in,
    v_mem,
    tau_param,
    neuron_type=1,
    tau=2.0,
    v_threshold=1.0,
    v_reset=0.0,
    reset_mode=0,
    decay_input=0,
):
    """GPU SNN node forward. Returns dict with spikes, v_mem, h_out."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.snn_node_forward(
            dev,
            _ensure_f32_contiguous(x_in),
            _ensure_f32_contiguous(v_mem),
            _ensure_f32_contiguous(tau_param),
            neuron_type,
            tau,
            v_threshold,
            v_reset,
            reset_mode,
            decay_input,
        )
    except Exception as e:
        _record_fallback("snn_node_forward", e)
        return None


def snn_node_backward(grad_spike, h_cache, alpha=2.0, surrogate_type=0, v_threshold=1.0):
    """GPU SNN node backward. Returns grad_x array."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.snn_node_backward(
            dev,
            _ensure_f32_contiguous(grad_spike),
            _ensure_f32_contiguous(h_cache),
            alpha,
            surrogate_type,
            v_threshold,
        )
    except Exception as e:
        _record_fallback("snn_node_backward", e)
        return None


def hebbian_learning(
    pre, post, weights, batch_size=1, time_steps=1, learning_rate=0.01, weight_decay=0.0
):
    """GPU Hebbian learning. Returns updated weights."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.hebbian_learning(
            dev,
            _ensure_f32_contiguous(pre),
            _ensure_f32_contiguous(post),
            _ensure_f32_contiguous(weights),
            batch_size,
            time_steps,
            learning_rate,
            weight_decay,
        )
    except Exception as e:
        _record_fallback("hebbian_learning", e)
        return None


def stdp_learning(
    pre,
    post,
    weights,
    pre_trace,
    post_trace,
    batch_size=1,
    time_steps=1,
    lr_pot=0.01,
    lr_dep=0.01,
    trace_decay=0.95,
):
    """GPU STDP learning. Returns dict with weights, pre_trace, post_trace."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.stdp_learning(
            dev,
            _ensure_f32_contiguous(pre),
            _ensure_f32_contiguous(post),
            _ensure_f32_contiguous(weights),
            _ensure_f32_contiguous(pre_trace),
            _ensure_f32_contiguous(post_trace),
            batch_size,
            time_steps,
            lr_pot,
            lr_dep,
            trace_decay,
        )
    except Exception as e:
        _record_fallback("stdp_learning", e)
        return None


def oja_learning(memories, inputs, num_vectors, dim, eta=0.01):
    """GPU Oja learning. Updates memory vectors via Oja's self-normalizing rule.

    Dispatches oja-learning.spv: Δm = η·y·(x - y·m) where y = m·x.
    Returns updated memory vectors.

    Args:
        memories: Memory vectors (num_vectors * dim) float32.
        inputs: Input vectors (num_vectors * dim) float32.
        num_vectors: Number of vector pairs to update.
        dim: VSA dimension per vector.
        eta: Learning rate.

    Returns:
        Updated memory vectors, or None on failure.
    """
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.oja_learning(
            dev,
            _ensure_f32_contiguous(memories),
            _ensure_f32_contiguous(inputs),
            num_vectors,
            dim,
            eta,
        )
    except Exception as e:
        _record_fallback("oja_learning", e)
        return None


def synapse_filter(x_in, y_state, decay=0.95):
    """GPU synapse filter. Returns updated y_state."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.synapse_filter(
            dev, _ensure_f32_contiguous(x_in), _ensure_f32_contiguous(y_state), decay
        )
    except Exception as e:
        _record_fallback("synapse_filter", e)
        return None


def gif_neuron_step(
    input,
    v_mem,
    i_adapt,
    g_input,
    g_forget,
    t_refrac,
    t_last_spike,
    dt=1.0,
    current_time=0.0,
    tau_mem=20.0,
    v_rest=0.0,
    v_reset=0.0,
    v_thresh=1.0,
    r_mem=1.0,
    tau_adapt=100.0,
    delta_adapt=0.1,
    b_adapt=0.0,
    tau_gate=50.0,
    gate_strength=1.0,
    t_refrac_period=0.0,
):
    """GPU GIF neuron step. Returns dict with all updated state."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.gif_neuron_step(
            dev,
            _ensure_f32_contiguous(input),
            _ensure_f32_contiguous(v_mem),
            _ensure_f32_contiguous(i_adapt),
            _ensure_f32_contiguous(g_input),
            _ensure_f32_contiguous(g_forget),
            _ensure_f32_contiguous(t_refrac),
            _ensure_f32_contiguous(t_last_spike),
            dt,
            current_time,
            tau_mem,
            v_rest,
            v_reset,
            v_thresh,
            r_mem,
            tau_adapt,
            delta_adapt,
            b_adapt,
            tau_gate,
            gate_strength,
            t_refrac_period,
        )
    except Exception as e:
        _record_fallback("gif_neuron_step", e)
        return None


# ── Conv2d ───────────────────────────────────────────────────────────────


def conv2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """GPU Conv2d. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        x = _ensure_f32_contiguous(x)
        weight = _ensure_f32_contiguous(weight)
        bias = _ensure_f32_contiguous(bias)
        return _core.conv2d(
            dev, x, weight, bias, list(stride), list(padding), list(dilation), groups
        )
    except Exception as e:
        _record_fallback("conv2d", e)
        return None


# ── VSA-CNN Fused Ops ────────────────────────────────────────────────────


def conv2d_3x3_gelu(x, weight, bias):
    """Fused 3x3 Conv2D + GELU via conv2d-3x3-gelu.spv. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.conv2d_3x3_gelu(
            dev,
            _ensure_f32_contiguous(x),
            _ensure_f32_contiguous(weight),
            _ensure_f32_contiguous(bias),
        )
    except Exception as e:
        _record_fallback("conv2d_3x3_gelu", e)
        return None


def maxpool2x2(x):
    """2x2 MaxPool stride 2 via maxpool-2x2.spv. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.maxpool2x2(dev, _ensure_f32_contiguous(x))
    except Exception as e:
        _record_fallback("maxpool2x2", e)
        return None


def adaptive_avgpool_3x3(x):
    """Adaptive AvgPool to 3x3 via adaptive-avgpool-3x3.spv. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.adaptive_avgpool_3x3(dev, _ensure_f32_contiguous(x))
    except Exception as e:
        _record_fallback("adaptive_avgpool_3x3", e)
        return None


# ── LayerNorm ────────────────────────────────────────────────────────────


def layernorm(x, gamma, beta, eps=1e-5):
    """GPU LayerNorm. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        x = _ensure_f32_contiguous(x)
        gamma = _ensure_f32_contiguous(gamma)
        beta = _ensure_f32_contiguous(beta)
        return _core.layernorm(dev, x, gamma, beta, eps)
    except Exception as e:
        _record_fallback("layernorm", e)
        return None


def layernorm_backward(grad_output, input, gamma, mean, var, eps=1e-5):
    """GPU LayerNorm backward. Returns dict with grad_input, grad_gamma, grad_beta."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.layernorm_backward(
            dev,
            _ensure_f32_contiguous(grad_output),
            _ensure_f32_contiguous(input),
            _ensure_f32_contiguous(gamma),
            _ensure_f32_contiguous(mean),
            _ensure_f32_contiguous(var),
            eps,
        )
    except Exception as e:
        _record_fallback("layernorm_backward", e)
        return None


# ── RMSNorm ─────────────────────────────────────────────────────────────


def rmsnorm(x, weight, eps=1e-5):
    """GPU RMSNorm: weight * x * rsqrt(mean(x^2) + eps). Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        x = _ensure_f32_contiguous(x)
        weight = _ensure_f32_contiguous(weight)
        return _core.rmsnorm(dev, x, weight, eps)
    except Exception as e:
        _record_fallback("rmsnorm", e)
        return None


# ── Additional Activation Ops ────────────────────────────────────────────


def tanh_backward(grad_output, tanh_output):
    """GPU tanh backward. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.tanh_backward(
            dev, _ensure_f32_contiguous(grad_output), _ensure_f32_contiguous(tanh_output)
        )
    except Exception as e:
        _record_fallback("tanh_backward", e)
        return None


def softmax(x, dim=-1):
    """GPU Softmax. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.softmax(dev, _ensure_f32_contiguous(x), dim)
    except Exception as e:
        _record_fallback("softmax", e)
        return None


def softmax_backward(grad_output, softmax_output):
    """GPU Softmax backward. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.softmax_backward(
            dev, _ensure_f32_contiguous(grad_output), _ensure_f32_contiguous(softmax_output)
        )
    except Exception as e:
        _record_fallback("softmax_backward", e)
        return None


def sampled_bce_fused(hidden, table, ids):
    """GPU fused sampled-BCE head (softmax-free): per-token loss + grad_hidden
    + grad_table in ONE submit. hidden (N,d) f32, table (V,d) f32,
    ids (N,1+K) uint32 with col 0 = the true target.
    Returns (losses, grad_hidden, grad_table) numpy arrays, or None on failure
    (caller falls back to the numpy path)."""
    dev = _get_device()
    if dev is None:
        return None
    if not hasattr(_core, "sampled_bce_fused"):
        return None
    try:
        hidden = _ensure_f32_contiguous(hidden)
        table = _ensure_f32_contiguous(table)
        ids = np.ascontiguousarray(np.asarray(ids), dtype=np.uint32)
        loss_t, gh_t, gw_t = _core.sampled_bce_fused(dev, hidden, table, ids)
        return (np.asarray(loss_t.numpy()), np.asarray(gh_t.numpy()),
                np.asarray(gw_t.numpy()))
    except Exception as e:
        _record_fallback("sampled_bce_fused", e)
        return None


def nce_fused(hidden, table, ids, logkq, bias, use_correction=True):
    """GPU fused NCE head (corrected softmax-free): per-token loss + grad_hidden
    + grad_table + grad_bias in ONE submit. hidden (N,d), table (V,d),
    ids (N,1+K) uint32 col0=target, logkq (V,) = log(k*q_id), bias (V,) learned.
    use_correction=False recovers the SGNS head. Returns
    (losses, grad_hidden, grad_table, grad_bias) or None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    if not hasattr(_core, "nce_fused"):
        return None
    try:
        hidden = _ensure_f32_contiguous(hidden)
        table = _ensure_f32_contiguous(table)
        logkq = _ensure_f32_contiguous(logkq)
        bias = _ensure_f32_contiguous(bias)
        ids = np.ascontiguousarray(np.asarray(ids), dtype=np.uint32)
        l_t, gh_t, gw_t, gb_t = _core.nce_fused(
            dev, hidden, table, ids, logkq, bias, bool(use_correction))
        return (np.asarray(l_t.numpy()), np.asarray(gh_t.numpy()),
                np.asarray(gw_t.numpy()), np.asarray(gb_t.numpy()))
    except Exception as e:
        _record_fallback("nce_fused", e)
        return None


def pack_ternary(w):
    """Quantize fp32 weights (N, K) to BitNet b1.58 and 2-bit pack for the
    ternary-gemm kernel. Returns (packed uint32 (N, ceil(K/16)), alpha float).

      alpha = mean|W|;  trit = clip(round(W/alpha), -1, 1) in {-1,0,+1}
      2-bit code: 0->0, +1->1, -1->2, packed 16 trits/uint32 (LSB-first).

    This is the host seam: swap to 5-trits/8-bit for true 1.58-bit later
    without touching the kernel's accumulation. Matches cubby ffn.ternarize_ste
    (alpha = mean|W|, same round/clip) so trained latent weights pack directly.
    """
    w = np.asarray(w, dtype=np.float32)
    N, K = w.shape
    alpha = float(np.abs(w).mean()) + 1e-5
    trit = np.clip(np.round(w / alpha), -1.0, 1.0).astype(np.int32)  # {-1,0,1}
    code = np.where(trit > 0, 1, np.where(trit < 0, 2, 0)).astype(np.uint32)
    words_per_row = (K + 15) // 16
    packed = np.zeros((N, words_per_row), dtype=np.uint32)
    for k in range(K):
        packed[:, k // 16] |= (code[:, k] & 0x3) << ((k % 16) * 2)
    return packed, alpha


def ternary_gemm(activations, weights_packed, K, alpha):
    """GPU multiply-free ternary GEMM: alpha * (A @ trit^T). activations (M,K)
    fp32, weights_packed (N, ceil(K/16)) uint32 from pack_ternary. Returns
    (M, N) fp32, or None on failure (caller falls back to fp32 matmul)."""
    dev = _get_device()
    if dev is None or not hasattr(_core, "ternary_gemm"):
        return None
    try:
        a = _ensure_f32_contiguous(activations)
        wp = np.ascontiguousarray(np.asarray(weights_packed), dtype=np.uint32)
        return np.asarray(_core.ternary_gemm(dev, a, wp, int(K), float(alpha)))
    except Exception as e:
        _record_fallback("ternary_gemm", e)
        return None


def mf_softmax(x, dim=-1):
    """GPU multiplication-free softmax (ReLU-normalized). Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.mf_softmax(dev, _ensure_f32_contiguous(x), dim)
    except Exception as e:
        _record_fallback("mf_softmax", e)
        return None


def mf_softplus(x, beta=1.0):
    """GPU algebraic softplus (sqrt form). Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.mf_softplus(dev, _ensure_f32_contiguous(x), float(beta))
    except Exception as e:
        _record_fallback("mf_softplus", e)
        return None


def mf_sigmoid(x):
    """GPU rational sigmoid x/(1+|x|). Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.mf_sigmoid(dev, _ensure_f32_contiguous(x))
    except Exception as e:
        _record_fallback("mf_sigmoid", e)
        return None


# ── Linear Backward + Dropout ────────────────────────────────────────────


def linear_backward(grad_output, input, weights):
    """GPU linear backward. Returns dict with grad_input, grad_weight, grad_bias."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.linear_backward(
            dev,
            _ensure_f32_contiguous(grad_output),
            _ensure_f32_contiguous(input),
            _ensure_f32_contiguous(weights),
        )
    except Exception as e:
        _record_fallback("linear_backward", e)
        return None


def dropout(x, random_mask, p=0.5, training=True):
    """GPU dropout. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.dropout(
            dev, _ensure_f32_contiguous(x), _ensure_f32_contiguous(random_mask), p, training
        )
    except Exception as e:
        _record_fallback("dropout", e)
        return None


# ── Conv2d Backward ──────────────────────────────────────────────────────


def conv2d_backward_input(
    grad_output, weight, input_shape, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1
):
    """GPU conv2d backward w.r.t. input. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.conv2d_backward_input(
            dev,
            _ensure_f32_contiguous(grad_output),
            _ensure_f32_contiguous(weight),
            list(input_shape),
            list(stride),
            list(padding),
            list(dilation),
            groups,
        )
    except Exception as e:
        _record_fallback("conv2d_backward_input", e)
        return None


def conv2d_backward_weight(
    grad_output,
    input,
    weight_shape,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    has_bias=False,
):
    """GPU conv2d backward w.r.t. weight. Returns dict or None."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.conv2d_backward_weight(
            dev,
            _ensure_f32_contiguous(grad_output),
            _ensure_f32_contiguous(input),
            list(weight_shape),
            list(stride),
            list(padding),
            list(dilation),
            groups,
            has_bias,
        )
    except Exception as e:
        _record_fallback("conv2d_backward_weight", e)
        return None


# ── Attention Ops ────────────────────────────────────────────────────────


def attention_scores(Q, K, scale=0.0):
    """GPU attention scores: Q @ K^T / sqrt(d_h). Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.attention_scores(
            dev, _ensure_f32_contiguous(Q), _ensure_f32_contiguous(K), scale
        )
    except Exception as e:
        _record_fallback("attention_scores", e)
        return None


def attention_mask(scores, mask=None, causal=True, mask_value=-1e9):
    """GPU attention mask. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        m = _ensure_f32_contiguous(mask) if mask is not None else None
        return _core.attention_mask(dev, _ensure_f32_contiguous(scores), m, causal, mask_value)
    except Exception as e:
        _record_fallback("attention_mask", e)
        return None


def attention_output(weights, V):
    """GPU attention output: softmax(scores) @ V. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.attention_output(
            dev, _ensure_f32_contiguous(weights), _ensure_f32_contiguous(V)
        )
    except Exception as e:
        _record_fallback("attention_output", e)
        return None


def attention_scores_softmax_output(Q, K, V, scale=0.0):
    """Fused attention: scores + softmax + weighted V in one GPU submit.

    Requires Q, K, V shaped (B, H, S, D) with identical S (self-attention / equal length).
    Returns ``(output, softmax_weights)`` as core tensors, or None on failure.
    """
    dev = _get_device()
    if dev is None:
        return None
    fused = getattr(_core, "attention_scores_softmax_output", None)
    if fused is None:
        return None
    try:
        return fused(
            dev,
            _ensure_f32_contiguous(Q),
            _ensure_f32_contiguous(K),
            _ensure_f32_contiguous(V),
            scale,
        )
    except Exception as e:
        _record_fallback("attention_scores_softmax_output", e)
        return None


def attention_concat_heads(mh_output):
    """GPU concat multi-head output. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.attention_concat_heads(dev, _ensure_f32_contiguous(mh_output))
    except Exception as e:
        _record_fallback("attention_concat_heads", e)
        return None


def rope(x, cos_table=None, sin_table=None, base=10000.0, scaling=1.0):
    """GPU RoPE. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        ct = _ensure_f32_contiguous(cos_table) if cos_table is not None else None
        st = _ensure_f32_contiguous(sin_table) if sin_table is not None else None
        return _core.rope(dev, _ensure_f32_contiguous(x), ct, st, base, scaling)
    except Exception as e:
        _record_fallback("rope", e)
        return None


def flash_attention2(Q, K, V, mask=None, scale=0.0, tile_size_q=64, tile_size_k=64):
    """GPU Flash Attention 2 with online softmax tiling.
    Q/K/V shape: (batch, heads, seq_len, head_dim). Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        Q = _ensure_f32_contiguous(Q)
        K = _ensure_f32_contiguous(K)
        V = _ensure_f32_contiguous(V)
        m = _ensure_f32_contiguous(mask) if mask is not None else None
        return _core.flash_attention2(dev, Q, K, V, m, scale, tile_size_q, tile_size_k)
    except Exception as e:
        _record_fallback("flash_attention2", e)
        return None


# ── Fused Transformer Ops ────────────────────────────────────────────────


def fused_mlp_gelu(x, w1, b1, w2, b2):
    """Fused MLP: Linear(d_in→d_hidden) → GELU → Linear(d_hidden→d_out).

    GPU shader: fused-mlp-gelu.spv — intermediate hidden stays in LDS,
    never hits VRAM. Eliminates 2 read/write cycles per layer.

    Args:
        x:  (seq_len, d_in) float32 input.
        w1: (d_hidden, d_in) float32 fc1 weights.
        b1: (d_hidden,) float32 fc1 bias.
        w2: (d_out, d_hidden) float32 fc2 weights.
        b2: (d_out,) float32 fc2 bias.

    Returns:
        (seq_len, d_out) float32 output, or None on failure.
    """
    # Try native fused dispatch
    dev = _get_device()
    if dev is not None:
        try:
            result = _core.fused_mlp_gelu(
                dev,
                _ensure_f32_contiguous(x),
                _ensure_f32_contiguous(w1),
                _ensure_f32_contiguous(b1),
                _ensure_f32_contiguous(w2),
                _ensure_f32_contiguous(b2),
            )
            if result is not None:
                return result
        except (AttributeError, Exception):
            pass

    # Fallback: 3 separate ops via bridge
    x = _ensure_f32_contiguous(x)
    h = linear(x, w1, b1)
    if h is None:
        h = (x @ w1.T + b1).astype(np.float32)
    h_gelu = gelu(h)
    if h_gelu is None:
        h = np.clip(h, -10, 10)
        h_gelu = (0.5 * h * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * h ** 3)))).astype(np.float32)
    out = linear(h_gelu, w2, b2)
    if out is None:
        out = (h_gelu @ w2.T + b2).astype(np.float32)
    return out


def fused_layernorm_linear(x, ln_weight, ln_bias, proj_weight, proj_bias, eps=1e-6):
    """Fused LayerNorm + Linear projection.

    GPU shader: fused-layernorm-linear.spv — normalized values stay in LDS.
    Saves 1 VRAM round-trip per pre-norm attention layer.

    Args:
        x:           (seq_len, d_in) float32 input.
        ln_weight:   (d_in,) float32 LayerNorm weight.
        ln_bias:     (d_in,) float32 LayerNorm bias.
        proj_weight: (d_out, d_in) float32 projection weights.
        proj_bias:   (d_out,) float32 projection bias.

    Returns:
        (seq_len, d_out) float32 output, or None on failure.
    """
    dev = _get_device()
    if dev is not None:
        try:
            result = _core.fused_layernorm_linear(
                dev,
                _ensure_f32_contiguous(x),
                _ensure_f32_contiguous(ln_weight),
                _ensure_f32_contiguous(ln_bias),
                _ensure_f32_contiguous(proj_weight),
                _ensure_f32_contiguous(proj_bias),
            )
            if result is not None:
                return result
        except (AttributeError, Exception):
            pass

    # Fallback: separate layernorm + linear
    x = _ensure_f32_contiguous(x)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    normed = ((x - mean) / np.sqrt(var + eps) * ln_weight + ln_bias).astype(np.float32)
    out = linear(normed, proj_weight, proj_bias)
    if out is None:
        out = (normed @ proj_weight.T + proj_bias).astype(np.float32)
    return out


# ── Pooling Ops ──────────────────────────────────────────────────────────


def maxpool2d(x, kernel_size, stride=(2, 2), padding=(0, 0), dilation=(1, 1)):
    """GPU MaxPool2d. Returns dict with output, indices. None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.maxpool2d(
            dev,
            _ensure_f32_contiguous(x),
            list(kernel_size),
            list(stride),
            list(padding),
            list(dilation),
        )
    except Exception as e:
        _record_fallback("maxpool2d", e)
        return None


def avgpool2d(x, kernel_size, stride=(2, 2), padding=(0, 0), count_include_pad=True):
    """GPU AvgPool2d. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.avgpool2d(
            dev,
            _ensure_f32_contiguous(x),
            list(kernel_size),
            list(stride),
            list(padding),
            count_include_pad,
        )
    except Exception as e:
        _record_fallback("avgpool2d", e)
        return None


def mean_pool(x):
    """GPU mean pooling over sequence dim: (B,S,D) -> (B,D)."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.mean_pool(dev, _ensure_f32_contiguous(x))
    except Exception as e:
        _record_fallback("mean_pool", e)
        return None


# ── BatchNorm2d ──────────────────────────────────────────────────────────


def batchnorm2d_forward(
    x, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.1, training=True
):
    """GPU BatchNorm2d forward. Returns dict or None."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.batchnorm2d_forward(
            dev,
            _ensure_f32_contiguous(x),
            _ensure_f32_contiguous(gamma),
            _ensure_f32_contiguous(beta),
            _ensure_f32_contiguous(running_mean),
            _ensure_f32_contiguous(running_var),
            eps,
            momentum,
            training,
        )
    except Exception as e:
        _record_fallback("batchnorm2d_forward", e)
        return None


# ── Loss Functions ───────────────────────────────────────────────────────


def cross_entropy_loss(logits, targets, label_smoothing=0.0):
    """GPU cross-entropy loss. Returns loss array or None."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        targets = np.asarray(targets)
        if targets.dtype != np.uint32:
            targets = targets.astype(np.uint32)
        return _core.cross_entropy_loss(
            dev, _ensure_f32_contiguous(logits), targets, label_smoothing
        )
    except Exception as e:
        _record_fallback("cross_entropy_loss", e)
        return None


def cross_entropy_backward(logits, targets):
    """GPU cross-entropy backward. Returns grad_logits or None."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        targets = np.asarray(targets)
        if targets.dtype != np.uint32:
            targets = targets.astype(np.uint32)
        return _core.cross_entropy_backward(dev, _ensure_f32_contiguous(logits), targets)
    except Exception as e:
        _record_fallback("cross_entropy_backward", e)
        return None


# ── Optimizer GPU Ops ────────────────────────────────────────────────────


def adam_update(
    weights,
    grad,
    m,
    v,
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    beta1_t=0.9,
    beta2_t=0.999,
    clear_grad=False,
):
    """GPU Adam step. Returns dict with weights, grad, m, v. None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.adam_update(
            dev,
            _ensure_f32_contiguous(weights),
            _ensure_f32_contiguous(grad),
            _ensure_f32_contiguous(m),
            _ensure_f32_contiguous(v),
            lr,
            beta1,
            beta2,
            eps,
            beta1_t,
            beta2_t,
            clear_grad,
        )
    except Exception as e:
        _record_fallback("adam_update", e)
        return None


def adamw_update(
    weights,
    grad,
    m,
    v,
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.01,
    beta1_t=0.9,
    beta2_t=0.999,
    clear_grad=False,
):
    """GPU AdamW step. Returns dict with weights, grad, m, v. None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.adamw_update(
            dev,
            _ensure_f32_contiguous(weights),
            _ensure_f32_contiguous(grad),
            _ensure_f32_contiguous(m),
            _ensure_f32_contiguous(v),
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            beta1_t,
            beta2_t,
            clear_grad,
        )
    except Exception as e:
        _record_fallback("adamw_update", e)
        return None


# ── Embedding ────────────────────────────────────────────────────────────


def embedding_lookup(token_ids, embeddings):
    """GPU embedding lookup. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        token_ids = np.asarray(token_ids)
        if token_ids.dtype != np.uint32:
            token_ids = token_ids.astype(np.uint32)
        return _core.embedding_lookup(dev, token_ids, _ensure_f32_contiguous(embeddings))
    except Exception as e:
        _record_fallback("embedding_lookup", e)
        return None


# ── KV Cache ────────────────────────────────────────────────────────────


def create_kv_cache(
    max_seq_len,
    num_heads,
    head_dim,
    num_layers,
    compression_ratio=1,
    max_cache_tokens=0,
    use_asymmetric_quant=False,
    value_bits=16,
    cross_layer_sharing=False,
    use_h2o=False,
    use_speculative_eviction=False,
    eviction_threshold=0.1,
):
    """Create a GPU KV cache. Returns KVCache handle or None."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        if max_cache_tokens == 0:
            max_cache_tokens = max_seq_len
        return _core.create_kv_cache(
            dev,
            max_seq_len,
            num_heads,
            head_dim,
            num_layers,
            compression_ratio,
            max_cache_tokens,
            use_asymmetric_quant,
            value_bits,
            cross_layer_sharing,
            use_h2o,
            use_speculative_eviction,
            eviction_threshold,
        )
    except Exception as e:
        _record_fallback("create_kv_cache", e)
        return None


def kv_cache_append(kv_cache, new_keys, new_values):
    """Append new KV pairs to cache. Returns True on success, None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        new_keys = _ensure_f32_contiguous(new_keys)
        new_values = _ensure_f32_contiguous(new_values)
        _core.kv_cache_append(dev, kv_cache, new_keys, new_values)
        return True
    except Exception as e:
        _record_fallback("kv_cache_append", e)
        return None


def kv_cache_decode(kv_cache):
    """Decode KV from compressed cache. Returns dict with keys, values or None."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.kv_cache_decode(dev, kv_cache)
    except Exception as e:
        _record_fallback("kv_cache_decode", e)
        return None


def kv_cache_evict_h2o(kv_cache, attention_scores=None, num_evict=0):
    """Run H2O eviction on KV cache. Returns True on success, None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        scores = _ensure_f32_contiguous(attention_scores) if attention_scores is not None else None
        _core.kv_cache_evict_h2o(dev, kv_cache, scores, num_evict)
        return True
    except Exception as e:
        _record_fallback("kv_cache_evict_h2o", e)
        return None


# ── Tensor Ops (transpose, cat, where, index_select, bmm) ────────────────
# These bridge functions call the GPU shaders compiled from:
#   shaders/tensor-transpose.glsl, tensor-cat.glsl, tensor-where.glsl,
#   tensor-index-select.glsl, tensor-bmm.glsl
# If the C++ binding is unavailable they fall back to numpy so callers never
# need to guard against None — they always get a valid numpy array back.


def tensor_transpose(x):
    """Transpose a 2D matrix on GPU. Falls back to numpy for non-2D or no GPU.

    Args:
        x: 2-D numpy array of shape (rows, cols).

    Returns:
        numpy array of shape (cols, rows).
    """
    x = _ensure_f32_contiguous(x)
    if x.ndim != 2:
        return np.ascontiguousarray(x.T, dtype=np.float32)
    dev = _get_device()
    if dev is None:
        return np.ascontiguousarray(x.T, dtype=np.float32)
    try:
        return _core.tensor_transpose(dev, x)
    except Exception:
        return np.ascontiguousarray(x.T, dtype=np.float32)


def tensor_cat(a, b, axis=-1):
    """Concatenate two tensors along their last axis on GPU.

    Args:
        a: numpy array.
        b: numpy array with same shape except last dim.
        axis: concatenation axis; only last-axis (-1 / ndim-1) uses the GPU
              shader — other axes fall through to numpy.

    Returns:
        numpy array with last dimension equal to a.shape[-1] + b.shape[-1].
    """
    a = _ensure_f32_contiguous(a)
    b = _ensure_f32_contiguous(b)
    # Normalise axis
    ndim = a.ndim
    if axis < 0:
        axis = ndim + axis
    if axis != ndim - 1:
        # Non-last-axis concatenation: numpy fallback
        return np.concatenate([a, b], axis=axis)
    dev = _get_device()
    if dev is None:
        return np.concatenate([a, b], axis=-1)
    try:
        return _core.tensor_cat(dev, a, b)
    except Exception:
        return np.concatenate([a, b], axis=-1)


def tensor_where(cond, a, b):
    """Elementwise conditional select: out[i] = cond[i] > 0 ? a[i] : b[i].

    Args:
        cond: float32 condition array (positive = True).
        a:    float32 array selected when cond > 0.
        b:    float32 array selected when cond <= 0.

    Returns:
        numpy float32 array of same shape.
    """
    cond = _ensure_f32_contiguous(cond)
    a = _ensure_f32_contiguous(a)
    b = _ensure_f32_contiguous(b)
    dev = _get_device()
    if dev is None:
        return np.where(cond > 0, a, b).astype(np.float32)
    try:
        return _core.tensor_where(dev, cond, a, b)
    except Exception:
        return np.where(cond > 0, a, b).astype(np.float32)


def tensor_index_select(x, indices):
    """Gather rows by integer indices: out[i] = x[indices[i]].

    Args:
        x:       2-D float32 array of shape (num_rows, row_size).
        indices: 1-D integer array of row indices to select.

    Returns:
        numpy float32 array of shape (len(indices), row_size).
    """
    x = _ensure_f32_contiguous(x)
    indices = np.asarray(indices, dtype=np.uint32)
    if not indices.flags["C_CONTIGUOUS"]:
        indices = np.ascontiguousarray(indices)
    dev = _get_device()
    if dev is None:
        return x[indices]
    try:
        return _core.tensor_index_select(dev, x, indices)
    except Exception:
        return x[indices]


def tensor_bmm(a, b):
    """Batched matrix multiply: out[i] = a[i] @ b[i].

    Args:
        a: float32 array of shape (batch, M, K).
        b: float32 array of shape (batch, K, N).

    Returns:
        numpy float32 array of shape (batch, M, N).
    """
    a = _ensure_f32_contiguous(a)
    b = _ensure_f32_contiguous(b)
    dev = _get_device()
    if dev is None:
        return np.einsum("bik,bkj->bij", a, b).astype(np.float32)
    try:
        return _core.tensor_bmm(dev, a, b)
    except Exception:
        return np.einsum("bik,bkj->bij", a, b).astype(np.float32)


# ── HDC Packed Binary Hypervector Ops ────────────────────────────────────


def _ensure_uint32_contiguous(arr):
    """Ensure array is uint32 and C-contiguous for the C++ side."""
    arr = np.asarray(arr)
    if arr.dtype != np.uint32:
        arr = arr.astype(np.uint32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def hdc_bind_packed(a, b):
    """Binding of two packed binary hypervectors: element-wise XOR on uint32 arrays.

    Binary hypervectors are packed 32 bits per uint32 word (32x memory compression
    over float32). Binding = XOR, which is its own inverse.

    Args:
        a: uint32 numpy array of shape (num_words,) or (batch, words_per_vec).
        b: uint32 numpy array, same shape as a.

    Returns:
        uint32 numpy array, same shape as a.
    """
    a = _ensure_uint32_contiguous(a)
    b = _ensure_uint32_contiguous(b)
    # numpy fallback (GPU path not yet wired into grilly_core)
    return np.bitwise_xor(a, b)


def hdc_bundle_packed(vectors, words_per_vec):
    """Bundle N packed binary hypervectors via majority vote.

    For each bit position across N vectors, the output bit is 1 if more than
    half the input vectors have that bit set. Ties (even N) are resolved toward 0.

    Args:
        vectors: uint32 numpy array of shape (num_vectors, words_per_vec), laid
                 out as vec0_word0, vec0_word1, ..., vecN_wordK (row-major).
        words_per_vec: int, number of uint32 words per hypervector (dim // 32).

    Returns:
        uint32 numpy array of shape (words_per_vec,).
    """
    vectors = _ensure_uint32_contiguous(vectors)
    num_vectors = vectors.shape[0]
    threshold = num_vectors / 2.0
    # Count set-bits across vectors for each bit position using popcount trick:
    # broadcast each bit mask across all words, then sum
    result = np.zeros(words_per_vec, dtype=np.uint32)
    for bit in range(32):
        mask = np.uint32(1 << bit)
        counts = np.sum((vectors & mask) != 0, axis=0)  # shape (words_per_vec,)
        result |= np.where(counts > threshold, mask, np.uint32(0)).astype(np.uint32)
    return result


def hdc_similarity_packed(query, codebook, dim):
    """Hamming similarity between a packed query and each entry in a packed codebook.

    Hamming similarity = 1 - hamming_distance / dim, where hamming distance is the
    number of differing bits. Uses popcount (bitCount in GLSL) on XOR'd words for
    fast packed computation.

    Args:
        query:    uint32 numpy array of shape (words_per_vec,).
        codebook: uint32 numpy array of shape (num_entries, words_per_vec).
        dim:      int, original hypervector dimension (words_per_vec * 32).

    Returns:
        float32 numpy array of shape (num_entries,) with similarities in [0, 1].
        Value 1.0 = identical vectors, 0.5 = random/uncorrelated, 0.0 = complement.
    """
    query = _ensure_uint32_contiguous(query)
    codebook = _ensure_uint32_contiguous(codebook)
    # XOR each codebook entry with the query, then count differing bits via popcount
    xored = np.bitwise_xor(codebook, query[np.newaxis, :])  # (num_entries, words_per_vec)
    # popcount per word: unpack bits and sum
    hamming = np.zeros(xored.shape[0], dtype=np.int32)
    for shift in range(32):
        hamming += ((xored >> np.uint32(shift)) & np.uint32(1)).astype(np.int32).sum(axis=1)
    return (1.0 - hamming.astype(np.float32) / float(dim)).astype(np.float32)


def hdc_permute_packed(data, words_per_vec, shift):
    """Cyclic bit permutation of a packed binary hypervector.

    Shifts all bits by `shift` positions cyclically within the hypervector.
    Used for positional encoding in HDC — each position in a sequence gets a
    distinct role-filler binding via permutation.

    Args:
        data:         uint32 numpy array of shape (words_per_vec,).
        words_per_vec: int, number of uint32 words (dim // 32).
        shift:        int, number of bit positions to shift (cyclic, mod dim).

    Returns:
        uint32 numpy array of shape (words_per_vec,).
    """
    data = _ensure_uint32_contiguous(data)
    total_bits = words_per_vec * 32
    shift = int(shift) % total_bits
    if shift == 0:
        return data.copy()

    # Unpack all bits into a bool array, rotate, repack
    bits = np.unpackbits(data.view(np.uint8), bitorder="little")  # shape (total_bits,)
    rotated = np.roll(bits, shift)
    # repack back to uint32
    packed_bytes = np.packbits(rotated, bitorder="little")
    return packed_bytes.view(np.uint32)


def hdc_overlap_metrics(query, codebook, dim):
    """Overlap, Jaccard, and overlap-coefficient for packed binary hypervectors.

    GPU shader: hdc-overlap-metrics.glsl (bitCount + subgroupAdd).
    CPU fallback: numpy popcount via bit-shifting.

    Args:
        query:    uint32 numpy array of shape (words_per_vec,).
        codebook: uint32 numpy array of shape (num_entries, words_per_vec).
        dim:      int, original hypervector dimension (words_per_vec * 32).

    Returns:
        Dict with keys:
          'overlap':      float32 (num_entries,) — |A&B| / dim
          'jaccard':      float32 (num_entries,) — |A&B| / |A|B|
          'overlap_coef': float32 (num_entries,) — |A&B| / min(|A|, |B|)
    """
    query = _ensure_uint32_contiguous(query)
    codebook = _ensure_uint32_contiguous(codebook)

    def _popcount_axis1(arr):
        """Popcount each row of a uint32 array."""
        total = np.zeros(arr.shape[0], dtype=np.int32)
        for shift in range(32):
            total += ((arr >> np.uint32(shift)) & np.uint32(1)).astype(np.int32).sum(axis=1)
        return total

    and_bits = np.bitwise_and(codebook, query[np.newaxis, :])
    or_bits = np.bitwise_or(codebook, query[np.newaxis, :])

    count_and = _popcount_axis1(and_bits).astype(np.float32)
    count_or = _popcount_axis1(or_bits).astype(np.float32)

    # Query popcount (same for all entries)
    q_pop = 0
    for shift in range(32):
        q_pop += int(np.sum((query >> np.uint32(shift)) & np.uint32(1)))
    count_a = np.full(codebook.shape[0], q_pop, dtype=np.float32)

    # Per-entry popcount
    count_b = _popcount_axis1(codebook).astype(np.float32)

    overlap = count_and / float(dim)
    jaccard = np.where(count_or > 0, count_and / count_or, 0.0).astype(np.float32)
    min_ab = np.minimum(count_a, count_b)
    overlap_coef = np.where(min_ab > 0, count_and / min_ab, 0.0).astype(np.float32)

    return {
        'overlap': overlap,
        'jaccard': jaccard,
        'overlap_coef': overlap_coef,
    }


def hamming_topk(query, cache, dim, k=10):
    """Return the K nearest entries by Hamming distance from a packed binary cache.

    GPU shader: hamming-topk.glsl (bitCount + subgroupAdd, all distances written).
    CPU fallback: numpy popcount + argpartition.

    Args:
        query: uint32 numpy array of shape (words_per_vec,).
        cache: uint32 numpy array of shape (num_entries, words_per_vec).
        dim:   int, original hypervector dimension.
        k:     int, number of nearest entries to return.

    Returns:
        Dict with keys:
          'indices':    int32 (k,) — indices of the k nearest entries.
          'distances':  int32 (k,) — Hamming distances (ascending).
          'similarities': float32 (k,) — 1 - distance/dim (descending).
    """
    query = _ensure_uint32_contiguous(query)
    cache = _ensure_uint32_contiguous(cache)
    n = cache.shape[0]
    k = min(k, n)

    # Compute all Hamming distances
    xored = np.bitwise_xor(cache, query[np.newaxis, :])
    distances = np.zeros(n, dtype=np.int32)
    for shift in range(32):
        distances += ((xored >> np.uint32(shift)) & np.uint32(1)).astype(np.int32).sum(axis=1)

    # Top-k via argpartition (O(n) average, no full sort)
    if k < n:
        topk_idx = np.argpartition(distances, k)[:k]
    else:
        topk_idx = np.arange(n)

    # Sort the top-k by distance
    sorted_order = np.argsort(distances[topk_idx])
    topk_idx = topk_idx[sorted_order]
    topk_dist = distances[topk_idx]

    return {
        'indices': topk_idx.astype(np.int32),
        'distances': topk_dist.astype(np.int32),
        'similarities': (1.0 - topk_dist.astype(np.float32) / float(dim)),
    }


def moqe_dynamic_quantize(activations, block_size=32):
    """Dynamic block-wise symmetric quantization: FP32 → INT8.

    GPU shader: moqe-dynamic-quant.glsl (subgroupMax for absmax).
    CPU fallback: numpy block-wise quantization.

    Args:
        activations: float32 numpy array of shape (dim,) or (batch, dim).
        block_size:  int, quantization block size (default 32).

    Returns:
        Dict with keys:
          'quantized': int8 array, same shape as activations.
          'scales':    float32 array of shape (num_blocks,) — one scale per block.
    """
    activations = np.asarray(activations, dtype=np.float32)
    flat = activations.ravel()
    n = len(flat)

    # Pad to multiple of block_size
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

    blocks = flat.reshape(-1, block_size)
    absmax = np.max(np.abs(blocks), axis=1)
    absmax = np.where(absmax < 1e-7, 1e-7, absmax)
    scales = absmax / 127.0

    quantized = np.clip(
        np.round(blocks / scales[:, np.newaxis]),
        -127, 127,
    ).astype(np.int8)

    # Remove padding
    quantized = quantized.ravel()[:n].reshape(activations.shape)
    num_blocks = (n + block_size - 1) // block_size
    scales = scales[:num_blocks]

    return {'quantized': quantized, 'scales': scales.astype(np.float32)}


def moqe_fused_gemv(activations, weights_int8, weight_scales, block_size=32):
    """Fused dynamic quantization + GEMV (no VRAM round-trip for activations).

    GPU shader: moqe-fused-gemv.glsl (register-level quant + integer dot + subgroupAdd).
    CPU fallback: quantize activations, integer matmul, scale.

    Args:
        activations:   float32 (dim,) — input activation vector.
        weights_int8:  int8 (out_dim, dim) — pre-quantized weight matrix.
        weight_scales: float32 (out_dim, num_blocks) — per-block weight scales.
        block_size:    int, quantization block size.

    Returns:
        float32 (out_dim,) — output vector.
    """
    activations = np.asarray(activations, dtype=np.float32)
    dim = len(activations)
    out_dim = weights_int8.shape[0]

    # Quantize activations (in a real GPU kernel this stays in registers)
    q = moqe_dynamic_quantize(activations, block_size)
    q_act = q['quantized'].astype(np.int32)
    a_scales = q['scales']

    # Block-wise integer dot product
    num_blocks = len(a_scales)
    output = np.zeros(out_dim, dtype=np.float32)

    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, dim)
        act_block = q_act[start:end]

        for row in range(out_dim):
            w_block = weights_int8[row, start:end].astype(np.int32)
            idot = int(np.dot(act_block, w_block))
            output[row] += float(idot) * a_scales[b] * weight_scales[row, b]

    return output.astype(np.float32)


def moqe_route_and_gemv(activations, choice, expert_weights, expert_scales, block_size=32):
    """MoQE: route to expert, then fused GEMV.

    GPU shader: moqe-fused-gemv-dp4a.glsl (hard routing + DP4a).
    CPU fallback: pick expert, then moqe_fused_gemv.

    Args:
        activations:    float32 (dim,) — input activation vector.
        choice:         int — expert index (0 or 1).
        expert_weights: list of int8 (out_dim, dim) — one per expert.
        expert_scales:  list of float32 (out_dim, num_blocks) — one per expert.
        block_size:     int, quantization block size.

    Returns:
        float32 (out_dim,) — output vector from the chosen expert.
    """
    return moqe_fused_gemv(
        activations,
        expert_weights[choice],
        expert_scales[choice],
        block_size,
    )


# ── Fused MoE (grilly_core moe_* — same lazy device + shaders as other bridge ops) ──


def moe_upload(embed_w, pos_w, expert_ws, router_ws, router_bs, out_w, n_layers, n_experts):
    """Upload MoE weights to GPU; returns an opaque integer handle.

    Uses the shared bridge :func:`_get_device` (lazy init + ``shaders/spv`` load).

    Args:
        embed_w: float32 (vocab, d_model), C-contiguous.
        pos_w: float32 (max_seq, d_model).
        expert_ws: list of length ``n_layers * n_experts``, each (d_model, d_model).
        router_ws: list of length ``n_layers``, each (n_experts, d_model).
        router_bs: list of length ``n_layers``, each (n_experts,).
        out_w: float32 (vocab, d_model) — output projection.
        n_layers, n_experts: int.

    Returns:
        int handle, or ``None`` if the C++ extension / device is unavailable.
    """
    if not _NATIVE or not hasattr(_core, "moe_upload"):
        return None
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.moe_upload(
            dev,
            _ensure_f32_contiguous(embed_w),
            _ensure_f32_contiguous(pos_w),
            expert_ws,
            router_ws,
            router_bs,
            _ensure_f32_contiguous(out_w),
            int(n_layers),
            int(n_experts),
        )
    except Exception as e:
        _record_fallback("moe_upload", e)
        return None


def moe_release(handle):
    """Free GPU resources for a handle from :func:`moe_upload`.

    Returns:
        ``True`` if released, ``False`` if bridge/device unavailable.
    """
    if not _NATIVE or not hasattr(_core, "moe_release"):
        return False
    dev = _get_device()
    if dev is None:
        return False
    try:
        _core.moe_release(dev, int(handle))
        return True
    except Exception as e:
        _record_fallback("moe_release", e)
        return False


def moe_forward(handle, input_ids):
    """Run fused MoE forward; returns logits (seq_len, vocab).

    Args:
        handle: int from :func:`moe_upload`.
        input_ids: int32 1-D (seq_len,), C-contiguous.

    Returns:
        float32 ndarray or ``None`` on failure.
    """
    if not _NATIVE or not hasattr(_core, "moe_forward"):
        return None
    dev = _get_device()
    if dev is None:
        return None
    ids = np.asarray(input_ids, dtype=np.int32)
    if not ids.flags.c_contiguous:
        ids = np.ascontiguousarray(ids, dtype=np.int32)
    try:
        return _core.moe_forward(dev, int(handle), ids)
    except Exception as e:
        _record_fallback("moe_forward", e)
        return None


def moe_update_weights(handle, embed_w, pos_w, expert_ws, router_ws, router_bs, out_w):
    """Re-upload weights in place (same shapes as :func:`moe_upload`).

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    if not _NATIVE or not hasattr(_core, "moe_update_weights"):
        return False
    dev = _get_device()
    if dev is None:
        return False
    try:
        _core.moe_update_weights(
            dev,
            int(handle),
            _ensure_f32_contiguous(embed_w),
            _ensure_f32_contiguous(pos_w),
            expert_ws,
            router_ws,
            router_bs,
            _ensure_f32_contiguous(out_w),
        )
        return True
    except Exception as e:
        _record_fallback("moe_update_weights", e)
        return False


def moe_backward(handle, input_ids, grad_logits):
    """CPU backward for MoE (gradients dict); uses CPU mirrors from upload.

    Args:
        handle: int from :func:`moe_upload`.
        input_ids: int32 (seq_len,) — same sequence as forward.
        grad_logits: float32 (seq_len, vocab).

    Returns:
        dict with keys ``grad_embed``, ``grad_pos``, ``grad_experts``, ``grad_routers_W``,
        ``grad_routers_b``, ``grad_out_w``, or ``None`` on failure.
    """
    if not _NATIVE or not hasattr(_core, "moe_backward"):
        return None
    dev = _get_device()
    if dev is None:
        return None
    ids = np.asarray(input_ids, dtype=np.int32)
    if not ids.flags.c_contiguous:
        ids = np.ascontiguousarray(ids, dtype=np.int32)
    g = _ensure_f32_contiguous(grad_logits)
    try:
        return _core.moe_backward(dev, int(handle), ids, g)
    except Exception as e:
        _record_fallback("moe_backward", e)
        return None


# ── Fused VSA-LM (grilly_core vsa_lm_* — AdditionLinear FFN + sign activation) ──


def vsa_lm_upload(
    embed_w, pos_w,
    ffn_up_patterns, ffn_up_biases, ffn_down_patterns, ffn_down_biases,
    ln_gammas, ln_betas, out_w,
    n_layers, d_model, d_ffn,
):
    """Upload VSA-LM weights to GPU; returns an opaque integer handle.

    Uses the shared bridge :func:`_get_device` (lazy init + ``shaders/spv`` load).

    Args:
        embed_w: float32 (vocab, d_model).
        pos_w: float32 (max_seq, d_model).
        ffn_up_patterns: list of (d_ffn, d_model) float32, length n_layers.
        ffn_up_biases: list of (d_ffn,) float32, length n_layers.
        ffn_down_patterns: list of (d_model, d_ffn) float32, length n_layers.
        ffn_down_biases: list of (d_model,) float32, length n_layers.
        ln_gammas: list of (d_model,) float32, length n_layers.
        ln_betas: list of (d_model,) float32, length n_layers.
        out_w: float32 (vocab, d_model).
        n_layers, d_model, d_ffn: int.

    Returns:
        int handle, or ``None`` if the C++ extension / device is unavailable.
    """
    if not _NATIVE or not hasattr(_core, "vsa_lm_upload"):
        return None
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.vsa_lm_upload(
            dev,
            _ensure_f32_contiguous(embed_w),
            _ensure_f32_contiguous(pos_w),
            ffn_up_patterns, ffn_up_biases,
            ffn_down_patterns, ffn_down_biases,
            ln_gammas, ln_betas,
            _ensure_f32_contiguous(out_w),
            int(n_layers), int(d_model), int(d_ffn),
        )
    except Exception as e:
        _record_fallback("vsa_lm_upload", e)
        return None


def vsa_lm_release(handle):
    """Free GPU resources for a handle from :func:`vsa_lm_upload`.

    Returns:
        ``True`` if released, ``False`` if bridge/device unavailable.
    """
    if not _NATIVE or not hasattr(_core, "vsa_lm_release"):
        return False
    dev = _get_device()
    if dev is None:
        return False
    try:
        _core.vsa_lm_release(dev, int(handle))
        return True
    except Exception as e:
        _record_fallback("vsa_lm_release", e)
        return False


def vsa_lm_forward(handle, input_ids):
    """Run fused VSA-LM forward; returns logits (seq_len, vocab).

    Args:
        handle: int from :func:`vsa_lm_upload`.
        input_ids: int32 1-D (seq_len,), C-contiguous.

    Returns:
        float32 ndarray or ``None`` on failure.
    """
    if not _NATIVE or not hasattr(_core, "vsa_lm_forward"):
        return None
    dev = _get_device()
    if dev is None:
        return None
    ids = np.asarray(input_ids, dtype=np.int32)
    if not ids.flags.c_contiguous:
        ids = np.ascontiguousarray(ids, dtype=np.int32)
    try:
        return _core.vsa_lm_forward(dev, int(handle), ids)
    except Exception as e:
        _record_fallback("vsa_lm_forward", e)
        return None


def vsa_lm_backward(handle, input_ids, grad_logits):
    """CPU backward for VSA-LM (gradients dict); uses CPU mirrors from upload.

    Args:
        handle: int from :func:`vsa_lm_upload`.
        input_ids: int32 (seq_len,) — same sequence as forward.
        grad_logits: float32 (seq_len, vocab).

    Returns:
        dict with keys ``grad_embed``, ``grad_pos``, ``grad_out_w``,
        ``grad_ffn_up_w``, ``grad_ffn_up_b``, ``grad_ffn_down_w``,
        ``grad_ffn_down_b``, ``grad_ln_gamma``, ``grad_ln_beta``,
        or ``None`` on failure.
    """
    if not _NATIVE or not hasattr(_core, "vsa_lm_backward"):
        return None
    dev = _get_device()
    if dev is None:
        return None
    ids = np.asarray(input_ids, dtype=np.int32)
    if not ids.flags.c_contiguous:
        ids = np.ascontiguousarray(ids, dtype=np.int32)
    g = _ensure_f32_contiguous(grad_logits)
    try:
        return _core.vsa_lm_backward(dev, int(handle), ids, g)
    except Exception as e:
        _record_fallback("vsa_lm_backward", e)
        return None


def vsa_lm_update_weights(
    handle, embed_w, pos_w,
    ffn_up_patterns, ffn_up_biases, ffn_down_patterns, ffn_down_biases,
    ln_gammas, ln_betas, out_w,
):
    """Re-upload VSA-LM weights in place (same shapes as :func:`vsa_lm_upload`).

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    if not _NATIVE or not hasattr(_core, "vsa_lm_update_weights"):
        return False
    dev = _get_device()
    if dev is None:
        return False
    try:
        _core.vsa_lm_update_weights(
            dev, int(handle),
            _ensure_f32_contiguous(embed_w),
            _ensure_f32_contiguous(pos_w),
            ffn_up_patterns, ffn_up_biases,
            ffn_down_patterns, ffn_down_biases,
            ln_gammas, ln_betas,
            _ensure_f32_contiguous(out_w),
        )
        return True
    except Exception as e:
        _record_fallback("vsa_lm_update_weights", e)
        return False


# ── Bridge support for functional modules without native kernels ──────────


def faiss_distance(query, vectors, distance_type="l2"):
    """Compute FAISS-style distances with numpy fallback."""
    q = np.asarray(query, dtype=np.float32)
    v = np.asarray(vectors, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if distance_type == "l2":
        diff = q[:, None, :] - v[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32), dtype=np.float32)
    if distance_type == "cosine":
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        v_norm = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        return (1.0 - (q_norm @ v_norm.T)).astype(np.float32)
    return (-(q @ v.T)).astype(np.float32)


def faiss_topk(distances, k):
    """Select top-k nearest neighbors from a distance matrix."""
    d = np.asarray(distances, dtype=np.float32)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    k = max(1, min(int(k), d.shape[1]))
    idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(d, idx, axis=1)
    order = np.argsort(vals, axis=1)
    idx = np.take_along_axis(idx, order, axis=1).astype(np.int32)
    vals = np.take_along_axis(vals, order, axis=1).astype(np.float32)
    return idx, vals


def memory_read(queries, memory_keys, memory_values, temperature=None):
    """Memory read via attention-style lookup."""
    q = np.asarray(queries, dtype=np.float32)
    k = np.asarray(memory_keys, dtype=np.float32)
    v = np.asarray(memory_values, dtype=np.float32)
    temp = float(temperature) if temperature is not None else float(np.sqrt(k.shape[-1]))
    scores = (q @ k.T) / max(temp, 1e-8)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return (weights @ v).astype(np.float32)


def memory_write(new_key, new_value, memory_keys, memory_values, write_index, write_mode=0, blend_factor=0.5):
    """Memory write for key/value banks."""
    keys = np.array(memory_keys, dtype=np.float32, copy=True)
    vals = np.array(memory_values, dtype=np.float32, copy=True)
    idx = int(write_index)
    if write_mode == 1:
        a = float(blend_factor)
        keys[idx] = (1.0 - a) * keys[idx] + a * np.asarray(new_key, dtype=np.float32)
        vals[idx] = (1.0 - a) * vals[idx] + a * np.asarray(new_value, dtype=np.float32)
    else:
        keys[idx] = np.asarray(new_key, dtype=np.float32)
        vals[idx] = np.asarray(new_value, dtype=np.float32)
    return keys, vals


def memory_query_pooling(x, w_query, b_query):
    """Pool sequence to query vectors by mean-pool + projection."""
    x_arr = np.asarray(x, dtype=np.float32)
    pooled = np.mean(x_arr, axis=1)
    return (pooled @ np.asarray(w_query, dtype=np.float32).T + np.asarray(b_query, dtype=np.float32)).astype(np.float32)


def memory_inject_gate(x, memory_context, w_gate, b_gate, w_mem_proj):
    """Gate between token state and projected memory context."""
    x_arr = np.asarray(x, dtype=np.float32)
    mem = np.asarray(memory_context, dtype=np.float32)
    batch, seq_len, dim = x_arr.shape
    mem_expand = np.broadcast_to(mem[:, None, :], (batch, seq_len, dim))
    concat = np.concatenate([x_arr, mem_expand], axis=-1)
    gate = 1.0 / (1.0 + np.exp(-(concat @ np.asarray(w_gate, dtype=np.float32).T + np.asarray(b_gate, dtype=np.float32))))
    mem_proj = mem @ np.asarray(w_mem_proj, dtype=np.float32).T
    mem_proj = np.broadcast_to(mem_proj[:, None, :], (batch, seq_len, dim))
    return ((1.0 - gate) * x_arr + gate * mem_proj).astype(np.float32)


def fisher_info_update(gradients, fisher, momentum=0.9, use_ema=True, reset=False):
    """Update Fisher information estimate."""
    g = np.asarray(gradients, dtype=np.float32)
    f = np.zeros_like(g, dtype=np.float32) if reset else np.asarray(fisher, dtype=np.float32)
    g2 = g * g
    if use_ema:
        return (float(momentum) * f + (1.0 - float(momentum)) * g2).astype(np.float32)
    return (f + g2).astype(np.float32)


def ewc_penalty(current_params, important_params, fisher, lambda_ewc=1.0):
    """Compute scalar EWC penalty."""
    cur = np.asarray(current_params, dtype=np.float32)
    imp = np.asarray(important_params, dtype=np.float32)
    f = np.asarray(fisher, dtype=np.float32)
    diff = cur - imp
    return float(0.5 * float(lambda_ewc) * np.sum(f * diff * diff, dtype=np.float32))


def natural_gradient(gradients, fisher, eps=1e-8):
    """Approximate natural gradient using diagonal Fisher."""
    g = np.asarray(gradients, dtype=np.float32)
    f = np.asarray(fisher, dtype=np.float32)
    return (g / (f + float(eps))).astype(np.float32)


def nlms_predict(x, w, bias=0.0):
    """NLMS prediction."""
    return float(np.dot(np.asarray(x, dtype=np.float32), np.asarray(w, dtype=np.float32)) + float(bias))


def nlms_update(x, y_true, w, bias, mu=0.5, eps=1e-6):
    """NLMS weight update."""
    x_arr = np.asarray(x, dtype=np.float32)
    w_arr = np.asarray(w, dtype=np.float32)
    pred = float(np.dot(x_arr, w_arr) + float(bias))
    err = float(y_true) - pred
    denom = float(np.dot(x_arr, x_arr) + float(eps))
    step = float(mu) * err / denom
    w_new = (w_arr + step * x_arr).astype(np.float32)
    b_new = float(bias + step)
    return w_new, b_new


def whitening_transform(data, mean=None, std=None):
    """Compute/apply whitening transform."""
    x = np.asarray(data, dtype=np.float32)
    m = np.asarray(mean, dtype=np.float32) if mean is not None else np.mean(x, axis=0)
    s = np.asarray(std, dtype=np.float32) if std is not None else np.std(x, axis=0)
    out = (x - m) / (s + 1e-8)
    return out.astype(np.float32), m.astype(np.float32), s.astype(np.float32)


def place_cell(agent_position, field_centers, field_width=1.0, max_rate=20.0, baseline_rate=0.1):
    """Gaussian place-cell tuning curves."""
    pos = np.asarray(agent_position, dtype=np.float32)
    centers = np.asarray(field_centers, dtype=np.float32)
    if pos.ndim == 1:
        pos = pos[None, :]
    diff = pos[:, None, :] - centers[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    rates = float(baseline_rate) + (float(max_rate) - float(baseline_rate)) * np.exp(
        -d2 / (2.0 * float(field_width) * float(field_width) + 1e-8)
    )
    return rates.astype(np.float32).squeeze(0) if np.asarray(agent_position).ndim == 1 else rates.astype(np.float32)


def time_cell(current_time, preferred_times, time_constant=1.0, max_rate=15.0, baseline_rate=0.1, membrane_state=None):
    """Temporal Gaussian tuning with optional EMA membrane state."""
    t = float(current_time)
    pref = np.asarray(preferred_times, dtype=np.float32)
    rates = float(baseline_rate) + (float(max_rate) - float(baseline_rate)) * np.exp(
        -((t - pref) ** 2) / (2.0 * float(time_constant) * float(time_constant) + 1e-8)
    )
    if membrane_state is None:
        mem = rates.astype(np.float32)
    else:
        mem = (0.9 * np.asarray(membrane_state, dtype=np.float32) + 0.1 * rates).astype(np.float32)
    return rates.astype(np.float32), mem


def continuous_to_spikes(continuous, num_timesteps=10, encoding_type=0, projection_weights=None, projection_bias=None):
    """Convert continuous vectors to spike trains."""
    x = np.asarray(continuous, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if projection_weights is not None:
        x = x @ np.asarray(projection_weights, dtype=np.float32).T
        if projection_bias is not None:
            x = x + np.asarray(projection_bias, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    t = int(num_timesteps)
    if encoding_type == 1:
        spikes = np.zeros((x.shape[0], t, x.shape[1]), dtype=np.float32)
        idx = np.minimum((1.0 - x) * (t - 1), t - 1).astype(np.int32)
        b_idx = np.arange(x.shape[0])[:, None]
        f_idx = np.arange(x.shape[1])[None, :]
        spikes[b_idx, idx, f_idx] = 1.0
        return spikes
    if encoding_type == 2:
        phase = np.linspace(0.0, 2.0 * np.pi, t, endpoint=False, dtype=np.float32)
        return ((np.sin(phase[None, :, None] + x[:, None, :] * 2.0 * np.pi) > 0).astype(np.float32))
    return (np.random.rand(x.shape[0], t, x.shape[1]).astype(np.float32) < x[:, None, :]).astype(np.float32)


def spikes_to_continuous(spikes, encoding_type=0, time_window=5, temporal_weights=None, projection_weights=None, projection_bias=None):
    """Convert spike trains to continuous representations."""
    s = np.asarray(spikes, dtype=np.float32)
    if s.ndim != 3:
        raise ValueError(f"spikes must be 3D (batch, time, features), got shape={s.shape}")
    if encoding_type == 1:
        if temporal_weights is None:
            tw = np.linspace(1.0, 0.0, s.shape[1], dtype=np.float32)
        else:
            tw = np.asarray(temporal_weights, dtype=np.float32)
        tw = tw / (np.sum(tw) + 1e-8)
        cont = np.sum(s * tw[None, :, None], axis=1)
    elif encoding_type == 2:
        phase = np.linspace(0.0, 2.0 * np.pi, s.shape[1], endpoint=False, dtype=np.float32)
        cont = np.sum(s * np.sin(phase)[None, :, None], axis=1) / max(s.shape[1], 1)
    else:
        window = max(1, min(int(time_window), s.shape[1]))
        cont = np.mean(s[:, -window:, :], axis=1)
    if projection_weights is not None:
        cont = cont @ np.asarray(projection_weights, dtype=np.float32).T
        if projection_bias is not None:
            cont = cont + np.asarray(projection_bias, dtype=np.float32)
    return cont.astype(np.float32)


def q_similarity(queries):
    """Compute q-similarity (TAPPA metric) for attention queries.

    Measures how similar consecutive queries are in an attention head.
    High q-similarity → predictable/compressible attention pattern.
    Low q-similarity → retrieval head, needs full KV cache.

    Args:
        queries: shape (batch, seq_len, head_dim) or (batch, heads, seq_len, head_dim)

    Returns:
        q_sim: shape (batch,) or (batch, heads) — mean cosine similarity
               between consecutive queries
    """
    try:
        dev = _get_device()
        if dev is not None:
            return dev.q_similarity(queries)
    except Exception:
        pass
    # numpy fallback
    q = np.asarray(queries, dtype=np.float32)
    if q.ndim == 4:
        # (batch, heads, seq, dim) → compute per head
        B, H, S, D = q.shape
        sims = np.zeros((B, H), dtype=np.float32)
        for b in range(B):
            for h in range(H):
                for t in range(S - 1):
                    a = q[b, h, t]
                    b_vec = q[b, h, t + 1]
                    dot = np.dot(a, b_vec)
                    denom = np.linalg.norm(a) * np.linalg.norm(b_vec)
                    sims[b, h] += dot / max(denom, 1e-8)
                sims[b, h] /= max(S - 1, 1)
        return sims
    elif q.ndim == 3:
        B, S, D = q.shape
        sims = np.zeros(B, dtype=np.float32)
        for b_idx in range(B):
            for t in range(S - 1):
                a = q[b_idx, t]
                b_vec = q[b_idx, t + 1]
                dot = np.dot(a, b_vec)
                denom = np.linalg.norm(a) * np.linalg.norm(b_vec)
                sims[b_idx] += dot / max(denom, 1e-8)
            sims[b_idx] /= max(S - 1, 1)
        return sims
    else:
        raise ValueError(f"queries must be 3D or 4D, got {q.ndim}D")


# ── Block Code (NVSA sparse block codes) ─────────────────────────────────


def blockcode_bind(a, b, num_blocks, block_size):
    """Per-block circular convolution: bind two sparse block-code vectors.

    For each block, if a has its hot position at index i and b at index j,
    the result has hot position at (i + j) % block_size.

    Args:
        a: np.ndarray, shape (batch_size, num_blocks * block_size) or (num_blocks * block_size,)
        b: np.ndarray, same shape as a
        num_blocks: int, number of blocks (k)
        block_size: int, size of each block (l)

    Returns:
        np.ndarray, same shape as a — bound block-code vector
    """
    a_in = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b_in = np.atleast_2d(np.asarray(b, dtype=np.float32))
    squeezed = np.ndim(a) == 1
    a_in = _ensure_f32_contiguous(a_in)
    b_in = _ensure_f32_contiguous(b_in)
    batch_size = a_in.shape[0]

    dev = _get_device()
    if dev is not None:
        try:
            result = dev.blockcode_bind(a_in, b_in, num_blocks, block_size)
            if result is not None:
                return result.squeeze(0) if squeezed else result
        except Exception:
            pass

    # Numpy fallback: per-block circular shift
    out = np.zeros_like(a_in)
    for batch_idx in range(batch_size):
        for block_idx in range(num_blocks):
            base = block_idx * block_size
            hot_a = int(np.argmax(a_in[batch_idx, base : base + block_size]))
            hot_b = int(np.argmax(b_in[batch_idx, base : base + block_size]))
            hot_out = (hot_a + hot_b) % block_size
            out[batch_idx, base + hot_out] = 1.0
    return out.squeeze(0) if squeezed else out


def blockcode_unbind(composite, key, num_blocks, block_size):
    """Inverse binding via circular correlation: recover a vector from composite ⊛ key.

    For each block: hot position = (hot_composite - hot_key + block_size) % block_size.

    Args:
        composite: np.ndarray, shape (batch_size, num_blocks * block_size) or flat
        key: np.ndarray, same shape as composite
        num_blocks: int, number of blocks (k)
        block_size: int, size of each block (l)

    Returns:
        np.ndarray, same shape — unbound block-code vector
    """
    c_in = np.atleast_2d(np.asarray(composite, dtype=np.float32))
    k_in = np.atleast_2d(np.asarray(key, dtype=np.float32))
    squeezed = np.ndim(composite) == 1
    c_in = _ensure_f32_contiguous(c_in)
    k_in = _ensure_f32_contiguous(k_in)
    batch_size = c_in.shape[0]

    dev = _get_device()
    if dev is not None:
        try:
            result = dev.blockcode_unbind(c_in, k_in, num_blocks, block_size)
            if result is not None:
                return result.squeeze(0) if squeezed else result
        except Exception:
            pass

    # Numpy fallback
    out = np.zeros_like(c_in)
    for batch_idx in range(batch_size):
        for block_idx in range(num_blocks):
            base = block_idx * block_size
            hot_c = int(np.argmax(c_in[batch_idx, base : base + block_size]))
            hot_k = int(np.argmax(k_in[batch_idx, base : base + block_size]))
            hot_out = (hot_c - hot_k + block_size) % block_size
            out[batch_idx, base + hot_out] = 1.0
    return out.squeeze(0) if squeezed else out


def blockcode_similarity(query, codebook, num_blocks, block_size):
    """Block-code similarity: normalised sum of per-block dot products.

    For one-hot blocks the dot product is 1 if hot positions match, 0 otherwise,
    so the result is the fraction of matching blocks (in [0, 1]).

    Args:
        query: np.ndarray, shape (num_blocks * block_size,) — single query vector
        codebook: np.ndarray, shape (num_entries, num_blocks * block_size)
        num_blocks: int, number of blocks (k)
        block_size: int, size of each block (l)

    Returns:
        np.ndarray, shape (num_entries,) — similarity scores in [0, 1]
    """
    q = _ensure_f32_contiguous(np.asarray(query, dtype=np.float32).ravel())
    cb = _ensure_f32_contiguous(np.atleast_2d(np.asarray(codebook, dtype=np.float32)))
    num_entries = cb.shape[0]

    dev = _get_device()
    if dev is not None:
        try:
            result = dev.blockcode_similarity(q, cb, num_blocks, block_size)
            if result is not None:
                return result
        except Exception:
            pass

    # Numpy fallback
    sims = np.zeros(num_entries, dtype=np.float32)
    for entry_idx in range(num_entries):
        dot_sum = 0.0
        for b in range(num_blocks):
            base = b * block_size
            dot_sum += float(
                np.dot(q[base : base + block_size], cb[entry_idx, base : base + block_size])
            )
        sims[entry_idx] = dot_sum / num_blocks
    return sims


# ── VSA Core (Bitpacked / Full-Vector) ───────────────────────────────────


def vsa_bind(a, b):
    """GPU VSA cyclic binding (circular convolution)."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.vsa_bind(dev, _ensure_f32_contiguous(a), _ensure_f32_contiguous(b))
    except Exception as e:
        _record_fallback("vsa_bind", e)
        return None


def vsa_bundle(a, b):
    """GPU VSA bundling (vector addition)."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.vsa_bundle(dev, _ensure_f32_contiguous(a), _ensure_f32_contiguous(b))
    except Exception as e:
        _record_fallback("vsa_bundle", e)
        return None


def vsa_unbind(composite, key):
    """GPU VSA unbinding (circular correlation)."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.vsa_unbind(dev, _ensure_f32_contiguous(composite), _ensure_f32_contiguous(key))
    except Exception as e:
        _record_fallback("vsa_unbind", e)
        return None


def vsa_bitpack(x):
    """GPU VSA bitpacking: convert float vector to uint32 bitset."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.vsa_bitpack(dev, _ensure_f32_contiguous(x))
    except Exception as e:
        _record_fallback("vsa_bitpack", e)
        return None


def blake3_role(seed_string, dim):
    """Generate a pseudo-random VSA role vector using BLAKE3 hashing."""
    if not _NATIVE or not hasattr(_core, "blake3_role"):
        return None
    try:
        return _core.blake3_role(str(seed_string), int(dim))
    except Exception as e:
        _record_fallback("blake3_role", e)
        return None


# ── Search & Retrieval ───────────────────────────────────────────────────


def hamming_search(query, codebook):
    """GPU Hamming search for nearest bitpacked vector."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        # Codebook should be uint32 (bitpacked)
        return _core.hamming_search(dev, query, codebook)
    except Exception as e:
        _record_fallback("hamming_search", e)
        return None


# ── CubeMind & Cognitive Ops ─────────────────────────────────────────────


def resonator_step(query, codebooks):
    """GPU Resonator Network step for VSA factorisation."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        # codebooks should be a list of ndarrays
        cbs = [_ensure_f32_contiguous(cb) for cb in codebooks]
        return _core.resonator(dev, _ensure_f32_contiguous(query), cbs)
    except Exception as e:
        _record_fallback("resonator_step", e)
        return None


def semantic_assigner(x, semantic_codes):
    """GPU Semantic Assigner: maps continuous features to VSA semantic space."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.semantic_assigner(
            dev, _ensure_f32_contiguous(x), _ensure_f32_contiguous(semantic_codes)
        )
    except Exception as e:
        _record_fallback("semantic_assigner", e)
        return None


# ── Optimized Transformer & Training ─────────────────────────────────────


def flash_attention2(q, k, v, mask=None, scale=None, use_rope=False):
    """GPU Flash Attention 2. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        q = _ensure_f32_contiguous(q)
        k = _ensure_f32_contiguous(k)
        v = _ensure_f32_contiguous(v)
        mask = _ensure_f32_contiguous(mask)
        # Use default scale if None
        s = float(scale) if scale is not None else 1.0 / np.sqrt(q.shape[-1])
        return _core.flash_attention2(dev, q, k, v, mask, s)
    except Exception as e:
        _record_fallback("flash_attention2", e)
        return None


def adamw_update(
    weights, grad, m, v, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01,
    beta1_t=None, beta2_t=None, clear_grad=False
):
    """GPU AdamW optimizer update step."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        # Use provided bias correction factors or compute from t
        b1_t = float(beta1_t) if beta1_t is not None else 1.0
        b2_t = float(beta2_t) if beta2_t is not None else 1.0
        
        return _core.adamw_update(
            dev,
            _ensure_f32_contiguous(weights),
            _ensure_f32_contiguous(grad),
            _ensure_f32_contiguous(m),
            _ensure_f32_contiguous(v),
            float(lr), float(beta1), float(beta2), float(eps),
            float(weight_decay), b1_t, b2_t, bool(clear_grad)
        )
    except Exception as e:
        _record_fallback("adamw_update", e)
        return None
