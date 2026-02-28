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
"""

import os

import numpy as np

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


def _ensure_f32_contiguous(arr):
    """Ensure array is float32 and C-contiguous for the C++ side."""
    if arr is None:
        return None
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
    try:
        x = _ensure_f32_contiguous(x)
        weight = _ensure_f32_contiguous(weight)
        bias = _ensure_f32_contiguous(bias)
        return _core.linear(dev, x, weight, bias)
    except Exception:
        return None


# ── Activations ──────────────────────────────────────────────────────────


def relu(x):
    """GPU ReLU. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.relu(dev, _ensure_f32_contiguous(x))
    except Exception:
        return None


def gelu(x):
    """GPU GELU. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.gelu(dev, _ensure_f32_contiguous(x))
    except Exception:
        return None


def silu(x):
    """GPU SiLU. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.silu(dev, _ensure_f32_contiguous(x))
    except Exception:
        return None


def tanh(x):
    """GPU tanh activation. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.tanh_act(dev, _ensure_f32_contiguous(x))
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
        return None


def softmax(x, dim=-1):
    """GPU Softmax. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.softmax(dev, _ensure_f32_contiguous(x), dim)
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
        return None


def attention_mask(scores, mask=None, causal=True, mask_value=-1e9):
    """GPU attention mask. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        m = _ensure_f32_contiguous(mask) if mask is not None else None
        return _core.attention_mask(dev, _ensure_f32_contiguous(scores), m, causal, mask_value)
    except Exception:
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
    except Exception:
        return None


def attention_concat_heads(mh_output):
    """GPU concat multi-head output. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.attention_concat_heads(dev, _ensure_f32_contiguous(mh_output))
    except Exception:
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
    except Exception:
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
    except Exception:
        return None


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
    except Exception:
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
    except Exception:
        return None


def mean_pool(x):
    """GPU mean pooling over sequence dim: (B,S,D) -> (B,D)."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.mean_pool(dev, _ensure_f32_contiguous(x))
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
        return None


def kv_cache_decode(kv_cache):
    """Decode KV from compressed cache. Returns dict with keys, values or None."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.kv_cache_decode(dev, kv_cache)
    except Exception:
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
    except Exception:
        return None
