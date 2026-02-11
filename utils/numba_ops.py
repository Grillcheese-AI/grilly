"""
Numba-Accelerated Operations for CPU Fallbacks

JIT-compiled operations for when Vulkan shaders are unavailable.
Falls back to pure numpy if numba is not installed.

Performance hierarchy:
1. Vulkan GPU shader (fastest)
2. Numba JIT (fast CPU)
3. Pure numpy (baseline)
"""

import numpy as np

# Try to import numba
try:
    import numba
    from numba import float32, int32, int64, jit, prange

    NUMBA_AVAILABLE = True

    # Configure numba for best performance
    numba.config.THREADING_LAYER = "threadsafe"

except ImportError:
    NUMBA_AVAILABLE = False
    numba = None

    # Create no-op decorator for when numba is unavailable
    def jit(*args, **kwargs):
        """Provide a no-op replacement for numba.jit."""

        def decorator(func):
            """Return the original function unchanged."""
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range


# ============================================================================
# Layer Normalization
# ============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _layernorm_numba(
        x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        """
        Numba-accelerated LayerNorm.

        Normalizes across the last dimension.

        Args:
            x: Input array (..., features)
            gamma: Scale parameter (features,)
            beta: Bias parameter (features,)
            eps: Epsilon for numerical stability

        Returns:
            Normalized array (same shape as x)
        """
        # Flatten all but last dimension
        original_shape = x.shape
        features = x.shape[-1]
        flat_x = x.reshape(-1, features)
        n_samples = flat_x.shape[0]

        output = np.empty_like(flat_x)

        for i in prange(n_samples):
            # Compute mean
            mean = 0.0
            for j in range(features):
                mean += flat_x[i, j]
            mean /= features

            # Compute variance
            var = 0.0
            for j in range(features):
                diff = flat_x[i, j] - mean
                var += diff * diff
            var /= features

            # Normalize and apply affine
            inv_std = 1.0 / np.sqrt(var + eps)
            for j in range(features):
                output[i, j] = (flat_x[i, j] - mean) * inv_std * gamma[j] + beta[j]

        return output.reshape(original_shape)

else:

    def _layernorm_numba(
        x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        """Pure numpy fallback for LayerNorm"""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * gamma + beta


def layernorm(
    x: np.ndarray, gamma: np.ndarray = None, beta: np.ndarray = None, eps: float = 1e-5
) -> np.ndarray:
    """
    LayerNorm with numba acceleration.

    Args:
        x: Input array (..., features)
        gamma: Scale parameter (features,) - defaults to ones
        beta: Bias parameter (features,) - defaults to zeros
        eps: Epsilon for numerical stability

    Returns:
        Normalized array
    """
    features = x.shape[-1]

    if gamma is None:
        gamma = np.ones(features, dtype=np.float32)
    if beta is None:
        beta = np.zeros(features, dtype=np.float32)

    # Ensure correct dtypes
    x = np.ascontiguousarray(x, dtype=np.float32)
    gamma = np.ascontiguousarray(gamma, dtype=np.float32)
    beta = np.ascontiguousarray(beta, dtype=np.float32)

    return _layernorm_numba(x, gamma, beta, eps)


# ============================================================================
# Softmax
# ============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _softmax_numba(x: np.ndarray) -> np.ndarray:
        """
        Numba-accelerated softmax along last axis.

        Uses numerically stable computation (subtract max).
        """
        original_shape = x.shape
        features = x.shape[-1]
        flat_x = x.reshape(-1, features)
        n_samples = flat_x.shape[0]

        output = np.empty_like(flat_x)

        for i in prange(n_samples):
            # Find max for numerical stability
            max_val = flat_x[i, 0]
            for j in range(1, features):
                if flat_x[i, j] > max_val:
                    max_val = flat_x[i, j]

            # Compute exp and sum
            exp_sum = 0.0
            for j in range(features):
                output[i, j] = np.exp(flat_x[i, j] - max_val)
                exp_sum += output[i, j]

            # Normalize
            inv_sum = 1.0 / exp_sum
            for j in range(features):
                output[i, j] *= inv_sum

        return output.reshape(original_shape)

else:

    def _softmax_numba(x: np.ndarray) -> np.ndarray:
        """Pure numpy fallback for softmax"""
        x_max = x.max(axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax with numba acceleration.

    Args:
        x: Input array
        axis: Axis to compute softmax (default: -1)

    Returns:
        Softmax probabilities
    """
    x = np.ascontiguousarray(x, dtype=np.float32)

    if axis == -1 or axis == x.ndim - 1:
        return _softmax_numba(x)
    else:
        # Move axis to end, compute, move back
        x = np.moveaxis(x, axis, -1)
        result = _softmax_numba(x)
        return np.moveaxis(result, -1, axis)


# ============================================================================
# Linear (Matrix Multiply + Bias)
# ============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _linear_numba(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """
        Numba-accelerated linear layer.

        Computes: output = x @ weight.T + bias

        Args:
            x: Input (batch, in_features) or (in_features,)
            weight: Weight matrix (out_features, in_features)
            bias: Bias vector (out_features,) or None

        Returns:
            Output (batch, out_features) or (out_features,)
        """
        out_features, in_features = weight.shape

        if x.ndim == 1:
            output = np.zeros(out_features, dtype=np.float32)
            for j in prange(out_features):
                acc = 0.0
                for k in range(in_features):
                    acc += x[k] * weight[j, k]
                output[j] = acc + bias[j]
            return output
        else:
            batch_size = x.shape[0]
            output = np.zeros((batch_size, out_features), dtype=np.float32)

            for b in prange(batch_size):
                for j in range(out_features):
                    acc = 0.0
                    for k in range(in_features):
                        acc += x[b, k] * weight[j, k]
                    output[b, j] = acc + bias[j]

            return output

else:

    def _linear_numba(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Pure numpy fallback for linear"""
        return x @ weight.T + bias


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None) -> np.ndarray:
    """
    Linear layer with numba acceleration.

    Args:
        x: Input array (..., in_features)
        weight: Weight matrix (out_features, in_features)
        bias: Bias vector (out_features,) - defaults to zeros

    Returns:
        Output array (..., out_features)
    """
    out_features = weight.shape[0]

    if bias is None:
        bias = np.zeros(out_features, dtype=np.float32)

    # Handle >2D inputs by flattening
    original_shape = x.shape
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])

    x = np.ascontiguousarray(x, dtype=np.float32)
    weight = np.ascontiguousarray(weight, dtype=np.float32)
    bias = np.ascontiguousarray(bias, dtype=np.float32)

    result = _linear_numba(x, weight, bias)

    # Restore shape
    if len(original_shape) > 2:
        result = result.reshape(*original_shape[:-1], out_features)

    return result


# ============================================================================
# GELU Activation
# ============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _gelu_numba(x: np.ndarray) -> np.ndarray:
        """
        Numba-accelerated GELU activation.

        Uses approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        """
        output = np.empty_like(x)
        flat_x = x.ravel()
        flat_out = output.ravel()

        sqrt_2_over_pi = np.float32(0.7978845608)  # sqrt(2/pi)
        coef = np.float32(0.044715)

        for i in prange(len(flat_x)):
            xi = flat_x[i]
            inner = sqrt_2_over_pi * (xi + coef * xi * xi * xi)
            flat_out[i] = 0.5 * xi * (1.0 + np.tanh(inner))

        return output

else:

    def _gelu_numba(x: np.ndarray) -> np.ndarray:
        """Pure numpy fallback for GELU"""
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU activation with numba acceleration.

    Args:
        x: Input array

    Returns:
        GELU(x)
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    return _gelu_numba(x)


# ============================================================================
# SiLU / Swish Activation
# ============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _silu_numba(x: np.ndarray) -> np.ndarray:
        """
        Numba-accelerated SiLU/Swish activation.

        SiLU(x) = x * sigmoid(x)
        """
        output = np.empty_like(x)
        flat_x = x.ravel()
        flat_out = output.ravel()

        for i in prange(len(flat_x)):
            xi = flat_x[i]
            sigmoid_xi = 1.0 / (1.0 + np.exp(-xi))
            flat_out[i] = xi * sigmoid_xi

        return output

else:

    def _silu_numba(x: np.ndarray) -> np.ndarray:
        """Pure numpy fallback for SiLU"""
        return x / (1.0 + np.exp(-x))


def silu(x: np.ndarray) -> np.ndarray:
    """
    SiLU/Swish activation with numba acceleration.

    Args:
        x: Input array

    Returns:
        SiLU(x) = x * sigmoid(x)
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    return _silu_numba(x)


# ============================================================================
# ReLU Activation
# ============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _relu_numba(x: np.ndarray) -> np.ndarray:
        """Numba-accelerated ReLU"""
        output = np.empty_like(x)
        flat_x = x.ravel()
        flat_out = output.ravel()

        for i in prange(len(flat_x)):
            flat_out[i] = max(0.0, flat_x[i])

        return output

else:

    def _relu_numba(x: np.ndarray) -> np.ndarray:
        """Pure numpy fallback for ReLU"""
        return np.maximum(0, x)


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation with numba acceleration.

    Args:
        x: Input array

    Returns:
        max(0, x)
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    return _relu_numba(x)


# ============================================================================
# GCU (Growing Cosine Unit) Activation
# ============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _gcu_numba(x: np.ndarray) -> np.ndarray:
        """
        Numba-accelerated GCU activation.

        GCU(x) = x * cos(x)
        Oscillatory activation for neuromorphic systems.
        """
        output = np.empty_like(x)
        flat_x = x.ravel()
        flat_out = output.ravel()

        for i in prange(len(flat_x)):
            flat_out[i] = flat_x[i] * np.cos(flat_x[i])

        return output
else:

    def _gcu_numba(x: np.ndarray) -> np.ndarray:
        """Fallback GCU (pure numpy)"""
        return x * np.cos(x)


def gcu(x: np.ndarray) -> np.ndarray:
    """
    GCU (Growing Cosine Unit) activation with numba acceleration.

    Args:
        x: Input array

    Returns:
        x * cos(x)
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    return _gcu_numba(x)


# ============================================================================
# RoSwish (Rotating Swish) Activation
# ============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _roswish_numba(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """
        Numba-accelerated RoSwish activation.

        RoSwish(x) = (x + α) * sigmoid(β * x) - 0.5 * α
        Learnable parameters for adaptive gating.
        """
        output = np.empty_like(x)
        flat_x = x.ravel()
        flat_out = output.ravel()

        for i in prange(len(flat_x)):
            x_val = flat_x[i]
            # Numerically stable sigmoid
            beta_x = beta * x_val
            if beta_x >= 0.0:
                sigmoid_bx = 1.0 / (1.0 + np.exp(-beta_x))
            else:
                exp_bx = np.exp(beta_x)
                sigmoid_bx = exp_bx / (1.0 + exp_bx)

            flat_out[i] = (x_val + alpha) * sigmoid_bx - 0.5 * alpha

        return output
else:

    def _roswish_numba(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """Fallback RoSwish (pure numpy)"""
        sigmoid_bx = 1.0 / (1.0 + np.exp(-beta * x))
        return (x + alpha) * sigmoid_bx - 0.5 * alpha


def roswish(x: np.ndarray, alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    """
    RoSwish (Rotating Swish) activation with numba acceleration.

    Args:
        x: Input array
        alpha: Rotation parameter (default: 1.0)
        beta: Gating parameter (default: 1.0)

    Returns:
        (x + α) * sigmoid(β * x) - 0.5 * α
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    return _roswish_numba(x, float(alpha), float(beta))


# ============================================================================
# SwiGLU (Swish-Gated Linear Unit) Activation
# ============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _swiglu_numba(x: np.ndarray) -> np.ndarray:
        """
        Numba-accelerated SwiGLU activation.

        SwiGLU: Split input into [x1, x2], output = x1 * silu(x2)
        Input shape: (..., 2*hidden_dim)
        Output shape: (..., hidden_dim)
        """
        original_shape = x.shape
        # Reshape to (..., 2*hidden_dim)
        flat_x = x.reshape(-1, original_shape[-1])
        batch_size = flat_x.shape[0]
        full_dim = flat_x.shape[1]
        hidden_dim = full_dim // 2

        output = np.empty((batch_size, hidden_dim), dtype=np.float32)

        for i in prange(batch_size):
            for j in range(hidden_dim):
                x1 = flat_x[i, j]
                x2 = flat_x[i, j + hidden_dim]

                # SiLU(x2) = x2 * sigmoid(x2)
                if x2 >= 0.0:
                    sigmoid_x2 = 1.0 / (1.0 + np.exp(-x2))
                else:
                    exp_x2 = np.exp(x2)
                    sigmoid_x2 = exp_x2 / (1.0 + exp_x2)

                silu_x2 = x2 * sigmoid_x2
                output[i, j] = x1 * silu_x2

        # Reshape output to match input shape (replacing last dim)
        output_shape = original_shape[:-1] + (hidden_dim,)
        return output.reshape(output_shape)
else:

    def _swiglu_numba(x: np.ndarray) -> np.ndarray:
        """Fallback SwiGLU (pure numpy)"""
        hidden_dim = x.shape[-1] // 2
        x1 = x[..., :hidden_dim]
        x2 = x[..., hidden_dim:]
        sigmoid_x2 = 1.0 / (1.0 + np.exp(-x2))
        silu_x2 = x2 * sigmoid_x2
        return x1 * silu_x2


def swiglu(x: np.ndarray) -> np.ndarray:
    """
    SwiGLU (Swish-Gated Linear Unit) activation with numba acceleration.

    Used in LLaMA, PaLM, Mistral transformer FFN layers.

    Args:
        x: Input array of shape (..., 2*hidden_dim)

    Returns:
        Output array of shape (..., hidden_dim)
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    return _swiglu_numba(x)


# ============================================================================
# Attention Score Computation
# ============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _attention_scores_numba(q: np.ndarray, k: np.ndarray, scale: float) -> np.ndarray:
        """
        Numba-accelerated attention score computation.

        Computes: scores = (Q @ K.T) / scale

        Args:
            q: Query (batch, heads, seq_q, head_dim)
            k: Key (batch, heads, seq_k, head_dim)
            scale: Scale factor (usually sqrt(head_dim))

        Returns:
            Attention scores (batch, heads, seq_q, seq_k)
        """
        batch, heads, seq_q, head_dim = q.shape
        seq_k = k.shape[2]

        scores = np.zeros((batch, heads, seq_q, seq_k), dtype=np.float32)
        inv_scale = 1.0 / scale

        for b in prange(batch):
            for h in range(heads):
                for i in range(seq_q):
                    for j in range(seq_k):
                        acc = 0.0
                        for d in range(head_dim):
                            acc += q[b, h, i, d] * k[b, h, j, d]
                        scores[b, h, i, j] = acc * inv_scale

        return scores

else:

    def _attention_scores_numba(q: np.ndarray, k: np.ndarray, scale: float) -> np.ndarray:
        """Pure numpy fallback for attention scores"""
        return np.matmul(q, k.transpose(0, 1, 3, 2)) / scale


def attention_scores(q: np.ndarray, k: np.ndarray, scale: float = None) -> np.ndarray:
    """
    Compute attention scores with numba acceleration.

    Args:
        q: Query (batch, heads, seq_q, head_dim)
        k: Key (batch, heads, seq_k, head_dim)
        scale: Scale factor (default: sqrt(head_dim))

    Returns:
        Attention scores (batch, heads, seq_q, seq_k)
    """
    if scale is None:
        scale = np.sqrt(q.shape[-1])

    q = np.ascontiguousarray(q, dtype=np.float32)
    k = np.ascontiguousarray(k, dtype=np.float32)

    return _attention_scores_numba(q, k, scale)


# ============================================================================
# Embedding Lookup
# ============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, cache=True)
    def _embedding_lookup_numba(indices: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """
        Numba-accelerated embedding lookup.

        Args:
            indices: Token indices (batch, seq_len) - int32
            weight: Embedding table (vocab_size, embed_dim)

        Returns:
            Embeddings (batch, seq_len, embed_dim)
        """
        batch_size, seq_len = indices.shape
        embed_dim = weight.shape[1]

        output = np.empty((batch_size, seq_len, embed_dim), dtype=np.float32)

        for b in prange(batch_size):
            for s in range(seq_len):
                idx = indices[b, s]
                for d in range(embed_dim):
                    output[b, s, d] = weight[idx, d]

        return output

else:

    def _embedding_lookup_numba(indices: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Pure numpy fallback for embedding lookup"""
        return weight[indices]


def embedding_lookup(indices: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Embedding lookup with numba acceleration.

    Args:
        indices: Token indices (...) - integer array
        weight: Embedding table (vocab_size, embed_dim)

    Returns:
        Embeddings (..., embed_dim)
    """
    original_shape = indices.shape
    indices = indices.reshape(-1) if indices.ndim == 1 else indices.reshape(indices.shape[0], -1)

    if indices.ndim == 1:
        # Single sequence
        indices = indices.reshape(1, -1)
        result = _embedding_lookup_numba(indices.astype(np.int32), weight.astype(np.float32))
        return result.reshape(*original_shape, -1)
    else:
        result = _embedding_lookup_numba(indices.astype(np.int32), weight.astype(np.float32))
        return result.reshape(*original_shape, -1)


# ============================================================================
# RoPE (Rotary Position Embeddings)
# ============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _rope_numba(
        q_or_k: np.ndarray, position_ids: np.ndarray, rope_base: float, rope_scaling: float
    ) -> np.ndarray:
        """
        Numba-accelerated RoPE using PyTorch's rotate_half approach.

        ModernBERT uses: q_embed = (q * cos) + (rotate_half(q) * sin)
        where rotate_half(q) = [-q[head_dim//2:], q[:head_dim//2]]

        Args:
            q_or_k: Query or Key tensor (batch, seq_len, num_heads, head_dim)
            position_ids: Position indices (batch, seq_len)
            rope_base: Base frequency for RoPE
            rope_scaling: Scaling factor for extended context

        Returns:
            Rotated Q or K tensor (same shape)
        """
        batch_size, seq_len, num_heads, head_dim = q_or_k.shape
        half_dim = head_dim // 2

        result = np.empty_like(q_or_k)

        # Precompute inv_freq
        inv_freq = np.empty(half_dim, dtype=np.float32)
        for i in range(half_dim):
            inv_freq[i] = 1.0 / (rope_base ** (float(i * 2) / head_dim))

        for b in prange(batch_size):
            for s in range(seq_len):
                pos = float(position_ids[b, s]) / rope_scaling
                for h in range(num_heads):
                    # Apply RoPE with rotate_half
                    for d in range(half_dim):
                        freq = pos * inv_freq[d]
                        cos_val = np.cos(freq)
                        sin_val = np.sin(freq)

                        # First half: uses d and d + half_dim
                        qk_d = q_or_k[b, s, h, d]
                        qk_d_half = q_or_k[b, s, h, d + half_dim]

                        # rotate_half: [-second_half, first_half]
                        # For position d: rotated = -qk_d_half
                        # For position d+half_dim: rotated = qk_d
                        result[b, s, h, d] = qk_d * cos_val - qk_d_half * sin_val
                        result[b, s, h, d + half_dim] = qk_d_half * cos_val + qk_d * sin_val

        return result

else:

    def _rope_numba(
        q_or_k: np.ndarray, position_ids: np.ndarray, rope_base: float, rope_scaling: float
    ) -> np.ndarray:
        """Pure numpy fallback for RoPE"""
        batch_size, seq_len, num_heads, head_dim = q_or_k.shape
        half_dim = head_dim // 2

        # Compute inv_freq
        inv_freq = 1.0 / (rope_base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

        result = np.empty_like(q_or_k)

        for b in range(batch_size):
            for s in range(seq_len):
                pos = float(position_ids[b, s]) / rope_scaling
                freqs = pos * inv_freq
                cos_vals = np.cos(freqs)
                sin_vals = np.sin(freqs)

                for h in range(num_heads):
                    qk_first = q_or_k[b, s, h, :half_dim]
                    qk_second = q_or_k[b, s, h, half_dim:]

                    result[b, s, h, :half_dim] = qk_first * cos_vals - qk_second * sin_vals
                    result[b, s, h, half_dim:] = qk_second * cos_vals + qk_first * sin_vals

        return result


def rope(
    q_or_k: np.ndarray,
    position_ids: np.ndarray = None,
    rope_base: float = 10000.0,
    rope_scaling: float = 1.0,
) -> np.ndarray:
    """
    Apply RoPE (Rotary Position Embeddings) with numba acceleration.

    Args:
        q_or_k: Query or Key tensor (batch, seq_len, num_heads, head_dim)
        position_ids: Position indices (batch, seq_len). If None, uses [0, 1, 2, ...]
        rope_base: Base frequency for RoPE (default: 10000.0)
        rope_scaling: Scaling factor for extended context (default: 1.0)

    Returns:
        Rotated Q or K tensor (same shape)
    """
    batch_size, seq_len = q_or_k.shape[:2]

    if position_ids is None:
        position_ids = np.arange(seq_len, dtype=np.int32)
        position_ids = np.tile(position_ids, (batch_size, 1))

    q_or_k = np.ascontiguousarray(q_or_k, dtype=np.float32)
    position_ids = np.ascontiguousarray(position_ids, dtype=np.int32)

    return _rope_numba(q_or_k, position_ids, rope_base, rope_scaling)


# ============================================================================
# Prosody Modulation
# ============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _prosody_modulation_numba(
        attention_scores: np.ndarray,
        prosody_features: np.ndarray,
        prosody_weights: np.ndarray,
        prosody_strength: float,
    ) -> np.ndarray:
        """
        Numba-accelerated prosody modulation.

        Args:
            attention_scores: Attention scores (batch, num_heads, seq_len, seq_len)
            prosody_features: Prosody features (batch, seq_len, prosody_dim)
            prosody_weights: Prosody projection weights (num_heads, prosody_dim)
            prosody_strength: Modulation strength

        Returns:
            Modulated attention scores (same shape)
        """
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        prosody_dim = prosody_features.shape[2]

        result = attention_scores.copy()

        for b in prange(batch_size):
            for h in range(num_heads):
                for q_pos in range(seq_len):
                    for k_pos in range(seq_len):
                        # Compute prosody bias from key position
                        prosody_bias = 0.0
                        for d in range(prosody_dim):
                            prosody_bias += prosody_features[b, k_pos, d] * prosody_weights[h, d]

                        # Apply modulation
                        result[b, h, q_pos, k_pos] += prosody_strength * prosody_bias

        return result

else:

    def _prosody_modulation_numba(
        attention_scores: np.ndarray,
        prosody_features: np.ndarray,
        prosody_weights: np.ndarray,
        prosody_strength: float,
    ) -> np.ndarray:
        """Pure numpy fallback for prosody modulation"""
        batch_size, num_heads, seq_len, _ = attention_scores.shape

        # prosody_features: (batch, seq_len, prosody_dim)
        # prosody_weights: (num_heads, prosody_dim)
        # Compute: prosody_bias = prosody_features @ prosody_weights.T -> (batch, seq_len, num_heads)
        prosody_bias = np.einsum("bsd,hd->bsh", prosody_features, prosody_weights)

        # Broadcast to attention shape: (batch, num_heads, seq_len, seq_len)
        # prosody_bias[:, :, h] applies to all query positions for head h at key position
        prosody_bias = prosody_bias.transpose(0, 2, 1)  # (batch, num_heads, seq_len)
        prosody_bias = prosody_bias[:, :, np.newaxis, :]  # (batch, num_heads, 1, seq_len)

        return attention_scores + prosody_strength * prosody_bias


def prosody_modulation(
    attention_scores: np.ndarray,
    prosody_features: np.ndarray,
    prosody_weights: np.ndarray,
    prosody_strength: float = 0.3,
) -> np.ndarray:
    """
    Apply prosody modulation to attention scores with numba acceleration.

    Args:
        attention_scores: Attention scores (batch, num_heads, seq_len, seq_len)
        prosody_features: Prosody features (batch, seq_len, prosody_dim)
        prosody_weights: Prosody projection weights (num_heads, prosody_dim)
        prosody_strength: Modulation strength (default: 0.3)

    Returns:
        Modulated attention scores (same shape)
    """
    attention_scores = np.ascontiguousarray(attention_scores, dtype=np.float32)
    prosody_features = np.ascontiguousarray(prosody_features, dtype=np.float32)
    prosody_weights = np.ascontiguousarray(prosody_weights, dtype=np.float32)

    return _prosody_modulation_numba(
        attention_scores, prosody_features, prosody_weights, prosody_strength
    )


# ============================================================================
# Attention Output (Weighted Sum)
# ============================================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _attention_output_numba(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Numba-accelerated attention output computation.

        Computes: output = weights @ values

        Args:
            weights: Attention weights (batch, num_heads, seq_len, seq_len)
            values: Value tensor (batch, num_heads, seq_len, head_dim)

        Returns:
            Attention output (batch, num_heads, seq_len, head_dim)
        """
        batch, num_heads, seq_len, _ = weights.shape
        head_dim = values.shape[3]

        output = np.zeros((batch, num_heads, seq_len, head_dim), dtype=np.float32)

        for b in prange(batch):
            for h in range(num_heads):
                for q in range(seq_len):
                    for d in range(head_dim):
                        acc = 0.0
                        for k in range(seq_len):
                            acc += weights[b, h, q, k] * values[b, h, k, d]
                        output[b, h, q, d] = acc

        return output

else:

    def _attention_output_numba(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Pure numpy fallback for attention output"""
        return np.matmul(weights, values)


def attention_output(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Compute attention output with numba acceleration.

    Args:
        weights: Attention weights (batch, num_heads, seq_len, seq_len)
        values: Value tensor (batch, num_heads, seq_len, head_dim)

    Returns:
        Attention output (batch, num_heads, seq_len, head_dim)
    """
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    values = np.ascontiguousarray(values, dtype=np.float32)

    return _attention_output_numba(weights, values)


# ============================================================================
# Utility: Check if numba is available
# ============================================================================


def is_numba_available() -> bool:
    """Check if numba is available for JIT compilation"""
    return NUMBA_AVAILABLE


def get_backend_info() -> dict:
    """Get information about available backends"""
    info = {
        "numba_available": NUMBA_AVAILABLE,
        "backend": "numba" if NUMBA_AVAILABLE else "numpy",
    }

    if NUMBA_AVAILABLE:
        info["numba_version"] = numba.__version__
        info["threading_layer"] = numba.config.THREADING_LAYER

    return info
