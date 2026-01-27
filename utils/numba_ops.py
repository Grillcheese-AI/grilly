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
from typing import Tuple, Optional

# Try to import numba
try:
    import numba
    from numba import jit, prange, float32, int32, int64
    NUMBA_AVAILABLE = True

    # Configure numba for best performance
    numba.config.THREADING_LAYER = 'threadsafe'

except ImportError:
    NUMBA_AVAILABLE = False
    numba = None

    # Create no-op decorator for when numba is unavailable
    def jit(*args, **kwargs):
        def decorator(func):
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
    def _layernorm_numba(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                         eps: float = 1e-5) -> np.ndarray:
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
    def _layernorm_numba(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                         eps: float = 1e-5) -> np.ndarray:
        """Pure numpy fallback for LayerNorm"""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * gamma + beta


def layernorm(x: np.ndarray, gamma: np.ndarray = None, beta: np.ndarray = None,
              eps: float = 1e-5) -> np.ndarray:
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
    def _linear_numba(x: np.ndarray, weight: np.ndarray,
                      bias: np.ndarray) -> np.ndarray:
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
    def _linear_numba(x: np.ndarray, weight: np.ndarray,
                      bias: np.ndarray) -> np.ndarray:
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
# Attention Score Computation
# ============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _attention_scores_numba(q: np.ndarray, k: np.ndarray,
                                 scale: float) -> np.ndarray:
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
    def _attention_scores_numba(q: np.ndarray, k: np.ndarray,
                                 scale: float) -> np.ndarray:
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
# Utility: Check if numba is available
# ============================================================================

def is_numba_available() -> bool:
    """Check if numba is available for JIT compilation"""
    return NUMBA_AVAILABLE


def get_backend_info() -> dict:
    """Get information about available backends"""
    info = {
        'numba_available': NUMBA_AVAILABLE,
        'backend': 'numba' if NUMBA_AVAILABLE else 'numpy',
    }

    if NUMBA_AVAILABLE:
        info['numba_version'] = numba.__version__
        info['threading_layer'] = numba.config.THREADING_LAYER

    return info
