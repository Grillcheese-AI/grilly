"""
Functional Embedding Operations

Uses: embedding-lookup.glsl, embedding-normalize.glsl, embedding-position.glsl,
      embedding-pool.glsl, embedding-ffn.glsl, embedding-attention.glsl
"""

import numpy as np


def _get_backend():
    """Get backend instance"""
    try:
        from ..backend.compute import Compute

        return Compute()
    except Exception:
        return None


def embedding_lookup(weight: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Lookup embeddings from weight matrix.

    Uses: embedding-lookup.glsl

    Args:
        weight: Embedding weight matrix (num_embeddings, embedding_dim)
        indices: Token indices (batch, seq_len) or (batch,)

    Returns:
        Embeddings (batch, seq_len, embedding_dim) or (batch, embedding_dim)
    """
    from grilly import Compute

    backend = Compute()
    # Backend expects (token_ids, embedding_table) - note the order!
    return backend.embedding_lookup(indices, weight)


def embedding_normalize(embeddings: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize embeddings.

    Uses: embedding-normalize.glsl

    Args:
        embeddings: Embeddings (batch, seq_len, dim) or (batch, dim)
        eps: Small constant for numerical stability

    Returns:
        Normalized embeddings (same shape)
    """
    backend = _get_backend()

    # Try GPU shader if available
    if backend and hasattr(backend, "shaders") and "embedding-normalize" in backend.shaders:
        try:
            # GPU embedding normalization would go here
            # For now, use CPU fallback
            pass
        except Exception:
            pass  # Fall back to CPU

    # CPU fallback
    norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / (norm + eps)


def embedding_position(
    embeddings: np.ndarray, max_seq_len: int = 512, dim: int | None = None
) -> np.ndarray:
    """
    Add positional embeddings.

    Uses: embedding-position.glsl

    Args:
        embeddings: Input embeddings (batch, seq_len, dim)
        max_seq_len: Maximum sequence length
        dim: Embedding dimension (inferred from input if None)

    Returns:
        Embeddings with positional encoding (same shape)
    """
    backend = _get_backend()

    # Try GPU shader if available
    if backend and hasattr(backend, "shaders") and "embedding-position" in backend.shaders:
        try:
            # GPU positional encoding would go here
            # For now, use CPU fallback
            pass
        except Exception:
            pass  # Fall back to CPU

    # CPU fallback (sinusoidal positional encoding)
    batch_size, seq_len, dim = embeddings.shape

    # Create positional encoding
    pos_enc = np.zeros((seq_len, dim), dtype=np.float32)
    for pos in range(seq_len):
        for i in range(0, dim, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** (i / dim)))
            if i + 1 < dim:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / dim)))

    return embeddings + pos_enc[None, :, :]


def embedding_pool(embeddings: np.ndarray, pool_type: str = "mean") -> np.ndarray:
    """
    Pool embeddings over sequence dimension.

    Uses: embedding-pool.glsl

    Args:
        embeddings: Embeddings (batch, seq_len, dim)
        pool_type: Pooling type ('mean', 'max', 'sum')

    Returns:
        Pooled embeddings (batch, dim)
    """
    backend = _get_backend()

    # Try GPU shader if available
    if backend and hasattr(backend, "shaders") and "embedding-pool" in backend.shaders:
        try:
            # GPU embedding pooling would go here
            # For now, use CPU fallback
            pass
        except Exception:
            pass  # Fall back to CPU

    # CPU fallback
    if pool_type == "mean":
        return embeddings.mean(axis=1)
    elif pool_type == "max":
        return embeddings.max(axis=1)
    elif pool_type == "sum":
        return embeddings.sum(axis=1)
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")


def embedding_ffn(
    embeddings: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    activation: str = "gelu",
) -> np.ndarray:
    """
    Apply feed-forward network to embeddings.

    Uses: embedding-ffn.glsl

    Args:
        embeddings: Input embeddings (batch, seq_len, dim)
        W1: First layer weights (ffn_dim, dim)
        b1: First layer bias (ffn_dim,)
        W2: Second layer weights (dim, ffn_dim)
        b2: Second layer bias (dim,)
        activation: Activation function ('gelu', 'relu', 'silu')

    Returns:
        Output embeddings (batch, seq_len, dim)
    """
    backend = _get_backend()

    # Try GPU shader if available
    if backend and hasattr(backend, "shaders") and "embedding-ffn" in backend.shaders:
        try:
            # GPU embedding FFN would go here
            # For now, use CPU fallback
            pass
        except Exception:
            pass  # Fall back to CPU

    # CPU fallback
    x = embeddings @ W1.T + b1

    if activation == "gelu":
        x = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    elif activation == "relu":
        x = np.maximum(0, x)
    elif activation == "silu":
        x = x * (1 / (1 + np.exp(-x)))
    else:
        raise ValueError(f"Unknown activation: {activation}")

    return x @ W2.T + b2


def embedding_attention(
    embeddings: np.ndarray, num_heads: int = 8, head_dim: int | None = None
) -> np.ndarray:
    """
    Apply self-attention to embeddings.

    Uses: embedding-attention.glsl

    Args:
        embeddings: Input embeddings (batch, seq_len, dim)
        num_heads: Number of attention heads
        head_dim: Dimension of each head (default: dim // num_heads)

    Returns:
        Attended embeddings (batch, seq_len, dim)
    """
    backend = _get_backend()

    # Try GPU shader if available
    if backend and hasattr(backend, "shaders") and "embedding-attention" in backend.shaders:
        try:
            # GPU embedding attention would go here
            # For now, use CPU fallback
            pass
        except Exception:
            pass  # Fall back to CPU

    # CPU fallback (simplified)
    from grilly.functional.attention import attention

    return attention(embeddings, embeddings, embeddings, num_heads=num_heads)
