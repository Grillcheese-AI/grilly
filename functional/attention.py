"""Functional attention helpers backed by Grilly compute kernels."""

import numpy as np


def _get_backend():
    """Get compute backend"""
    from grilly import Compute

    return Compute()


def attention(
    query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-head attention
    Uses: attention-scores.glsl, attention-output.glsl, attention-concat-heads.glsl, attention-mask.glsl

    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        mask: Optional attention mask

    Returns:
        Tuple of (output, attention_weights)
    """
    backend = _get_backend()

    # Compute attention scores
    scores = backend.attention_scores(query, key)

    # Apply mask if provided
    if mask is not None:
        scores = backend.attention_mask(scores, mask)

    # Compute attention output
    output = backend.attention_output(scores, value)

    return output, scores


def flash_attention2(
    query: np.ndarray, key: np.ndarray, value: np.ndarray, use_rope: bool = False
) -> np.ndarray:
    """
    Flash Attention 2 (optimized attention)
    Uses: flash-attention2.glsl, flash-attention2-rope.glsl

    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        use_rope: Whether to use Rotary Position Embeddings

    Returns:
        Attention output
    """
    backend = _get_backend()
    return backend.flash_attention2(query, key, value, use_rope=use_rope)
