"""Functional attention helpers backed by Grilly compute kernels."""

import numpy as np


def _to_numpy(result):
    """Convert bridge result to numpy if it's a C++ Tensor."""
    if result is None:
        return None
    if isinstance(result, np.ndarray):
        return result
    if hasattr(result, "numpy"):
        return result.numpy()
    return np.asarray(result)


def _numpy_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _numpy_attention(
    query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(query, dtype=np.float32)
    k = np.asarray(key, dtype=np.float32)
    v = np.asarray(value, dtype=np.float32)
    scale = 1.0 / np.sqrt(float(q.shape[-1]))
    scores = (q @ np.swapaxes(k, -1, -2)) * scale
    if mask is not None:
        mask_arr = np.asarray(mask)
        if mask_arr.dtype == np.bool_:
            scores = np.where(mask_arr, scores, -1e9)
        else:
            scores = scores + mask_arr.astype(np.float32)
    weights = _numpy_softmax(scores, axis=-1).astype(np.float32)
    output = (weights @ v).astype(np.float32)
    return output, weights


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
    try:
        from grilly.backend import _bridge

        scores = _bridge.attention_scores(query, key)
        if scores is not None:
            if mask is not None:
                masked_scores = _bridge.attention_mask(scores, mask, False)
                if masked_scores is not None:
                    scores = masked_scores
            output = _bridge.attention_output(scores, value)
            if output is not None:
                return _to_numpy(output), _to_numpy(scores)
    except (ImportError, Exception):
        pass

    return _numpy_attention(query, key, value, mask)


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
    try:
        from grilly.backend import _bridge

        q = query
        k = key
        if use_rope:
            q_rope = _bridge.rope(q)
            k_rope = _bridge.rope(k)
            if q_rope is not None and k_rope is not None:
                q = q_rope
                k = k_rope

        result = _bridge.flash_attention2(q, k, value)
        if result is not None:
            return _to_numpy(result)
    except (ImportError, Exception):
        pass

    output, _ = _numpy_attention(query, key, value, mask=None)
    return output
