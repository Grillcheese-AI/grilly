"""Adaptive KV cache budget allocation using TAPPA q-similarity.

Heads with low q-similarity (unpredictable, retrieval) get more KV cache budget.
Heads with high q-similarity (predictable, streaming) get less budget.
"""

import numpy as np


def compute_head_budgets(
    queries: np.ndarray,
    total_budget: int,
    min_budget_per_head: int = 32,
) -> np.ndarray:
    """Compute per-head KV cache token budgets based on q-similarity.

    Args:
        queries: shape (batch, num_heads, seq_len, head_dim)
        total_budget: total tokens across all heads
        min_budget_per_head: minimum tokens per head (floor)

    Returns:
        budgets: shape (num_heads,) — token budget per head
    """
    from ._bridge import q_similarity

    # Compute q-similarity per head: shape (batch, num_heads)
    qsim = q_similarity(queries)

    # Average across batch
    if qsim.ndim == 2:
        qsim_per_head = qsim.mean(axis=0)  # (num_heads,)
    else:
        qsim_per_head = qsim  # already 1D

    num_heads = len(qsim_per_head)

    # Invert: low q-sim → high weight (needs more budget)
    # Transform from [-1,1] or [0,1] to positive weights
    weights = 1.0 - np.clip(qsim_per_head, 0.0, 1.0)
    weights = np.maximum(weights, 0.01)  # prevent zero weight

    # Normalize weights to sum to 1
    weights = weights / weights.sum()

    # Allocate budget proportionally
    budgets = np.maximum(
        np.round(weights * total_budget).astype(np.int32),
        min_budget_per_head,
    )

    # Adjust to exactly match total_budget
    diff = total_budget - budgets.sum()
    if diff != 0:
        # Add/remove from the head with highest weight
        idx = np.argmax(weights) if diff > 0 else np.argmin(weights)
        budgets[idx] += diff

    return budgets


def classify_heads(
    queries: np.ndarray,
    threshold_retrieval: float = 0.3,
    threshold_streaming: float = 0.8,
) -> dict:
    """Classify attention heads by TAPPA q-similarity.

    Args:
        queries: shape (batch, num_heads, seq_len, head_dim)
        threshold_retrieval: below this → retrieval head (unpredictable)
        threshold_streaming: above this → streaming head (predictable)

    Returns:
        dict with keys: retrieval_heads, mixed_heads, streaming_heads, q_similarities
    """
    from ._bridge import q_similarity

    qsim = q_similarity(queries)
    if qsim.ndim == 2:
        qsim_per_head = qsim.mean(axis=0)
    else:
        qsim_per_head = qsim

    retrieval = np.where(qsim_per_head < threshold_retrieval)[0].tolist()
    streaming = np.where(qsim_per_head > threshold_streaming)[0].tolist()
    mixed = [
        i
        for i in range(len(qsim_per_head))
        if i not in retrieval and i not in streaming
    ]

    return {
        "retrieval_heads": retrieval,
        "mixed_heads": mixed,
        "streaming_heads": streaming,
        "q_similarities": qsim_per_head.tolist(),
    }


def eviction_priority(
    cumulative_scores: np.ndarray,
    q_similarities: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Compute eviction priority combining H2O scores with q-similarity.

    Tokens in predictable (high q-sim) heads are evicted more aggressively.
    Tokens in retrieval (low q-sim) heads are preserved.

    Args:
        cumulative_scores: shape (num_heads, num_tokens) — H2O cumulative attention
        q_similarities: shape (num_heads,) — per-head q-similarity
        alpha: weight of q-similarity in eviction decision (0=pure H2O, 1=pure q-sim)

    Returns:
        priority: shape (num_heads, num_tokens) — lower = evict first
    """
    # Normalize cumulative scores per head
    score_norm = cumulative_scores / (cumulative_scores.max(axis=-1, keepdims=True) + 1e-8)

    # Q-similarity penalty: high q-sim heads get lower priority (evict more)
    qsim_penalty = q_similarities[:, np.newaxis]  # (num_heads, 1)

    # Combined priority: high score = keep, high q-sim = evict
    priority = (1 - alpha) * score_norm + alpha * (1 - qsim_penalty)

    return priority
