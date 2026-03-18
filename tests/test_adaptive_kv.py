"""Tests for TAPPA adaptive KV cache budget allocation (backend/adaptive_kv.py)."""

import numpy as np
import pytest

from backend.adaptive_kv import classify_heads, compute_head_budgets, eviction_priority


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_queries(batch=2, num_heads=4, seq_len=16, head_dim=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, num_heads, seq_len, head_dim)).astype(np.float32)


def _make_queries_with_known_qsim(batch=1, seq_len=16, head_dim=32):
    """Construct queries where head 0 is highly predictable and head 1 is random.

    Head 0: all queries identical → cosine sim = 1.0 (streaming head).
    Head 1: random queries → cosine sim ≈ 0.0 (retrieval head).
    """
    rng = np.random.default_rng(42)
    base = rng.standard_normal(head_dim).astype(np.float32)
    base /= np.linalg.norm(base)

    # Head 0: identical queries (q-sim ≈ 1.0)
    head0 = np.tile(base, (batch, 1, seq_len, 1))  # (B, 1, S, D)
    head0 = head0.transpose(0, 1, 2, 3)  # already correct shape

    # Head 1: random orthogonal-ish queries (q-sim ≈ 0.0)
    head1 = rng.standard_normal((batch, 1, seq_len, head_dim)).astype(np.float32)

    queries = np.concatenate([head0, head1], axis=1)  # (B, 2, S, D)
    return queries


# ── compute_head_budgets ──────────────────────────────────────────────────────


def test_budget_sums_to_total():
    queries = _make_queries()
    total_budget = 512
    budgets = compute_head_budgets(queries, total_budget)

    assert budgets.shape == (4,), f"Expected shape (4,), got {budgets.shape}"
    assert budgets.sum() == total_budget, (
        f"Budget sum {budgets.sum()} != total_budget {total_budget}"
    )


def test_budget_respects_min_per_head():
    queries = _make_queries(num_heads=8)
    min_budget = 16
    total_budget = 256
    budgets = compute_head_budgets(queries, total_budget, min_budget_per_head=min_budget)

    assert (budgets >= min_budget).all(), (
        f"Some head budgets are below minimum {min_budget}: {budgets}"
    )


def test_low_qsim_head_gets_more_budget():
    """Head with low q-similarity should receive more budget than high q-sim head."""
    queries = _make_queries_with_known_qsim(batch=2, seq_len=32, head_dim=32)
    total_budget = 256
    min_budget = 8
    budgets = compute_head_budgets(queries, total_budget, min_budget_per_head=min_budget)

    # head 0 is streaming (high q-sim, low budget)
    # head 1 is retrieval (low q-sim, high budget)
    assert budgets[1] >= budgets[0], (
        f"Retrieval head (1) budget {budgets[1]} should be >= streaming head (0) budget {budgets[0]}"
    )


def test_budget_returns_int32():
    queries = _make_queries()
    budgets = compute_head_budgets(queries, 128)
    assert budgets.dtype == np.int32, f"Expected int32 budgets, got {budgets.dtype}"


def test_budget_single_head():
    """Single-head edge case: all budget goes to that head."""
    queries = _make_queries(num_heads=1, seq_len=8)
    total_budget = 64
    budgets = compute_head_budgets(queries, total_budget, min_budget_per_head=8)

    assert budgets.shape == (1,)
    assert budgets.sum() == total_budget


# ── classify_heads ────────────────────────────────────────────────────────────


def test_classify_heads_returns_correct_keys():
    queries = _make_queries()
    result = classify_heads(queries)

    assert set(result.keys()) == {
        "retrieval_heads",
        "mixed_heads",
        "streaming_heads",
        "q_similarities",
    }


def test_classify_heads_covers_all_heads():
    queries = _make_queries(num_heads=6)
    result = classify_heads(queries)

    all_heads = set(result["retrieval_heads"]) | set(result["mixed_heads"]) | set(result["streaming_heads"])
    assert all_heads == set(range(6)), f"Not all heads covered: {all_heads}"


def test_classify_heads_no_overlap():
    queries = _make_queries(num_heads=6)
    result = classify_heads(queries)

    retrieval = set(result["retrieval_heads"])
    mixed = set(result["mixed_heads"])
    streaming = set(result["streaming_heads"])

    assert retrieval.isdisjoint(mixed), "retrieval and mixed overlap"
    assert retrieval.isdisjoint(streaming), "retrieval and streaming overlap"
    assert mixed.isdisjoint(streaming), "mixed and streaming overlap"


def test_classify_heads_known_categories():
    """Streaming head (q-sim ≈ 1.0) and retrieval head (q-sim ≈ 0.0) classified correctly."""
    queries = _make_queries_with_known_qsim(batch=2, seq_len=32, head_dim=32)
    result = classify_heads(queries, threshold_retrieval=0.3, threshold_streaming=0.8)

    assert 0 in result["streaming_heads"], (
        f"Head 0 (high q-sim) should be streaming, got: {result}"
    )
    assert 1 in result["retrieval_heads"], (
        f"Head 1 (low q-sim) should be retrieval, got: {result}"
    )


def test_classify_heads_q_similarities_length():
    queries = _make_queries(num_heads=5)
    result = classify_heads(queries)

    assert len(result["q_similarities"]) == 5


# ── eviction_priority ─────────────────────────────────────────────────────────


def test_eviction_priority_shape():
    num_heads, num_tokens = 4, 100
    cumulative_scores = np.random.rand(num_heads, num_tokens).astype(np.float32)
    q_similarities = np.array([0.1, 0.5, 0.8, 0.95], dtype=np.float32)

    priority = eviction_priority(cumulative_scores, q_similarities)

    assert priority.shape == (num_heads, num_tokens), (
        f"Expected shape ({num_heads}, {num_tokens}), got {priority.shape}"
    )


def test_eviction_priority_retrieval_head_preserved():
    """Tokens in retrieval (low q-sim) heads should have higher priority (not evicted)."""
    num_tokens = 50
    # Uniform H2O scores — q-sim drives the difference
    cumulative_scores = np.ones((2, num_tokens), dtype=np.float32)
    q_similarities = np.array([0.05, 0.95], dtype=np.float32)  # head0=retrieval, head1=streaming

    priority = eviction_priority(cumulative_scores, q_similarities, alpha=0.9)

    # Head 0 (retrieval, low q-sim) should have higher mean priority than head 1 (streaming)
    assert priority[0].mean() > priority[1].mean(), (
        f"Retrieval head mean priority {priority[0].mean():.4f} should exceed "
        f"streaming head {priority[1].mean():.4f}"
    )


def test_eviction_priority_pure_h2o(alpha=0.0):
    """alpha=0 reduces to pure H2O score normalization; q-sim has no effect."""
    num_heads, num_tokens = 3, 20
    cumulative_scores = np.random.rand(num_heads, num_tokens).astype(np.float32)
    q_similarities = np.array([0.1, 0.5, 0.9], dtype=np.float32)

    priority = eviction_priority(cumulative_scores, q_similarities, alpha=0.0)

    # With alpha=0 the result is score_norm; per-head max = max(score_norm) ≤ 1
    assert priority.max() <= 1.0 + 1e-5
    assert priority.min() >= 0.0 - 1e-5


def test_eviction_priority_pure_qsim():
    """alpha=1 means eviction depends entirely on q-similarity (uniform within a head)."""
    num_heads, num_tokens = 2, 30
    cumulative_scores = np.ones((num_heads, num_tokens), dtype=np.float32)
    q_similarities = np.array([0.2, 0.8], dtype=np.float32)

    priority = eviction_priority(cumulative_scores, q_similarities, alpha=1.0)

    # With alpha=1 and uniform scores, all tokens in a head get the same priority
    assert np.allclose(priority[0], priority[0, 0]), "Head 0 tokens should all be equal"
    assert np.allclose(priority[1], priority[1, 0]), "Head 1 tokens should all be equal"
    # Retrieval head (low q-sim) has higher priority than streaming head
    assert priority[0, 0] > priority[1, 0]


def test_eviction_priority_alpha_boundary():
    """alpha clamps work: 0 and 1 are valid edge cases."""
    scores = np.random.rand(4, 16).astype(np.float32)
    qsim = np.random.rand(4).astype(np.float32)

    for alpha in (0.0, 0.5, 1.0):
        p = eviction_priority(scores, qsim, alpha=alpha)
        assert p.shape == (4, 16), f"Bad shape at alpha={alpha}"
        assert np.isfinite(p).all(), f"Non-finite values at alpha={alpha}"
