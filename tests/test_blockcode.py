"""
Tests for NVSA sparse block-code operations (T-021).

Block codes: k blocks of size l, each block is a one-hot vector.
Binding is per-block circular convolution; unbind is circular correlation.
Similarity is the fraction of matching block positions (in [0, 1]).
"""

import numpy as np
import pytest
from grilly.backend._bridge import blockcode_bind, blockcode_similarity, blockcode_unbind

# Default small config used across most tests
NUM_BLOCKS = 10
BLOCK_SIZE = 8
VEC_DIM = NUM_BLOCKS * BLOCK_SIZE  # 80


# ── Helpers ───────────────────────────────────────────────────────────────


def make_random_block_vector(num_blocks, block_size, rng=None, batch_size=None):
    """Return a random one-hot block-code vector (or batch of them)."""
    if rng is None:
        rng = np.random.default_rng(42)
    if batch_size is None:
        out = np.zeros(num_blocks * block_size, dtype=np.float32)
        for b in range(num_blocks):
            out[b * block_size + rng.integers(0, block_size)] = 1.0
        return out
    out = np.zeros((batch_size, num_blocks * block_size), dtype=np.float32)
    for v in range(batch_size):
        for b in range(num_blocks):
            out[v, b * block_size + rng.integers(0, block_size)] = 1.0
    return out


def is_valid_block_code(vec, num_blocks, block_size):
    """Return True if every block is a valid one-hot vector."""
    for b in range(num_blocks):
        base = b * block_size
        block = vec[..., base : base + block_size]
        if not np.allclose(block.sum(axis=-1), 1.0):
            return False
        if not np.all((block == 0.0) | (block == 1.0)):
            return False
    return True


# ── blockcode_bind ────────────────────────────────────────────────────────


class TestBlockcodeBind:
    def test_output_is_valid_block_code(self):
        rng = np.random.default_rng(0)
        a = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        b = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        result = blockcode_bind(a, b, NUM_BLOCKS, BLOCK_SIZE)
        assert result.shape == (VEC_DIM,)
        assert is_valid_block_code(result, NUM_BLOCKS, BLOCK_SIZE)

    def test_bind_identity_element(self):
        """Binding with a zero-position one-hot (the identity) is a no-op."""
        rng = np.random.default_rng(1)
        a = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        # identity: hot position 0 in every block
        identity = np.zeros(VEC_DIM, dtype=np.float32)
        for b in range(NUM_BLOCKS):
            identity[b * BLOCK_SIZE] = 1.0
        result = blockcode_bind(a, identity, NUM_BLOCKS, BLOCK_SIZE)
        np.testing.assert_array_equal(result, a)

    def test_bind_is_commutative(self):
        rng = np.random.default_rng(2)
        a = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        b = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        ab = blockcode_bind(a, b, NUM_BLOCKS, BLOCK_SIZE)
        ba = blockcode_bind(b, a, NUM_BLOCKS, BLOCK_SIZE)
        np.testing.assert_array_equal(ab, ba)

    def test_bind_hot_positions_add_mod_block_size(self):
        """Manual check: hot positions should add modularly per block."""
        rng = np.random.default_rng(3)
        a = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        b = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        result = blockcode_bind(a, b, NUM_BLOCKS, BLOCK_SIZE)
        for block_idx in range(NUM_BLOCKS):
            base = block_idx * BLOCK_SIZE
            hot_a = int(np.argmax(a[base : base + BLOCK_SIZE]))
            hot_b = int(np.argmax(b[base : base + BLOCK_SIZE]))
            hot_r = int(np.argmax(result[base : base + BLOCK_SIZE]))
            assert hot_r == (hot_a + hot_b) % BLOCK_SIZE

    def test_bind_batch_shape(self):
        rng = np.random.default_rng(4)
        batch = 5
        a = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng, batch_size=batch)
        b = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng, batch_size=batch)
        result = blockcode_bind(a, b, NUM_BLOCKS, BLOCK_SIZE)
        assert result.shape == (batch, VEC_DIM)
        for i in range(batch):
            assert is_valid_block_code(result[i], NUM_BLOCKS, BLOCK_SIZE)

    def test_bind_output_dtype(self):
        rng = np.random.default_rng(5)
        a = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        b = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        result = blockcode_bind(a, b, NUM_BLOCKS, BLOCK_SIZE)
        assert result.dtype == np.float32


# ── blockcode_unbind ──────────────────────────────────────────────────────


class TestBlockcodeUnbind:
    def test_unbind_output_is_valid_block_code(self):
        rng = np.random.default_rng(10)
        a = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        b = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        ab = blockcode_bind(a, b, NUM_BLOCKS, BLOCK_SIZE)
        result = blockcode_unbind(ab, b, NUM_BLOCKS, BLOCK_SIZE)
        assert is_valid_block_code(result, NUM_BLOCKS, BLOCK_SIZE)

    def test_bind_unbind_roundtrip(self):
        """Unbinding with the key recovers the original vector exactly."""
        rng = np.random.default_rng(11)
        a = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        b = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        ab = blockcode_bind(a, b, NUM_BLOCKS, BLOCK_SIZE)
        recovered = blockcode_unbind(ab, b, NUM_BLOCKS, BLOCK_SIZE)
        np.testing.assert_array_equal(recovered, a)

    def test_bind_unbind_roundtrip_other_key(self):
        """Unbinding with the other key also recovers the right vector."""
        rng = np.random.default_rng(12)
        a = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        b = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        ab = blockcode_bind(a, b, NUM_BLOCKS, BLOCK_SIZE)
        recovered = blockcode_unbind(ab, a, NUM_BLOCKS, BLOCK_SIZE)
        np.testing.assert_array_equal(recovered, b)

    def test_unbind_batch_roundtrip(self):
        rng = np.random.default_rng(13)
        batch = 4
        a = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng, batch_size=batch)
        b = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng, batch_size=batch)
        ab = blockcode_bind(a, b, NUM_BLOCKS, BLOCK_SIZE)
        recovered = blockcode_unbind(ab, b, NUM_BLOCKS, BLOCK_SIZE)
        np.testing.assert_array_equal(recovered, a)

    def test_unbind_wrong_key_differs(self):
        """Unbinding with a wrong key should not equal the original in general."""
        rng = np.random.default_rng(14)
        a = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        b = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        wrong_key = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        ab = blockcode_bind(a, b, NUM_BLOCKS, BLOCK_SIZE)
        recovered = blockcode_unbind(ab, wrong_key, NUM_BLOCKS, BLOCK_SIZE)
        # With high probability this differs from a (block_size=8, num_blocks=10)
        assert not np.array_equal(recovered, a)


# ── blockcode_similarity ──────────────────────────────────────────────────


class TestBlockcodeSimilarity:
    def test_identical_vectors_score_one(self):
        rng = np.random.default_rng(20)
        v = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        codebook = v[np.newaxis, :]
        sims = blockcode_similarity(v, codebook, NUM_BLOCKS, BLOCK_SIZE)
        assert sims.shape == (1,)
        assert pytest.approx(float(sims[0]), abs=1e-6) == 1.0

    def test_different_vectors_lower_similarity(self):
        rng = np.random.default_rng(21)
        query = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        other = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        codebook = np.stack([query, other])
        sims = blockcode_similarity(query, codebook, NUM_BLOCKS, BLOCK_SIZE)
        assert sims.shape == (2,)
        assert pytest.approx(float(sims[0]), abs=1e-6) == 1.0
        assert float(sims[1]) < 1.0

    def test_similarity_in_range(self):
        """All similarity scores must be in [0, 1]."""
        rng = np.random.default_rng(22)
        query = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        codebook = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng, batch_size=50)
        sims = blockcode_similarity(query, codebook, NUM_BLOCKS, BLOCK_SIZE)
        assert np.all(sims >= 0.0)
        assert np.all(sims <= 1.0 + 1e-6)

    def test_random_similarity_near_reciprocal_block_size(self):
        """Random pairs should average to ~1/block_size by birthday paradox."""
        rng = np.random.default_rng(23)
        query = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        n_entries = 500
        codebook = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng, batch_size=n_entries)
        sims = blockcode_similarity(query, codebook, NUM_BLOCKS, BLOCK_SIZE)
        mean_sim = float(sims.mean())
        expected = 1.0 / BLOCK_SIZE
        # Allow ±2 sigma margin (binomial std ≈ sqrt(p*(1-p)/num_blocks/n_entries))
        tol = 0.05
        assert abs(mean_sim - expected) < tol, (
            f"Mean similarity {mean_sim:.4f} too far from expected {expected:.4f}"
        )

    def test_similarity_output_dtype(self):
        rng = np.random.default_rng(24)
        query = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        codebook = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng, batch_size=5)
        sims = blockcode_similarity(query, codebook, NUM_BLOCKS, BLOCK_SIZE)
        assert sims.dtype == np.float32

    def test_similarity_top_match_retrieval(self):
        """The bound-then-unbound vector should be the top match in the codebook."""
        rng = np.random.default_rng(25)
        n_entries = 100
        codebook = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng, batch_size=n_entries)
        target_idx = 17
        query = codebook[target_idx]
        sims = blockcode_similarity(query, codebook, NUM_BLOCKS, BLOCK_SIZE)
        assert int(np.argmax(sims)) == target_idx


# ── Magnitude / sparsity preservation ────────────────────────────────────


class TestMagnitudePreservation:
    def test_chained_bindings_stay_one_hot(self):
        """After 100 chained bindings the vector is still a valid block code."""
        rng = np.random.default_rng(30)
        v = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        for _ in range(100):
            b = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
            v = blockcode_bind(v, b, NUM_BLOCKS, BLOCK_SIZE)
        assert is_valid_block_code(v, NUM_BLOCKS, BLOCK_SIZE)

    def test_chained_bindings_magnitude_preserved(self):
        """Each block should still sum to exactly 1.0 after 100 bindings."""
        rng = np.random.default_rng(31)
        v = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        for _ in range(100):
            b = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
            v = blockcode_bind(v, b, NUM_BLOCKS, BLOCK_SIZE)
        for block_idx in range(NUM_BLOCKS):
            base = block_idx * BLOCK_SIZE
            block_sum = float(v[base : base + BLOCK_SIZE].sum())
            assert pytest.approx(block_sum, abs=1e-6) == 1.0, (
                f"Block {block_idx} sum = {block_sum}, expected 1.0"
            )

    def test_chained_bindings_unbind_chain(self):
        """Sequentially unbinding all keys recovers the original base vector."""
        rng = np.random.default_rng(32)
        base_vec = make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng)
        keys = [make_random_block_vector(NUM_BLOCKS, BLOCK_SIZE, rng) for _ in range(5)]

        # Bind all keys sequentially
        composite = base_vec.copy()
        for k in keys:
            composite = blockcode_bind(composite, k, NUM_BLOCKS, BLOCK_SIZE)

        # Unbind in reverse order
        recovered = composite.copy()
        for k in reversed(keys):
            recovered = blockcode_unbind(recovered, k, NUM_BLOCKS, BLOCK_SIZE)

        np.testing.assert_array_equal(recovered, base_vec)

    def test_large_vector_no_drift(self):
        """Test with NVSA-scale vectors: k=200, l=50 (10000-dim)."""
        k, l = 20, 50  # reduced for test speed (same ratio as k=200, l=50)
        rng = np.random.default_rng(33)
        v = make_random_block_vector(k, l, rng)
        for _ in range(100):
            b = make_random_block_vector(k, l, rng)
            v = blockcode_bind(v, b, k, l)
        assert is_valid_block_code(v, k, l)
