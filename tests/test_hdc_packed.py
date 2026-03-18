"""
Tests for bit-packed binary hypervector operations (T-020).

Binary hypervectors are packed into uint32 words for 32x memory compression
over float32. Four operations: bind (XOR), bundle (majority vote),
similarity (Hamming), and permute (cyclic bit shift).

All tests use the numpy fallback path (no GPU required).
"""

import numpy as np
import pytest

from grilly.backend import _bridge


# ── Helpers ──────────────────────────────────────────────────────────────

DIM = 256  # hypervector dimension
WORDS = DIM // 32  # = 8 uint32 words per vector


def random_hv(seed=None):
    """Return a random packed binary hypervector as uint32 array."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**32, size=WORDS, dtype=np.uint64).astype(np.uint32)


# ── hdc_bind_packed ───────────────────────────────────────────────────────


class TestHdcBindPacked:
    """XOR binding on packed uint32 hypervectors."""

    def test_bind_output_shape(self):
        """Output shape matches input shape."""
        a = random_hv(0)
        b = random_hv(1)
        result = _bridge.hdc_bind_packed(a, b)
        assert result.shape == (WORDS,)
        assert result.dtype == np.uint32

    def test_bind_is_xor(self):
        """Binding equals bitwise XOR."""
        a = random_hv(10)
        b = random_hv(20)
        expected = np.bitwise_xor(a, b)
        result = _bridge.hdc_bind_packed(a, b)
        np.testing.assert_array_equal(result, expected)

    def test_bind_self_inverse(self):
        """Binding a vector with itself gives zero (XOR self-inverse property)."""
        a = random_hv(42)
        result = _bridge.hdc_bind_packed(a, a)
        np.testing.assert_array_equal(result, np.zeros(WORDS, dtype=np.uint32))

    def test_bind_inverse_recovery(self):
        """Binding ab with b recovers a (unbinding via XOR)."""
        a = random_hv(7)
        b = random_hv(8)
        ab = _bridge.hdc_bind_packed(a, b)
        recovered = _bridge.hdc_bind_packed(ab, b)
        np.testing.assert_array_equal(recovered, a)

    def test_bind_2d_batch(self):
        """Binding works on 2D arrays (batch of hypervectors)."""
        batch = 4
        a = np.random.default_rng(0).integers(0, 2**32, size=(batch, WORDS), dtype=np.uint64).astype(np.uint32)
        b = np.random.default_rng(1).integers(0, 2**32, size=(batch, WORDS), dtype=np.uint64).astype(np.uint32)
        result = _bridge.hdc_bind_packed(a, b)
        expected = np.bitwise_xor(a, b)
        np.testing.assert_array_equal(result, expected)


# ── hdc_bundle_packed ─────────────────────────────────────────────────────


class TestHdcBundlePacked:
    """Majority-vote bundling on packed uint32 hypervectors."""

    def test_bundle_output_shape(self):
        """Output shape is (words_per_vec,)."""
        rng = np.random.default_rng(0)
        vecs = rng.integers(0, 2**32, size=(5, WORDS), dtype=np.uint64).astype(np.uint32)
        result = _bridge.hdc_bundle_packed(vecs, WORDS)
        assert result.shape == (WORDS,)
        assert result.dtype == np.uint32

    def test_bundle_all_ones(self):
        """Bundling N copies of all-ones gives all-ones."""
        ones = np.full(WORDS, 0xFFFFFFFF, dtype=np.uint32)
        vecs = np.stack([ones] * 5)
        result = _bridge.hdc_bundle_packed(vecs, WORDS)
        np.testing.assert_array_equal(result, ones)

    def test_bundle_all_zeros(self):
        """Bundling N copies of all-zeros gives all-zeros."""
        zeros = np.zeros(WORDS, dtype=np.uint32)
        vecs = np.stack([zeros] * 5)
        result = _bridge.hdc_bundle_packed(vecs, WORDS)
        np.testing.assert_array_equal(result, zeros)

    def test_bundle_majority_vote(self):
        """With 3 vectors, majority vote selects the bit that appears in >= 2."""
        # Construct 3 simple 1-word vectors for DIM=32 (WORDS=1)
        words_per_vec = 1
        # vec0: bit0=1, bit1=0  -> 0b01 = 1
        # vec1: bit0=1, bit1=1  -> 0b11 = 3
        # vec2: bit0=0, bit1=1  -> 0b10 = 2
        # majority: bit0=1 (2 votes), bit1=1 (2 votes) -> 0b11 = 3
        vecs = np.array([[1], [3], [2]], dtype=np.uint32)
        result = _bridge.hdc_bundle_packed(vecs, words_per_vec)
        assert result[0] == np.uint32(3)

    def test_bundle_single_vector(self):
        """Bundling a single vector returns that vector (trivial majority)."""
        v = random_hv(99)
        vecs = v[np.newaxis, :]
        result = _bridge.hdc_bundle_packed(vecs, WORDS)
        np.testing.assert_array_equal(result, v)


# ── hdc_similarity_packed ─────────────────────────────────────────────────


class TestHdcSimilarityPacked:
    """Hamming similarity between a packed query and a codebook."""

    def test_similarity_output_shape(self):
        """Output shape is (num_entries,) float32."""
        query = random_hv(0)
        num_entries = 10
        rng = np.random.default_rng(1)
        codebook = rng.integers(0, 2**32, size=(num_entries, WORDS), dtype=np.uint64).astype(np.uint32)
        result = _bridge.hdc_similarity_packed(query, codebook, DIM)
        assert result.shape == (num_entries,)
        assert result.dtype == np.float32

    def test_similarity_self_is_one(self):
        """A vector is identical to itself: similarity = 1.0 (Hamming dist = 0)."""
        v = random_hv(5)
        codebook = v[np.newaxis, :]
        result = _bridge.hdc_similarity_packed(v, codebook, DIM)
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_similarity_complement_is_zero(self):
        """Bitwise complement has Hamming dist = dim: similarity = 0.0.

        Formula: 1 - hamming/dim. When all dim bits differ, hamming=dim -> 0.0.
        """
        v = random_hv(6)
        complement = ~v
        codebook = complement[np.newaxis, :]
        result = _bridge.hdc_similarity_packed(v, codebook, DIM)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_similarity_range(self):
        """All similarities are in [0.0, 1.0].

        Formula: 1 - hamming/dim. Hamming distance is in [0, dim], so similarity
        is in [0, 1].
        """
        query = random_hv(0)
        rng = np.random.default_rng(2)
        num_entries = 50
        codebook = rng.integers(0, 2**32, size=(num_entries, WORDS), dtype=np.uint64).astype(np.uint32)
        result = _bridge.hdc_similarity_packed(query, codebook, DIM)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_similarity_random_near_half(self):
        """Random uncorrelated hypervectors have similarity near 0.5.

        Formula: 1 - hamming/dim. For 256-dim random vectors, expected Hamming
        distance ~ 128 (half the bits differ), so similarity ~ 1 - 0.5 = 0.5.
        """
        rng = np.random.default_rng(3)
        query = rng.integers(0, 2**32, size=WORDS, dtype=np.uint64).astype(np.uint32)
        num_entries = 200
        codebook = rng.integers(0, 2**32, size=(num_entries, WORDS), dtype=np.uint64).astype(np.uint32)
        result = _bridge.hdc_similarity_packed(query, codebook, DIM)
        # For 256-dim random vectors, mean Hamming distance ~ 128 -> sim ~ 0.5
        assert abs(float(result.mean()) - 0.5) < 0.05, (
            f"mean similarity {result.mean():.3f} should be near 0.5 for random vectors"
        )


# ── hdc_permute_packed ────────────────────────────────────────────────────


class TestHdcPermutePacked:
    """Cyclic bit permutation on packed uint32 hypervectors."""

    def test_permute_output_shape(self):
        """Output shape matches input shape."""
        v = random_hv(0)
        result = _bridge.hdc_permute_packed(v, WORDS, shift=1)
        assert result.shape == (WORDS,)
        assert result.dtype == np.uint32

    def test_permute_zero_shift_identity(self):
        """Shift of 0 returns the original vector unchanged."""
        v = random_hv(1)
        result = _bridge.hdc_permute_packed(v, WORDS, shift=0)
        np.testing.assert_array_equal(result, v)

    def test_permute_full_cycle_is_identity(self):
        """Shifting by dim (total_bits) is a full cycle — returns original."""
        v = random_hv(2)
        result = _bridge.hdc_permute_packed(v, WORDS, shift=DIM)
        np.testing.assert_array_equal(result, v)

    def test_permute_inverse(self):
        """Shifting by +k then -k (or equivalently dim-k) recovers original."""
        v = random_hv(3)
        k = 13
        permuted = _bridge.hdc_permute_packed(v, WORDS, shift=k)
        recovered = _bridge.hdc_permute_packed(permuted, WORDS, shift=DIM - k)
        np.testing.assert_array_equal(recovered, v)

    def test_permute_dissimilarity(self):
        """A permuted vector differs from the original (for non-trivial shifts)."""
        v = random_hv(4)
        # Use a shift that is not 0 or DIM
        permuted = _bridge.hdc_permute_packed(v, WORDS, shift=1)
        # They should not be identical for a random vector
        assert not np.array_equal(v, permuted), "permuted vector should differ from original"

    def test_permute_shift_by_32_rolls_one_word(self):
        """Shifting by 32 rolls exactly one uint32 word boundary."""
        # Create a vector with only bit 31 set in word 0: shifting by 1 should
        # move that bit to bit 0 of word 1.
        v = np.zeros(WORDS, dtype=np.uint32)
        v[0] = np.uint32(1 << 31)  # bit 31 of word 0 = global bit 31
        result = _bridge.hdc_permute_packed(v, WORDS, shift=1)
        # After +1 shift: global bit 31 -> global bit 32 -> word 1 bit 0
        assert result[1] & np.uint32(1), "bit should have rolled into word 1 bit 0"
        assert result[0] == np.uint32(0), "word 0 should be empty after shift"
