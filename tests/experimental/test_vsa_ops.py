"""
TDD Tests for VSA Operations: HolographicOps and BinaryOps.

These tests define the expected behavior of VSA operations.
Write tests first (RED), then implement to pass (GREEN).
"""

import numpy as np
import pytest


class TestBinaryOps:
    """Tests for BinaryOps (bipolar +1/-1 vectors)."""

    def test_bind_returns_correct_shape(self, dim, rng):
        """Bind should return a vector of the same dimension."""
        from grilly.experimental.vsa.ops import BinaryOps

        a = np.sign(rng.standard_normal(dim)).astype(np.float32)
        b = np.sign(rng.standard_normal(dim)).astype(np.float32)

        result = BinaryOps.bind(a, b)

        assert result.shape == (dim,)
        assert result.dtype == np.float32

    def test_bind_is_bipolar(self, dim, rng):
        """Bind of bipolar vectors should produce bipolar output."""
        from grilly.experimental.vsa.ops import BinaryOps

        a = np.sign(rng.standard_normal(dim)).astype(np.float32)
        b = np.sign(rng.standard_normal(dim)).astype(np.float32)

        result = BinaryOps.bind(a, b)

        # All values should be +1 or -1
        assert np.all(np.abs(result) == 1.0)

    def test_bind_inverse_property(self, dim, rng):
        """For bipolar: bind(a, b) then unbind(result, b) should recover a."""
        from grilly.experimental.vsa.ops import BinaryOps

        a = np.sign(rng.standard_normal(dim)).astype(np.float32)
        b = np.sign(rng.standard_normal(dim)).astype(np.float32)

        bound = BinaryOps.bind(a, b)
        recovered = BinaryOps.unbind(bound, b)

        # For bipolar, unbind is same as bind, so recovered should equal a
        np.testing.assert_array_almost_equal(recovered, a)

    def test_bind_commutative(self, dim, rng):
        """Bind should be commutative: bind(a, b) == bind(b, a)."""
        from grilly.experimental.vsa.ops import BinaryOps

        a = np.sign(rng.standard_normal(dim)).astype(np.float32)
        b = np.sign(rng.standard_normal(dim)).astype(np.float32)

        result1 = BinaryOps.bind(a, b)
        result2 = BinaryOps.bind(b, a)

        np.testing.assert_array_equal(result1, result2)

    def test_bind_associative(self, bipolar_vector_triple):
        """Bind should be associative: bind(bind(a,b),c) == bind(a,bind(b,c))."""
        from grilly.experimental.vsa.ops import BinaryOps

        a, b, c = bipolar_vector_triple

        result1 = BinaryOps.bind(BinaryOps.bind(a, b), c)
        result2 = BinaryOps.bind(a, BinaryOps.bind(b, c))

        np.testing.assert_array_equal(result1, result2)

    def test_bind_with_identity(self, dim, rng):
        """Bind with all-ones vector should return original."""
        from grilly.experimental.vsa.ops import BinaryOps

        a = np.sign(rng.standard_normal(dim)).astype(np.float32)
        identity = np.ones(dim, dtype=np.float32)

        result = BinaryOps.bind(a, identity)

        np.testing.assert_array_equal(result, a)

    def test_unbind_is_self_inverse_for_bipolar(self, dim, rng):
        """For bipolar vectors, unbind(x, x) should give identity (all ones)."""
        from grilly.experimental.vsa.ops import BinaryOps

        a = np.sign(rng.standard_normal(dim)).astype(np.float32)

        result = BinaryOps.unbind(a, a)

        np.testing.assert_array_equal(result, np.ones(dim, dtype=np.float32))

    def test_bundle_returns_correct_shape(self, dim, rng):
        """Bundle should return a vector of the same dimension."""
        from grilly.experimental.vsa.ops import BinaryOps

        vectors = [np.sign(rng.standard_normal(dim)).astype(np.float32) for _ in range(5)]

        result = BinaryOps.bundle(vectors)

        assert result.shape == (dim,)

    def test_bundle_preserves_components(self, dim, rng, similarity_tolerance):
        """Bundled vector should be similar to each component."""
        from grilly.experimental.vsa.ops import BinaryOps

        vectors = [np.sign(rng.standard_normal(dim)).astype(np.float32) for _ in range(3)]

        bundled = BinaryOps.bundle(vectors)

        for v in vectors:
            sim = BinaryOps.similarity(bundled, v)
            # Each component should have positive similarity with bundle
            assert sim > 0.0, f"Expected positive similarity, got {sim}"

    def test_bundle_majority_vote(self, dim):
        """Bundle should implement majority voting."""
        from grilly.experimental.vsa.ops import BinaryOps

        # Create vectors where position 0 has majority +1
        v1 = np.ones(dim, dtype=np.float32)
        v2 = np.ones(dim, dtype=np.float32)
        v3 = -np.ones(dim, dtype=np.float32)

        bundled = BinaryOps.bundle([v1, v2, v3])

        # Majority at each position should win
        assert bundled[0] == 1.0

    def test_similarity_range(self, dim, rng):
        """Similarity should always be in [-1, 1]."""
        from grilly.experimental.vsa.ops import BinaryOps

        for _ in range(10):
            a = np.sign(rng.standard_normal(dim)).astype(np.float32)
            b = np.sign(rng.standard_normal(dim)).astype(np.float32)

            sim = BinaryOps.similarity(a, b)

            assert -1.0 <= sim <= 1.0, f"Similarity {sim} out of range"

    def test_similarity_self(self, dim, rng):
        """Similarity of vector with itself should be 1.0."""
        from grilly.experimental.vsa.ops import BinaryOps

        a = np.sign(rng.standard_normal(dim)).astype(np.float32)

        sim = BinaryOps.similarity(a, a)

        assert sim == pytest.approx(1.0)

    def test_similarity_opposite(self, dim, rng):
        """Similarity of vector with its negation should be -1.0."""
        from grilly.experimental.vsa.ops import BinaryOps

        a = np.sign(rng.standard_normal(dim)).astype(np.float32)

        sim = BinaryOps.similarity(a, -a)

        assert sim == pytest.approx(-1.0)

    def test_random_vectors_nearly_orthogonal(self, dim, rng, orthogonality_threshold):
        """Random high-dimensional vectors should be nearly orthogonal."""
        from grilly.experimental.vsa.ops import BinaryOps

        similarities = []
        for _ in range(20):
            a = np.sign(rng.standard_normal(dim)).astype(np.float32)
            b = np.sign(rng.standard_normal(dim)).astype(np.float32)
            similarities.append(abs(BinaryOps.similarity(a, b)))

        avg_similarity = np.mean(similarities)
        assert avg_similarity < orthogonality_threshold, (
            f"Average similarity {avg_similarity} exceeds threshold {orthogonality_threshold}"
        )

    def test_random_bipolar_deterministic_with_seed(self, dim):
        """Random bipolar with same seed should produce same vector."""
        from grilly.experimental.vsa.ops import BinaryOps

        v1 = BinaryOps.random_bipolar(dim, seed=42)
        v2 = BinaryOps.random_bipolar(dim, seed=42)

        np.testing.assert_array_equal(v1, v2)

    def test_random_bipolar_different_seeds_different_vectors(self, dim):
        """Random bipolar with different seeds should produce different vectors."""
        from grilly.experimental.vsa.ops import BinaryOps

        v1 = BinaryOps.random_bipolar(dim, seed=42)
        v2 = BinaryOps.random_bipolar(dim, seed=43)

        # Should not be equal
        assert not np.array_equal(v1, v2)

    def test_hash_to_bipolar_deterministic(self, dim):
        """Hash to bipolar should be deterministic."""
        from grilly.experimental.vsa.ops import BinaryOps

        v1 = BinaryOps.hash_to_bipolar("test_string", dim)
        v2 = BinaryOps.hash_to_bipolar("test_string", dim)

        np.testing.assert_array_equal(v1, v2)

    def test_hash_to_bipolar_different_strings(self, dim):
        """Different strings should hash to different vectors."""
        from grilly.experimental.vsa.ops import BinaryOps

        v1 = BinaryOps.hash_to_bipolar("string_a", dim)
        v2 = BinaryOps.hash_to_bipolar("string_b", dim)

        assert not np.array_equal(v1, v2)


class TestHolographicOps:
    """Tests for HolographicOps (continuous vectors with FFT-based binding)."""

    def test_convolve_returns_correct_shape(self, dim, rng):
        """Convolve should return a vector of the same dimension."""
        from grilly.experimental.vsa.ops import HolographicOps

        a = rng.standard_normal(dim).astype(np.float32)
        b = rng.standard_normal(dim).astype(np.float32)

        result = HolographicOps.convolve(a, b)

        assert result.shape == (dim,)

    def test_convolve_correlate_inverse(self, dim, rng):
        """Correlate should undo convolve: correlate(convolve(a,b), b) ≈ a."""
        from grilly.experimental.vsa.ops import HolographicOps

        a = rng.standard_normal(dim).astype(np.float32)
        a = a / np.linalg.norm(a)  # Normalize
        b = rng.standard_normal(dim).astype(np.float32)
        b = b / np.linalg.norm(b)  # Normalize

        convolved = HolographicOps.convolve(a, b)
        recovered = HolographicOps.correlate(convolved, b)

        # Should recover a with similarity well above random (~0.03)
        # HRR typically achieves 0.6-0.9 depending on implementation
        sim = HolographicOps.similarity(recovered, a)
        assert sim > 0.6, f"Expected good similarity, got {sim}"

    def test_convolve_commutative(self, dim, rng):
        """Convolve should be commutative: convolve(a, b) == convolve(b, a)."""
        from grilly.experimental.vsa.ops import HolographicOps

        a = rng.standard_normal(dim).astype(np.float32)
        b = rng.standard_normal(dim).astype(np.float32)

        result1 = HolographicOps.convolve(a, b)
        result2 = HolographicOps.convolve(b, a)

        # Allow for floating point precision differences
        np.testing.assert_array_almost_equal(result1, result2, decimal=4)

    def test_convolve_associative(self, unit_vector_triple):
        """Convolve should be associative."""
        from grilly.experimental.vsa.ops import HolographicOps

        a, b, c = unit_vector_triple

        result1 = HolographicOps.convolve(HolographicOps.convolve(a, b), c)
        result2 = HolographicOps.convolve(a, HolographicOps.convolve(b, c))

        np.testing.assert_array_almost_equal(result1, result2, decimal=4)

    def test_bundle_returns_correct_shape(self, dim, rng):
        """Bundle should return a vector of the same dimension."""
        from grilly.experimental.vsa.ops import HolographicOps

        vectors = [rng.standard_normal(dim).astype(np.float32) for _ in range(5)]

        result = HolographicOps.bundle(vectors)

        assert result.shape == (dim,)

    def test_bundle_normalized(self, dim, rng):
        """Bundle with normalize=True should return unit vector."""
        from grilly.experimental.vsa.ops import HolographicOps

        vectors = [rng.standard_normal(dim).astype(np.float32) for _ in range(5)]

        result = HolographicOps.bundle(vectors, normalize=True)

        norm = np.linalg.norm(result)
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_bundle_preserves_components(self, dim, rng):
        """Bundled vector should be similar to each component."""
        from grilly.experimental.vsa.ops import HolographicOps

        vectors = [rng.standard_normal(dim).astype(np.float32) for _ in range(3)]
        # Normalize
        vectors = [v / np.linalg.norm(v) for v in vectors]

        bundled = HolographicOps.bundle(vectors, normalize=True)

        for v in vectors:
            sim = HolographicOps.similarity(bundled, v)
            assert sim > 0.3, f"Expected positive similarity, got {sim}"

    def test_similarity_range(self, dim, rng):
        """Similarity should always be in [-1, 1] for normalized vectors."""
        from grilly.experimental.vsa.ops import HolographicOps

        for _ in range(10):
            a = rng.standard_normal(dim).astype(np.float32)
            a = a / np.linalg.norm(a)
            b = rng.standard_normal(dim).astype(np.float32)
            b = b / np.linalg.norm(b)

            sim = HolographicOps.similarity(a, b)

            assert -1.0 <= sim <= 1.0, f"Similarity {sim} out of range"

    def test_similarity_self(self, dim, rng):
        """Similarity of vector with itself should be 1.0."""
        from grilly.experimental.vsa.ops import HolographicOps

        a = rng.standard_normal(dim).astype(np.float32)
        a = a / np.linalg.norm(a)

        sim = HolographicOps.similarity(a, a)

        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_random_vector_unit_norm(self, dim):
        """Random vector should have unit norm."""
        from grilly.experimental.vsa.ops import HolographicOps

        v = HolographicOps.random_vector(dim)

        norm = np.linalg.norm(v)
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_random_vector_deterministic_with_seed(self, dim):
        """Random vector with same seed should produce same vector."""
        from grilly.experimental.vsa.ops import HolographicOps

        v1 = HolographicOps.random_vector(dim, seed=42)
        v2 = HolographicOps.random_vector(dim, seed=42)

        np.testing.assert_array_almost_equal(v1, v2)

    def test_random_vectors_nearly_orthogonal(self, dim, rng, orthogonality_threshold):
        """Random high-dimensional vectors should be nearly orthogonal."""
        from grilly.experimental.vsa.ops import HolographicOps

        similarities = []
        for i in range(20):
            a = HolographicOps.random_vector(dim, seed=i)
            b = HolographicOps.random_vector(dim, seed=i + 1000)
            similarities.append(abs(HolographicOps.similarity(a, b)))

        avg_similarity = np.mean(similarities)
        assert avg_similarity < orthogonality_threshold, (
            f"Average similarity {avg_similarity} exceeds threshold {orthogonality_threshold}"
        )


class TestBindUnbindRoundTrip:
    """Tests for round-trip binding and unbinding."""

    def test_binary_bind_unbind_roundtrip(self, dim, rng):
        """Binary bind/unbind should perfectly recover original."""
        from grilly.experimental.vsa.ops import BinaryOps

        original = np.sign(rng.standard_normal(dim)).astype(np.float32)
        key = np.sign(rng.standard_normal(dim)).astype(np.float32)

        bound = BinaryOps.bind(original, key)
        recovered = BinaryOps.unbind(bound, key)

        np.testing.assert_array_equal(recovered, original)

    def test_holographic_convolve_correlate_roundtrip(self, dim, rng):
        """Holographic convolve/correlate should approximately recover original."""
        from grilly.experimental.vsa.ops import HolographicOps

        original = HolographicOps.random_vector(dim, seed=42)
        key = HolographicOps.random_vector(dim, seed=43)

        convolved = HolographicOps.convolve(original, key)
        recovered = HolographicOps.correlate(convolved, key)

        # HRR recovery is approximate; similarity should be well above random
        sim = HolographicOps.similarity(recovered, original)
        assert sim > 0.6, f"Expected good similarity recovery, got {sim}"

    def test_multiple_binds_unbinds(self, dim, rng):
        """Multiple binds should be reversible with unbinds in reverse order."""
        from grilly.experimental.vsa.ops import BinaryOps

        original = np.sign(rng.standard_normal(dim)).astype(np.float32)
        key1 = np.sign(rng.standard_normal(dim)).astype(np.float32)
        key2 = np.sign(rng.standard_normal(dim)).astype(np.float32)
        key3 = np.sign(rng.standard_normal(dim)).astype(np.float32)

        # Bind with multiple keys
        bound = BinaryOps.bind(original, key1)
        bound = BinaryOps.bind(bound, key2)
        bound = BinaryOps.bind(bound, key3)

        # Unbind in any order (since bipolar bind is commutative)
        recovered = BinaryOps.unbind(bound, key1)
        recovered = BinaryOps.unbind(recovered, key2)
        recovered = BinaryOps.unbind(recovered, key3)

        np.testing.assert_array_equal(recovered, original)
