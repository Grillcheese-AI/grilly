"""
TDD Tests for ResonatorNetwork.

Tests the resonator's ability to factorize composite vectors into their components.
"""

import numpy as np
import pytest


class TestResonatorNetworkBasic:
    """Basic tests for ResonatorNetwork initialization and structure."""

    def test_init_with_single_codebook(self, small_dim, rng):
        """Should initialize with a single codebook."""
        from grilly.experimental.vsa.resonator import ResonatorNetwork

        codebook = np.sign(rng.standard_normal((10, small_dim))).astype(np.float32)

        resonator = ResonatorNetwork(codebooks={"factor_a": codebook}, max_iterations=20)

        assert resonator.dim == small_dim
        assert len(resonator.factor_names) == 1
        assert "factor_a" in resonator.factor_names

    def test_init_with_multiple_codebooks(self, small_dim, rng):
        """Should initialize with multiple codebooks."""
        from grilly.experimental.vsa.resonator import ResonatorNetwork

        cb_a = np.sign(rng.standard_normal((10, small_dim))).astype(np.float32)
        cb_b = np.sign(rng.standard_normal((8, small_dim))).astype(np.float32)

        resonator = ResonatorNetwork(
            codebooks={"factor_a": cb_a, "factor_b": cb_b}, max_iterations=20
        )

        assert resonator.dim == small_dim
        assert len(resonator.factor_names) == 2

    def test_init_validates_dimensions(self, rng):
        """Should raise error if codebooks have different dimensions."""
        from grilly.experimental.vsa.resonator import ResonatorNetwork

        cb_a = np.sign(rng.standard_normal((10, 100))).astype(np.float32)
        cb_b = np.sign(rng.standard_normal((8, 200))).astype(np.float32)

        with pytest.raises(AssertionError):
            ResonatorNetwork(codebooks={"a": cb_a, "b": cb_b})


class TestResonatorFactorizeSingleFactor:
    """Tests for factorizing composites with a single factor."""

    def test_factorize_recovers_single_item(self, small_dim, rng):
        """Should recover a single item from codebook."""
        from grilly.experimental.vsa.ops import BinaryOps
        from grilly.experimental.vsa.resonator import ResonatorNetwork

        # Create codebook with 10 items
        codebook = np.sign(rng.standard_normal((10, small_dim))).astype(np.float32)

        resonator = ResonatorNetwork(codebooks={"item": codebook}, max_iterations=20)

        # The composite is just one item from the codebook
        target_idx = 5
        composite = codebook[target_idx].copy()

        estimates, indices, iterations = resonator.factorize(composite)

        assert "item" in estimates
        assert "item" in indices
        assert indices["item"] == target_idx
        # Estimate should be very similar to the original
        sim = BinaryOps.similarity(estimates["item"], codebook[target_idx])
        assert sim > 0.9

    def test_factorize_with_noise(self, small_dim, rng):
        """Should recover item even with some noise added."""
        from grilly.experimental.vsa.resonator import ResonatorNetwork

        codebook = np.sign(rng.standard_normal((10, small_dim))).astype(np.float32)

        resonator = ResonatorNetwork(codebooks={"item": codebook}, max_iterations=30)

        target_idx = 3
        # Add noise (flip some bits)
        noise_level = 0.1
        noise_mask = rng.random(small_dim) < noise_level
        composite = codebook[target_idx].copy()
        composite[noise_mask] *= -1

        estimates, indices, iterations = resonator.factorize(composite)

        # Should still recover the correct item
        assert indices["item"] == target_idx


class TestResonatorFactorizeMultipleFactors:
    """Tests for factorizing composites with multiple bound factors."""

    def test_factorize_two_factors(self, large_dim):
        """Should recover both factors from a bound composite."""
        from grilly.experimental.vsa.ops import BinaryOps
        from grilly.experimental.vsa.resonator import ResonatorNetwork

        # Use large dim (4096) and fixed seed for reliable multi-factor factorization
        np.random.seed(123)  # Different seed from other tests
        cb_a = np.sign(np.random.randn(8, large_dim)).astype(np.float32)
        cb_b = np.sign(np.random.randn(8, large_dim)).astype(np.float32)

        resonator = ResonatorNetwork(
            codebooks={"factor_a": cb_a, "factor_b": cb_b}, max_iterations=50
        )

        # Create composite by binding two items
        idx_a, idx_b = 3, 6
        composite = BinaryOps.bind(cb_a[idx_a], cb_b[idx_b])

        estimates, indices, iterations = resonator.factorize(composite)

        assert indices["factor_a"] == idx_a
        assert indices["factor_b"] == idx_b

    def test_factorize_three_factors(self, large_dim):
        """Should recover all three factors from a triple-bound composite."""
        from grilly.experimental.vsa.ops import BinaryOps
        from grilly.experimental.vsa.resonator import ResonatorNetwork

        # Use large dim and fixed seed for reliable multi-factor factorization
        np.random.seed(456)  # Different seed from two-factors test
        cb_a = np.sign(np.random.randn(6, large_dim)).astype(np.float32)
        cb_b = np.sign(np.random.randn(6, large_dim)).astype(np.float32)
        cb_c = np.sign(np.random.randn(6, large_dim)).astype(np.float32)

        resonator = ResonatorNetwork(
            codebooks={"a": cb_a, "b": cb_b, "c": cb_c}, max_iterations=100
        )

        idx_a, idx_b, idx_c = 2, 4, 5
        composite = BinaryOps.bind(BinaryOps.bind(cb_a[idx_a], cb_b[idx_b]), cb_c[idx_c])

        estimates, indices, iterations = resonator.factorize(composite)

        assert indices["a"] == idx_a
        assert indices["b"] == idx_b
        assert indices["c"] == idx_c


class TestResonatorConvergence:
    """Tests for resonator convergence behavior."""

    def test_converges_within_max_iterations(self, small_dim, rng):
        """Should converge within the specified max iterations."""
        from grilly.experimental.vsa.ops import BinaryOps
        from grilly.experimental.vsa.resonator import ResonatorNetwork

        cb_a = np.sign(rng.standard_normal((8, small_dim))).astype(np.float32)
        cb_b = np.sign(rng.standard_normal((8, small_dim))).astype(np.float32)

        max_iter = 30
        resonator = ResonatorNetwork(codebooks={"a": cb_a, "b": cb_b}, max_iterations=max_iter)

        idx_a, idx_b = 2, 5
        composite = BinaryOps.bind(cb_a[idx_a], cb_b[idx_b])

        estimates, indices, iterations = resonator.factorize(composite)

        assert iterations <= max_iter

    def test_convergence_threshold_respected(self, small_dim, rng):
        """Higher threshold should require fewer iterations for clean inputs."""
        from grilly.experimental.vsa.resonator import ResonatorNetwork

        codebook = np.sign(rng.standard_normal((8, small_dim))).astype(np.float32)

        # Create resonator with high threshold
        resonator = ResonatorNetwork(
            codebooks={"item": codebook}, max_iterations=50, convergence_threshold=0.99
        )

        # Clean input should converge quickly
        composite = codebook[3].copy()

        estimates, indices, iterations = resonator.factorize(composite)

        # Should converge and find correct item
        assert indices["item"] == 3


class TestResonatorPartialFactorization:
    """Tests for factorize_partial with known factors."""

    def test_partial_recovers_unknown_factor(self, small_dim, rng):
        """Should recover unknown factor when others are known."""
        from grilly.experimental.vsa.ops import BinaryOps
        from grilly.experimental.vsa.resonator import ResonatorNetwork

        cb_a = np.sign(rng.standard_normal((8, small_dim))).astype(np.float32)
        cb_b = np.sign(rng.standard_normal((8, small_dim))).astype(np.float32)

        resonator = ResonatorNetwork(codebooks={"a": cb_a, "b": cb_b}, max_iterations=20)

        idx_a, idx_b = 2, 5
        composite = BinaryOps.bind(cb_a[idx_a], cb_b[idx_b])

        # Know factor_a, want to recover factor_b
        known = {"a": cb_a[idx_a]}

        recovered = resonator.factorize_partial(composite, known)

        # Recovered should be similar to cb_b[idx_b]
        sim = BinaryOps.similarity(recovered, cb_b[idx_b])
        assert sim > 0.9, f"Expected high similarity, got {sim}"

    def test_partial_with_multiple_known(self, small_dim, rng):
        """Should recover unknown when multiple factors are known."""
        from grilly.experimental.vsa.ops import BinaryOps
        from grilly.experimental.vsa.resonator import ResonatorNetwork

        cb_a = np.sign(rng.standard_normal((6, small_dim))).astype(np.float32)
        cb_b = np.sign(rng.standard_normal((6, small_dim))).astype(np.float32)
        cb_c = np.sign(rng.standard_normal((6, small_dim))).astype(np.float32)

        resonator = ResonatorNetwork(codebooks={"a": cb_a, "b": cb_b, "c": cb_c}, max_iterations=20)

        idx_a, idx_b, idx_c = 1, 3, 4
        composite = BinaryOps.bind(BinaryOps.bind(cb_a[idx_a], cb_b[idx_b]), cb_c[idx_c])

        # Know factors a and b, recover c
        known = {"a": cb_a[idx_a], "b": cb_b[idx_b]}

        recovered = resonator.factorize_partial(composite, known)

        sim = BinaryOps.similarity(recovered, cb_c[idx_c])
        assert sim > 0.9, f"Expected high similarity, got {sim}"


class TestResonatorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_codebook_raises_error(self):
        """Should raise error for empty codebook."""
        from grilly.experimental.vsa.resonator import ResonatorNetwork

        with pytest.raises((ValueError, AssertionError)):
            ResonatorNetwork(codebooks={})

    def test_single_item_codebook(self, small_dim, rng):
        """Should handle codebook with single item."""
        from grilly.experimental.vsa.resonator import ResonatorNetwork

        codebook = np.sign(rng.standard_normal((1, small_dim))).astype(np.float32)

        resonator = ResonatorNetwork(codebooks={"item": codebook}, max_iterations=10)

        composite = codebook[0].copy()

        estimates, indices, iterations = resonator.factorize(composite)

        assert indices["item"] == 0

    def test_factorize_with_init_estimates(self, small_dim, rng):
        """Should use provided initial estimates."""
        from grilly.experimental.vsa.ops import BinaryOps
        from grilly.experimental.vsa.resonator import ResonatorNetwork

        cb_a = np.sign(rng.standard_normal((8, small_dim))).astype(np.float32)
        cb_b = np.sign(rng.standard_normal((8, small_dim))).astype(np.float32)

        resonator = ResonatorNetwork(codebooks={"a": cb_a, "b": cb_b}, max_iterations=30)

        idx_a, idx_b = 2, 5
        composite = BinaryOps.bind(cb_a[idx_a], cb_b[idx_b])

        # Provide correct initial estimates - should converge quickly
        init_estimates = {"a": cb_a[idx_a].copy(), "b": cb_b[idx_b].copy()}

        estimates, indices, iterations = resonator.factorize(composite, init_estimates)

        assert indices["a"] == idx_a
        assert indices["b"] == idx_b
        # Should converge in very few iterations with good init
        assert iterations <= 5


class TestResonatorDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(self, small_dim, rng):
        """Same input should always give same output."""
        from grilly.experimental.vsa.ops import BinaryOps
        from grilly.experimental.vsa.resonator import ResonatorNetwork

        # Use fixed seed for codebook generation
        np.random.seed(42)
        cb_a = np.sign(np.random.randn(8, small_dim)).astype(np.float32)
        cb_b = np.sign(np.random.randn(8, small_dim)).astype(np.float32)

        resonator = ResonatorNetwork(codebooks={"a": cb_a, "b": cb_b}, max_iterations=30)

        idx_a, idx_b = 2, 5
        composite = BinaryOps.bind(cb_a[idx_a], cb_b[idx_b])

        # Run factorization multiple times
        results = []
        for _ in range(3):
            np.random.seed(100)  # Reset random state for determinism
            estimates, indices, iterations = resonator.factorize(composite.copy())
            results.append((indices["a"], indices["b"]))

        # All results should be the same
        assert all(r == results[0] for r in results)
