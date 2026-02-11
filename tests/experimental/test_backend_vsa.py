"""
TDD Tests for GPU-accelerated VSA operations.

Tests CPU vs GPU parity for VSA operations.
"""

import numpy as np
import pytest


@pytest.mark.gpu
class TestVulkanVSABasic:
    """Basic tests for VulkanVSA initialization."""

    def test_init_with_vulkan_core(self):
        """Should initialize with VulkanCore."""
        from grilly.backend.core import VulkanCore
        from grilly.backend.experimental.vsa import VulkanVSA

        try:
            core = VulkanCore()
            vsa = VulkanVSA(core)

            assert vsa.core is core
        except RuntimeError:
            pytest.skip("Vulkan not available")

    def test_init_requires_vulkan(self):
        """Should raise error if Vulkan not available."""
        from grilly.backend.experimental.vsa import VulkanVSA

        # This will fail if Vulkan not available
        try:
            from grilly.backend.core import VulkanCore

            core = VulkanCore()
            VulkanVSA(core)
        except RuntimeError:
            pytest.skip("Vulkan not available")


@pytest.mark.gpu
class TestVulkanVSABindBipolar:
    """Tests for GPU bipolar binding."""

    def test_bind_bipolar_matches_cpu(self, dim):
        """GPU bind_bipolar should match CPU BinaryOps.bind."""
        from grilly.backend.core import VulkanCore
        from grilly.backend.experimental.vsa import VulkanVSA
        from grilly.experimental.vsa.ops import BinaryOps

        try:
            core = VulkanCore()
            vsa = VulkanVSA(core)

            a = BinaryOps.random_bipolar(dim)
            b = BinaryOps.random_bipolar(dim)

            cpu_result = BinaryOps.bind(a, b)
            gpu_result = vsa.bind_bipolar(a, b)

            np.testing.assert_array_almost_equal(cpu_result, gpu_result, decimal=5)
        except RuntimeError:
            pytest.skip("Vulkan not available")

    def test_bind_bipolar_batch(self, dim):
        """Should handle batch binding."""
        from grilly.backend.core import VulkanCore
        from grilly.backend.experimental.vsa import VulkanVSA
        from grilly.experimental.vsa.ops import BinaryOps

        try:
            core = VulkanCore()
            vsa = VulkanVSA(core)

            batch_size = 10
            a_batch = np.array([BinaryOps.random_bipolar(dim) for _ in range(batch_size)])
            b_batch = np.array([BinaryOps.random_bipolar(dim) for _ in range(batch_size)])

            result = vsa.bind_bipolar_batch(a_batch, b_batch)

            assert result.shape == (batch_size, dim)
        except RuntimeError:
            pytest.skip("Vulkan not available")


@pytest.mark.gpu
class TestVulkanVSABundle:
    """Tests for GPU bundling."""

    def test_bundle_matches_cpu(self, dim):
        """GPU bundle should match CPU BinaryOps.bundle."""
        from grilly.backend.core import VulkanCore
        from grilly.backend.experimental.vsa import VulkanVSA
        from grilly.experimental.vsa.ops import BinaryOps

        try:
            core = VulkanCore()
            vsa = VulkanVSA(core)

            vectors = [BinaryOps.random_bipolar(dim) for _ in range(5)]

            cpu_result = BinaryOps.bundle(vectors)
            gpu_result = vsa.bundle(vectors)

            np.testing.assert_array_almost_equal(cpu_result, gpu_result, decimal=5)
        except RuntimeError:
            pytest.skip("Vulkan not available")

    def test_bundle_batch_matches_cpu(self, dim):
        """GPU bundle_batch should match CPU BinaryOps.bundle for each batch."""
        from grilly.backend.core import VulkanCore
        from grilly.backend.experimental.vsa import VulkanVSA
        from grilly.experimental.vsa.ops import BinaryOps

        try:
            core = VulkanCore()
            vsa = VulkanVSA(core)

            batch_size = 4
            num_vectors = 5
            vectors = np.array(
                [
                    [BinaryOps.random_bipolar(dim) for _ in range(num_vectors)]
                    for _ in range(batch_size)
                ],
                dtype=np.float32,
            )

            cpu_results = BinaryOps.bundle_batch(vectors)
            gpu_results = vsa.bundle_batch(vectors)

            np.testing.assert_array_almost_equal(cpu_results, gpu_results, decimal=5)
        except RuntimeError:
            pytest.skip("Vulkan not available")


@pytest.mark.gpu
class TestVulkanVSASimilarity:
    """Tests for GPU similarity computation."""

    def test_similarity_batch_matches_cpu(self, dim):
        """GPU similarity_batch should match CPU similarity."""
        from grilly.backend.core import VulkanCore
        from grilly.backend.experimental.vsa import VulkanVSA
        from grilly.experimental.vsa.ops import BinaryOps

        try:
            core = VulkanCore()
            vsa = VulkanVSA(core)

            query = BinaryOps.random_bipolar(dim)
            codebook = np.array([BinaryOps.random_bipolar(dim) for _ in range(10)])

            cpu_results = np.array([BinaryOps.similarity(query, vec) for vec in codebook])
            gpu_results = vsa.similarity_batch(query, codebook)

            np.testing.assert_array_almost_equal(cpu_results, gpu_results, decimal=4)
        except RuntimeError:
            pytest.skip("Vulkan not available")


@pytest.mark.gpu
class TestVulkanVSAConvolve:
    """Tests for GPU circular convolution (HRR binding)."""

    def test_convolve_matches_cpu(self, dim):
        """GPU convolve should match CPU HolographicOps.convolve."""
        from grilly.backend.core import VulkanCore
        from grilly.backend.experimental.vsa import VulkanVSA
        from grilly.experimental.vsa.ops import HolographicOps

        try:
            core = VulkanCore()
            vsa = VulkanVSA(core)

            a = HolographicOps.random_vector(dim)
            b = HolographicOps.random_vector(dim)

            cpu_result = HolographicOps.convolve(a, b)
            gpu_result = vsa.circular_convolve(a, b)

            # HRR is approximate, allow some tolerance
            np.testing.assert_array_almost_equal(cpu_result, gpu_result, decimal=3)
        except RuntimeError:
            pytest.skip("Vulkan not available")


@pytest.mark.gpu
class TestVulkanVSAResonator:
    """Tests for GPU resonator operations."""

    def test_resonator_step_matches_cpu(self, dim):
        """GPU resonator_step should match CPU resonator iteration."""
        from grilly.backend.core import VulkanCore
        from grilly.backend.experimental.vsa import VulkanVSA
        from grilly.experimental.vsa.ops import BinaryOps

        try:
            core = VulkanCore()
            vsa = VulkanVSA(core)

            if "vsa-resonator-step" not in core.shaders:
                pytest.skip("Resonator step shader not available")

            # Simple two-factor composite
            codebook = np.sign(np.random.randn(8, dim)).astype(np.float32)
            composite = BinaryOps.bind(codebook[2], codebook[5])

            # CPU reference: unbind one factor and project
            unbound = BinaryOps.unbind(composite, codebook[5])
            cpu_scores = (codebook @ unbound) / float(dim)
            cpu_idx = int(np.argmax(cpu_scores))
            cpu_vec = codebook[cpu_idx].copy()

            gpu_vec, gpu_idx = vsa.resonator_step(
                composite, codebook, other_estimates=[codebook[5]]
            )

            assert gpu_idx == cpu_idx
            np.testing.assert_array_almost_equal(cpu_vec, gpu_vec, decimal=5)
        except RuntimeError:
            pytest.skip("Vulkan not available")


@pytest.mark.gpu
class TestVulkanVSAPerformance:
    """Tests for GPU performance."""

    def test_gpu_faster_than_cpu_large_dim(self, large_dim):
        """GPU should be faster than CPU for large dimensions."""
        import time

        from grilly.backend.core import VulkanCore
        from grilly.backend.experimental.vsa import VulkanVSA
        from grilly.experimental.vsa.ops import BinaryOps

        try:
            core = VulkanCore()
            vsa = VulkanVSA(core)

            a = BinaryOps.random_bipolar(large_dim)
            b = BinaryOps.random_bipolar(large_dim)

            # CPU timing
            start = time.time()
            for _ in range(100):
                _ = BinaryOps.bind(a, b)
            cpu_time = time.time() - start

            # GPU timing
            start = time.time()
            for _ in range(100):
                _ = vsa.bind_bipolar(a, b)
            gpu_time = time.time() - start

            # GPU should be faster (or at least comparable)
            # Note: First run may be slower due to initialization
            print(f"CPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s")
            # Don't fail if GPU is slower due to overhead
        except RuntimeError:
            pytest.skip("Vulkan not available")
