"""
GPU GEMM-based similarity tests (RDNA2-friendly tiled kernel).

These tests validate that vsa-similarity-gemm matches the CPU dot baseline.
"""

import numpy as np
import pytest


@pytest.mark.gpu
class TestVulkanVSASimilarityGEMM:
    def test_similarity_matrix_gemm_matches_cpu(self, small_dim, rng):
        from grilly.backend.core import VulkanCore
        from grilly.backend.experimental.vsa import VulkanVSA

        try:
            core = VulkanCore()
            vsa = VulkanVSA(core)

            if "vsa-similarity-gemm" not in core.shaders:
                pytest.skip("vsa-similarity-gemm shader not available")

            B = 4
            N = 64
            D = small_dim

            queries = np.sign(rng.standard_normal((B, D))).astype(np.float32)
            codebook = np.sign(rng.standard_normal((N, D))).astype(np.float32)

            cpu = (queries @ codebook.T) / float(D)
            gpu = vsa.similarity_matrix_gemm(queries, codebook, divide_by_dim=True)

            np.testing.assert_allclose(gpu, cpu.astype(np.float32), rtol=1e-4, atol=1e-4)

        except RuntimeError:
            pytest.skip("Vulkan not available")
