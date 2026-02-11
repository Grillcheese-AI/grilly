"""
GPU Top-K tests (GEMM -> argmax rows -> mask).

Validates that similarity_topk_gemm returns the same top-1 as CPU argmax.
"""

import numpy as np
import pytest


@pytest.mark.gpu
class TestVulkanVSATopK:
    def test_top1_matches_cpu(self, small_dim, rng):
        from grilly.backend.core import VulkanCore
        from grilly.backend.experimental.vsa import VulkanVSA

        try:
            core = VulkanCore()
            vsa = VulkanVSA(core)

            needed = {"vsa-similarity-gemm", "vsa-argmax-rows", "vsa-mask-selected"}
            if not needed.issubset(set(core.shaders.keys())):
                pytest.skip("Top-k shaders not available")

            B = 8
            N = 128
            D = small_dim

            queries = np.sign(rng.standard_normal((B, D))).astype(np.float32)
            codebook = np.sign(rng.standard_normal((N, D))).astype(np.float32)

            cpu_scores = (queries @ codebook.T) / float(D)
            cpu_idx = np.argmax(cpu_scores, axis=1).astype(np.uint32)

            idx, val = vsa.similarity_topk_gemm(queries, codebook, top_k=1)
            gpu_idx = idx[:, 0]

            assert gpu_idx.shape == (B,)
            np.testing.assert_array_equal(gpu_idx, cpu_idx)

        except RuntimeError:
            pytest.skip("Vulkan not available")

    def test_topk_shape(self, small_dim, rng):
        from grilly.backend.core import VulkanCore
        from grilly.backend.experimental.vsa import VulkanVSA

        try:
            core = VulkanCore()
            vsa = VulkanVSA(core)

            needed = {"vsa-similarity-gemm", "vsa-argmax-rows", "vsa-mask-selected"}
            if not needed.issubset(set(core.shaders.keys())):
                pytest.skip("Top-k shaders not available")

            B = 3
            N = 50
            D = small_dim

            queries = np.sign(rng.standard_normal((B, D))).astype(np.float32)
            codebook = np.sign(rng.standard_normal((N, D))).astype(np.float32)

            k = 4
            idx, val = vsa.similarity_topk_gemm(queries, codebook, top_k=k)

            assert idx.shape == (B, k)
            assert val.shape == (B, k)
            assert idx.dtype == np.uint32
            assert val.dtype == np.float32

        except RuntimeError:
            pytest.skip("Vulkan not available")
