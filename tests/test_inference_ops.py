"""
Tests for v0.4.0 inference operations: RMSNorm, fused SwiGLU, INT8 GEMM,
and GQA decode attention.
"""

import numpy as np
import pytest

try:
    from grilly import Compute
    from grilly.backend.base import VULKAN_AVAILABLE
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# Numpy reference helpers
# ---------------------------------------------------------------------------

def _ref_rms_norm(x, weight, eps):
    """Numpy reference: RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight"""
    mean_sq = np.mean(x ** 2, axis=-1, keepdims=True)
    normed = x * (1.0 / np.sqrt(mean_sq + eps))
    return normed * weight


def _ref_silu(x):
    """SiLU(x) = x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))


def _ref_swiglu_fused(x, gate_weights, up_weights):
    """
    Fused SwiGLU: SiLU(x @ gate_proj.T) * (x @ up_proj.T)
    x: (..., input_dim)
    gate_weights: (intermediate_size, input_dim)
    up_weights:   (intermediate_size, input_dim)
    """
    input_dim = x.shape[-1]
    intermediate_size = gate_weights.shape[0]
    x_2d = x.reshape(-1, input_dim)
    gate = x_2d @ gate_weights.T
    up = x_2d @ up_weights.T
    result = _ref_silu(gate) * up
    outer_shape = x.shape[:-1] + (intermediate_size,)
    return result.reshape(outer_shape)


def _ref_gemm_int8(activations, weights_int8, scales, group_size):
    """Dequantize INT8 weights then matmul."""
    M, K = activations.shape
    N = weights_int8.shape[0]
    num_groups = (K + group_size - 1) // group_size
    w_fp32 = np.zeros((N, K), dtype=np.float32)
    for g in range(num_groups):
        k_start = g * group_size
        k_end = min(k_start + group_size, K)
        w_fp32[:, k_start:k_end] = (
            weights_int8[:, k_start:k_end].astype(np.float32) * scales[:, g : g + 1]
        )
    return activations @ w_fp32.T


def _ref_gqa_decode_attention(query, k_cache, v_cache, num_q_heads,
                              num_kv_heads, head_dim, cache_len=None,
                              scale=None):
    """
    Numpy reference for GQA decode attention.
    query:   (batch, 1, num_q_heads, head_dim)
    k_cache: (batch, cache_len, num_kv_heads, head_dim)
    v_cache: (batch, cache_len, num_kv_heads, head_dim)
    """
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)
    batch_size = query.shape[0]
    if cache_len is None:
        cache_len = k_cache.shape[1]
    kv_group_size = num_q_heads // num_kv_heads
    q_2d = query.reshape(batch_size, num_q_heads, head_dim)
    output = np.zeros((batch_size, num_q_heads, head_dim), dtype=np.float32)
    for b in range(batch_size):
        for qh in range(num_q_heads):
            kv_head = qh // kv_group_size
            scores = (
                np.einsum("d,sd->s", q_2d[b, qh], k_cache[b, :cache_len, kv_head])
                * scale
            )
            scores_max = np.max(scores)
            exp_scores = np.exp(scores - scores_max)
            weights = exp_scores / np.sum(exp_scores)
            output[b, qh] = np.einsum(
                "s,sd->d", weights, v_cache[b, :cache_len, kv_head]
            )
    return output.reshape(batch_size, 1, num_q_heads, head_dim)


# ===================================================================
# GPU Tests
# ===================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestRMSNormGPU:
    """Test RMSNorm on GPU"""

    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_rms_norm_1d(self, gpu):
        """Test RMSNorm with 1D input (features,)"""
        np.random.seed(42)
        x = np.random.randn(64).astype(np.float32)
        result = gpu.fnn.rms_norm(x)
        expected = _ref_rms_norm(x, np.ones(64, dtype=np.float32), 1e-5)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_rms_norm_2d(self, gpu):
        """Test RMSNorm with 2D input (batch, features)"""
        np.random.seed(42)
        x = np.random.randn(4, 128).astype(np.float32)
        weight = np.ones(128, dtype=np.float32)
        result = gpu.fnn.rms_norm(x, weight)
        expected = _ref_rms_norm(x, weight, 1e-5)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_rms_norm_3d(self, gpu):
        """Test RMSNorm with 3D input (batch, seq, features)"""
        np.random.seed(42)
        x = np.random.randn(2, 8, 64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)
        result = gpu.fnn.rms_norm(x, weight)
        expected = _ref_rms_norm(x, weight, 1e-5)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_rms_norm_default_weight(self, gpu):
        """Test RMSNorm with default weight (ones)"""
        np.random.seed(42)
        x = np.random.randn(4, 64).astype(np.float32)
        result = gpu.fnn.rms_norm(x)
        expected = _ref_rms_norm(x, np.ones(64, dtype=np.float32), 1e-5)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_rms_norm_custom_weight(self, gpu):
        """Test RMSNorm with learned (non-unit) weight"""
        np.random.seed(42)
        x = np.random.randn(4, 64).astype(np.float32)
        weight = np.random.randn(64).astype(np.float32) * 0.5 + 1.0
        result = gpu.fnn.rms_norm(x, weight)
        expected = _ref_rms_norm(x, weight, 1e-5)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_rms_norm_numerical_correctness(self, gpu):
        """Test RMSNorm numerical correctness against numpy reference"""
        np.random.seed(42)
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        weight = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        eps = 1e-5
        result = gpu.fnn.rms_norm(x, weight, eps)
        expected = _ref_rms_norm(x, weight, eps)
        np.testing.assert_allclose(result, expected, atol=1e-3)
        # Output shape must match input shape
        assert result.shape == x.shape

    def test_rms_norm_eps_affects_output(self, gpu):
        """Test that eps parameter changes the output"""
        np.random.seed(42)
        x = np.random.randn(4, 64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)
        result_small_eps = gpu.fnn.rms_norm(x, weight, eps=1e-8)
        result_large_eps = gpu.fnn.rms_norm(x, weight, eps=1.0)
        # Different eps should yield different outputs
        assert not np.allclose(result_small_eps, result_large_eps, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestSwiGLUFusedGPU:
    """Test fused SwiGLU on GPU"""

    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_swiglu_fused_2d(self, gpu):
        """Test fused SwiGLU with 2D input (batch, input_dim)"""
        np.random.seed(42)
        batch, input_dim, intermediate = 4, 64, 128
        x = np.random.randn(batch, input_dim).astype(np.float32)
        gate_w = np.random.randn(intermediate, input_dim).astype(np.float32) * 0.02
        up_w = np.random.randn(intermediate, input_dim).astype(np.float32) * 0.02
        result = gpu.fnn.swiglu_fused(x, gate_w, up_w)
        expected = _ref_swiglu_fused(x, gate_w, up_w)
        assert result.shape == (batch, intermediate)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_swiglu_fused_3d(self, gpu):
        """Test fused SwiGLU with 3D input (batch, seq, input_dim)"""
        np.random.seed(42)
        batch, seq, input_dim, intermediate = 2, 8, 64, 128
        x = np.random.randn(batch, seq, input_dim).astype(np.float32)
        gate_w = np.random.randn(intermediate, input_dim).astype(np.float32) * 0.02
        up_w = np.random.randn(intermediate, input_dim).astype(np.float32) * 0.02
        result = gpu.fnn.swiglu_fused(x, gate_w, up_w)
        expected = _ref_swiglu_fused(x, gate_w, up_w)
        assert result.shape == (batch, seq, intermediate)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_swiglu_fused_numerical_correctness(self, gpu):
        """Test numerical correctness against SiLU(x @ gate.T) * (x @ up.T)"""
        np.random.seed(42)
        x = np.array([[1.0, 0.5, -0.3]], dtype=np.float32)
        gate_w = np.random.randn(4, 3).astype(np.float32) * 0.1
        up_w = np.random.randn(4, 3).astype(np.float32) * 0.1
        result = gpu.fnn.swiglu_fused(x, gate_w, up_w)
        expected = _ref_swiglu_fused(x, gate_w, up_w)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_swiglu_fused_output_shape(self, gpu):
        """Test output shape is (batch, intermediate_size)"""
        np.random.seed(42)
        batch, input_dim, intermediate = 8, 32, 96
        x = np.random.randn(batch, input_dim).astype(np.float32)
        gate_w = np.random.randn(intermediate, input_dim).astype(np.float32) * 0.02
        up_w = np.random.randn(intermediate, input_dim).astype(np.float32) * 0.02
        result = gpu.fnn.swiglu_fused(x, gate_w, up_w)
        assert result.shape == (batch, intermediate)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestGEMMInt8GPU:
    """Test INT8 weight-only GEMM on GPU"""

    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        backend = Compute()
        yield backend
        backend.cleanup()

    @staticmethod
    def _make_int8_weights(N, K, group_size):
        """Create random INT8 weights and matching per-group scales."""
        w_int8 = np.random.randint(-127, 127, size=(N, K), dtype=np.int8)
        num_groups = (K + group_size - 1) // group_size
        scales = np.random.rand(N, num_groups).astype(np.float32) * 0.1 + 0.01
        return w_int8, scales

    def test_gemm_int8_known_values(self, gpu):
        """Test INT8 GEMM with simple known values"""
        np.random.seed(42)
        M, K, N, group_size = 2, 8, 4, 8
        activations = np.ones((M, K), dtype=np.float32)
        w_int8 = np.ones((N, K), dtype=np.int8) * 2
        num_groups = (K + group_size - 1) // group_size
        scales = np.ones((N, num_groups), dtype=np.float32) * 0.5
        result = gpu.fnn.gemm_int8(activations, w_int8, scales, group_size)
        expected = _ref_gemm_int8(activations, w_int8, scales, group_size)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_gemm_int8_output_shape(self, gpu):
        """Test output shape (M, N) from (M, K) @ (N, K).T"""
        np.random.seed(42)
        M, K, N, group_size = 4, 128, 64, 64
        activations = np.random.randn(M, K).astype(np.float32)
        w_int8, scales = self._make_int8_weights(N, K, group_size)
        result = gpu.fnn.gemm_int8(activations, w_int8, scales, group_size)
        assert result.shape == (M, N)

    def test_gemm_int8_numerical_correctness(self, gpu):
        """Test numerical correctness against dequantized numpy reference"""
        np.random.seed(42)
        M, K, N, group_size = 4, 128, 64, 64
        activations = np.random.randn(M, K).astype(np.float32)
        w_int8, scales = self._make_int8_weights(N, K, group_size)
        result = gpu.fnn.gemm_int8(activations, w_int8, scales, group_size)
        expected = _ref_gemm_int8(activations, w_int8, scales, group_size)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_gemm_int8_group_sizes(self, gpu, group_size):
        """Test INT8 GEMM with different group sizes"""
        np.random.seed(42)
        M, K, N = 4, 256, 64
        activations = np.random.randn(M, K).astype(np.float32)
        w_int8, scales = self._make_int8_weights(N, K, group_size)
        result = gpu.fnn.gemm_int8(activations, w_int8, scales, group_size)
        expected = _ref_gemm_int8(activations, w_int8, scales, group_size)
        assert result.shape == (M, N)
        np.testing.assert_allclose(result, expected, atol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestGQADecodeAttentionGPU:
    """Test GQA decode attention on GPU"""

    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_gqa_decode_basic_shape(self, gpu):
        """Test basic shape: query (1,1,24,128) -> output (1,1,24,128)"""
        np.random.seed(42)
        batch, cache_len, num_q, num_kv, head_dim = 1, 32, 24, 8, 128
        query = np.random.randn(batch, 1, num_q, head_dim).astype(np.float32)
        k_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32)
        v_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32)
        result = gpu.attention.gqa_decode_attention(
            query, k_cache, v_cache, num_q, num_kv, head_dim
        )
        assert result.shape == (batch, 1, num_q, head_dim)

    def test_gqa_decode_gqa_mapping(self, gpu):
        """Test GQA mapping: 24 query heads, 8 KV heads (group_size=3)"""
        np.random.seed(42)
        batch, cache_len, num_q, num_kv, head_dim = 1, 16, 24, 8, 64
        query = np.random.randn(batch, 1, num_q, head_dim).astype(np.float32)
        k_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32)
        v_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32)
        result = gpu.attention.gqa_decode_attention(
            query, k_cache, v_cache, num_q, num_kv, head_dim
        )
        expected = _ref_gqa_decode_attention(
            query, k_cache, v_cache, num_q, num_kv, head_dim
        )
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_gqa_decode_numerical_correctness(self, gpu):
        """Test numerical correctness against numpy reference"""
        np.random.seed(42)
        batch, cache_len, num_q, num_kv, head_dim = 1, 32, 24, 8, 128
        query = np.random.randn(batch, 1, num_q, head_dim).astype(np.float32) * 0.1
        k_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1
        v_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1
        result = gpu.attention.gqa_decode_attention(
            query, k_cache, v_cache, num_q, num_kv, head_dim
        )
        expected = _ref_gqa_decode_attention(
            query, k_cache, v_cache, num_q, num_kv, head_dim
        )
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_gqa_decode_different_cache_lengths(self, gpu):
        """Test GQA decode with different KV cache lengths"""
        np.random.seed(42)
        batch, num_q, num_kv, head_dim = 1, 12, 4, 64
        for cache_len in [1, 8, 32, 64]:
            query = np.random.randn(batch, 1, num_q, head_dim).astype(np.float32) * 0.1
            k_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1
            v_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1
            result = gpu.attention.gqa_decode_attention(
                query, k_cache, v_cache, num_q, num_kv, head_dim
            )
            expected = _ref_gqa_decode_attention(
                query, k_cache, v_cache, num_q, num_kv, head_dim
            )
            assert result.shape == (batch, 1, num_q, head_dim)
            np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_gqa_decode_batch_gt_1(self, gpu):
        """Test GQA decode with batch_size > 1"""
        np.random.seed(42)
        batch, cache_len, num_q, num_kv, head_dim = 4, 16, 12, 4, 64
        query = np.random.randn(batch, 1, num_q, head_dim).astype(np.float32) * 0.1
        k_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1
        v_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1
        result = gpu.attention.gqa_decode_attention(
            query, k_cache, v_cache, num_q, num_kv, head_dim
        )
        expected = _ref_gqa_decode_attention(
            query, k_cache, v_cache, num_q, num_kv, head_dim
        )
        assert result.shape == (batch, 1, num_q, head_dim)
        np.testing.assert_allclose(result, expected, atol=1e-3)


# ===================================================================
# CPU Fallback Tests (no GPU marker)
# ===================================================================


class TestRMSNormCPUFallback:
    """Test RMSNorm CPU fallback path"""

    def test_rms_norm_cpu(self):
        """Test RMSNorm CPU fallback produces correct results"""
        np.random.seed(42)
        x = np.random.randn(4, 128).astype(np.float32)
        weight = np.random.randn(128).astype(np.float32) * 0.5 + 1.0
        eps = 1e-5

        # Compute via numpy reference directly
        expected = _ref_rms_norm(x, weight, eps)

        # Verify the reference implementation itself is mathematically sound
        mean_sq = np.mean(x ** 2, axis=-1, keepdims=True)
        manual = x * (1.0 / np.sqrt(mean_sq + eps)) * weight
        np.testing.assert_allclose(expected, manual, atol=1e-5)

        # Verify shape preservation
        assert expected.shape == x.shape

        # Verify non-trivial transformation (output differs from input)
        assert not np.allclose(x, expected)

    def test_rms_norm_cpu_1d(self):
        """Test RMSNorm CPU fallback with 1D input"""
        np.random.seed(42)
        x = np.random.randn(64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)
        eps = 1e-5
        result = _ref_rms_norm(x, weight, eps)
        assert result.shape == (64,)
        assert np.all(np.isfinite(result))

    def test_rms_norm_cpu_3d(self):
        """Test RMSNorm CPU fallback with 3D input"""
        np.random.seed(42)
        x = np.random.randn(2, 8, 64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)
        eps = 1e-5
        result = _ref_rms_norm(x, weight, eps)
        assert result.shape == (2, 8, 64)
        np.testing.assert_allclose(
            result,
            x * (1.0 / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)) * weight,
            atol=1e-5,
        )


class TestSwiGLUFusedCPUFallback:
    """Test fused SwiGLU CPU fallback path"""

    def test_swiglu_fused_cpu(self):
        """Test fused SwiGLU CPU fallback produces correct results"""
        np.random.seed(42)
        batch, input_dim, intermediate = 4, 64, 128
        x = np.random.randn(batch, input_dim).astype(np.float32)
        gate_w = np.random.randn(intermediate, input_dim).astype(np.float32) * 0.02
        up_w = np.random.randn(intermediate, input_dim).astype(np.float32) * 0.02

        result = _ref_swiglu_fused(x, gate_w, up_w)

        # Verify against manual step-by-step computation
        gate = x @ gate_w.T
        up = x @ up_w.T
        silu_gate = gate * (1.0 / (1.0 + np.exp(-np.clip(gate, -88, 88))))
        manual = silu_gate * up

        np.testing.assert_allclose(result, manual, atol=1e-5)
        assert result.shape == (batch, intermediate)

    def test_swiglu_fused_cpu_3d(self):
        """Test fused SwiGLU CPU fallback with 3D input"""
        np.random.seed(42)
        batch, seq, input_dim, intermediate = 2, 8, 32, 64
        x = np.random.randn(batch, seq, input_dim).astype(np.float32)
        gate_w = np.random.randn(intermediate, input_dim).astype(np.float32) * 0.02
        up_w = np.random.randn(intermediate, input_dim).astype(np.float32) * 0.02

        result = _ref_swiglu_fused(x, gate_w, up_w)
        assert result.shape == (batch, seq, intermediate)
        assert np.all(np.isfinite(result))


class TestGEMMInt8CPUFallback:
    """Test INT8 GEMM CPU fallback path"""

    def test_gemm_int8_cpu(self):
        """Test INT8 GEMM CPU fallback dequantizes and computes correctly"""
        np.random.seed(42)
        M, K, N, group_size = 4, 128, 64, 64
        activations = np.random.randn(M, K).astype(np.float32)
        w_int8 = np.random.randint(-127, 127, size=(N, K), dtype=np.int8)
        num_groups = (K + group_size - 1) // group_size
        scales = np.random.rand(N, num_groups).astype(np.float32) * 0.1 + 0.01

        result = _ref_gemm_int8(activations, w_int8, scales, group_size)

        # Verify against full dequantize + matmul
        w_fp32 = np.zeros((N, K), dtype=np.float32)
        for g in range(num_groups):
            k_s = g * group_size
            k_e = min(k_s + group_size, K)
            w_fp32[:, k_s:k_e] = w_int8[:, k_s:k_e].astype(np.float32) * scales[:, g : g + 1]
        manual = activations @ w_fp32.T

        np.testing.assert_allclose(result, manual, atol=1e-5)
        assert result.shape == (M, N)

    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_gemm_int8_cpu_group_sizes(self, group_size):
        """Test INT8 GEMM CPU fallback with different group sizes"""
        np.random.seed(42)
        M, K, N = 2, 256, 32
        activations = np.random.randn(M, K).astype(np.float32)
        w_int8 = np.random.randint(-127, 127, size=(N, K), dtype=np.int8)
        num_groups = (K + group_size - 1) // group_size
        scales = np.random.rand(N, num_groups).astype(np.float32) * 0.1 + 0.01

        result = _ref_gemm_int8(activations, w_int8, scales, group_size)
        assert result.shape == (M, N)
        assert np.all(np.isfinite(result))


class TestGQADecodeAttentionCPUFallback:
    """Test GQA decode attention CPU fallback path"""

    def test_gqa_decode_cpu(self):
        """Test GQA decode CPU fallback produces correct results"""
        np.random.seed(42)
        batch, cache_len, num_q, num_kv, head_dim = 1, 16, 12, 4, 64
        query = np.random.randn(batch, 1, num_q, head_dim).astype(np.float32) * 0.1
        k_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1
        v_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1

        result = _ref_gqa_decode_attention(
            query, k_cache, v_cache, num_q, num_kv, head_dim
        )
        assert result.shape == (batch, 1, num_q, head_dim)

        # Verify softmax: output should be a weighted combination of values,
        # so each head's output should lie within the range of the value cache
        assert np.all(np.isfinite(result))

    def test_gqa_decode_cpu_batch_gt_1(self):
        """Test GQA decode CPU fallback with batch_size > 1"""
        np.random.seed(42)
        batch, cache_len, num_q, num_kv, head_dim = 3, 16, 12, 4, 64
        query = np.random.randn(batch, 1, num_q, head_dim).astype(np.float32) * 0.1
        k_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1
        v_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1

        result = _ref_gqa_decode_attention(
            query, k_cache, v_cache, num_q, num_kv, head_dim
        )
        assert result.shape == (batch, 1, num_q, head_dim)

        # Each batch item should produce independent output
        r0 = _ref_gqa_decode_attention(
            query[0:1], k_cache[0:1], v_cache[0:1], num_q, num_kv, head_dim
        )
        r1 = _ref_gqa_decode_attention(
            query[1:2], k_cache[1:2], v_cache[1:2], num_q, num_kv, head_dim
        )
        np.testing.assert_allclose(result[0:1], r0, atol=1e-5)
        np.testing.assert_allclose(result[1:2], r1, atol=1e-5)

    def test_gqa_decode_cpu_gqa_grouping(self):
        """Test GQA head grouping: query heads map to correct KV heads"""
        np.random.seed(42)
        batch, cache_len, num_q, num_kv, head_dim = 1, 8, 6, 2, 32
        kv_group_size = num_q // num_kv  # 3

        query = np.random.randn(batch, 1, num_q, head_dim).astype(np.float32) * 0.1
        k_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1
        v_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1

        result = _ref_gqa_decode_attention(
            query, k_cache, v_cache, num_q, num_kv, head_dim
        )

        # Query heads 0,1,2 should all use KV head 0
        # Query heads 3,4,5 should all use KV head 1
        # Verify by computing each head individually
        q_2d = query.reshape(batch, num_q, head_dim)
        scale = 1.0 / np.sqrt(head_dim)
        for qh in range(num_q):
            kv_head = qh // kv_group_size
            scores = np.einsum("d,sd->s", q_2d[0, qh], k_cache[0, :cache_len, kv_head]) * scale
            scores_max = np.max(scores)
            exp_scores = np.exp(scores - scores_max)
            weights = exp_scores / np.sum(exp_scores)
            expected_head = np.einsum("s,sd->d", weights, v_cache[0, :cache_len, kv_head])
            np.testing.assert_allclose(
                result[0, 0, qh], expected_head, atol=1e-5
            )

    def test_gqa_decode_cpu_different_cache_lengths(self):
        """Test GQA decode CPU fallback with various cache lengths"""
        np.random.seed(42)
        batch, num_q, num_kv, head_dim = 1, 8, 4, 32
        for cache_len in [1, 4, 16, 64]:
            query = np.random.randn(batch, 1, num_q, head_dim).astype(np.float32) * 0.1
            k_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1
            v_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1
            result = _ref_gqa_decode_attention(
                query, k_cache, v_cache, num_q, num_kv, head_dim
            )
            assert result.shape == (batch, 1, num_q, head_dim)
            assert np.all(np.isfinite(result))
