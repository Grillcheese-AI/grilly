"""
Tests for GPU operations (requires Vulkan)
"""

import numpy as np
import pytest

try:
    from grilly import Compute
    from grilly.backend.base import VULKAN_AVAILABLE
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestLIFOperations:
    """Test LIF neuron operations on GPU"""

    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_lif_single_neuron(self, gpu):
        """Test single LIF neuron on GPU"""
        input_current = np.array([0.5], dtype=np.float32)
        membrane = np.array([0.3], dtype=np.float32)
        refractory = np.array([0.0], dtype=np.float32)

        mem_out, ref_out, spikes = gpu.lif_step(
            input_current, membrane, refractory, dt=0.001, tau_mem=20.0, v_thresh=1.0
        )

        # Should integrate but not spike with this input
        assert spikes[0] == 0.0
        assert mem_out[0] > membrane[0]  # Membrane increased

    def test_lif_spike_threshold(self, gpu):
        """Test LIF neuron spikes above threshold"""
        input_current = np.array([20.0], dtype=np.float32)
        membrane = np.array([0.5], dtype=np.float32)
        refractory = np.array([0.0], dtype=np.float32)

        mem_out, ref_out, spikes = gpu.lif_step(
            input_current, membrane, refractory, dt=0.6, tau_mem=20.0, v_thresh=1.0
        )

        # Should spike and reset
        assert spikes[0] == 1.0
        assert mem_out[0] == 0.0  # Reset
        assert ref_out[0] > 0.0  # In refractory period

    def test_lif_batch(self, gpu):
        """Test batch of neurons"""
        n = 100
        input_current = np.random.randn(n).astype(np.float32) * 0.5
        membrane = np.random.rand(n).astype(np.float32) * 0.5
        refractory = np.zeros(n, dtype=np.float32)

        mem_out, ref_out, spikes = gpu.lif_step(input_current, membrane, refractory)

        spike_count = spikes.sum()

        assert spikes.shape == (n,)
        assert 0 <= spike_count <= n

    def test_lif_refractory_period(self, gpu):
        """Test refractory period prevents spikes"""
        input_current = np.array([1.0], dtype=np.float32)
        membrane = np.array([0.0], dtype=np.float32)
        refractory = np.array([2.0], dtype=np.float32)  # In refractory

        mem_out, ref_out, spikes = gpu.lif_step(input_current, membrane, refractory, dt=0.001)

        # Should not spike
        assert spikes[0] == 0.0
        assert mem_out[0] == 0.0  # Held at reset
        assert ref_out[0] < refractory[0]  # Refractory decreased


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestFNNOperations:
    """Test FNN operations on GPU"""

    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_activation_relu(self, gpu):
        """Test ReLU activation"""
        input_data = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)

        output = gpu.activation_relu(input_data)

        expected = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_activation_gelu(self, gpu):
        """Test GELU activation"""
        input_data = np.array([0.0, 1.0, -1.0], dtype=np.float32)

        output = gpu.activation_gelu(input_data)

        # GELU(0) should be 0
        assert abs(output[0]) < 1e-5
        # GELU should be positive for positive inputs
        assert output[1] > 0

    def test_activation_gcu(self, gpu):
        """Test GCU (Growing Cosine Unit) activation: x * cos(x)"""
        input_data = np.array([0.0, 1.0, -1.0, np.pi / 2, -np.pi / 2], dtype=np.float32)

        output = gpu.activation_gcu(input_data)

        # GCU(0) = 0 * cos(0) = 0 * 1 = 0
        assert abs(output[0]) < 1e-5

        # GCU(1) = 1 * cos(1) ≈ 0.5403
        expected_1 = 1.0 * np.cos(1.0)
        np.testing.assert_allclose(output[1], expected_1, rtol=1e-5)

        # GCU(-1) = -1 * cos(-1) ≈ -0.5403
        expected_neg1 = -1.0 * np.cos(-1.0)
        np.testing.assert_allclose(output[2], expected_neg1, rtol=1e-5)

        # GCU(π/2) = π/2 * cos(π/2) ≈ π/2 * 0 = 0
        expected_pi2 = (np.pi / 2) * np.cos(np.pi / 2)
        np.testing.assert_allclose(output[3], expected_pi2, rtol=1e-4, atol=1e-5)

        # GCU(-π/2) = -π/2 * cos(-π/2) ≈ -π/2 * 0 = 0
        expected_neg_pi2 = (-np.pi / 2) * np.cos(-np.pi / 2)
        np.testing.assert_allclose(output[4], expected_neg_pi2, rtol=1e-4, atol=1e-5)

    def test_activation_roswish(self, gpu):
        """Test RoSwish activation: (x + α) * sigmoid(β * x) - 0.5 * α"""
        input_data = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
        alpha = 1.0
        beta = 1.0

        output = gpu.activation_roswish(input_data, alpha=alpha, beta=beta)

        # Manual computation for verification
        sigmoid_bx = 1.0 / (1.0 + np.exp(-beta * input_data))
        expected = (input_data + alpha) * sigmoid_bx - 0.5 * alpha

        np.testing.assert_allclose(output, expected, rtol=1e-5)

        # Test with different parameters
        alpha2 = 0.5
        beta2 = 2.0
        output2 = gpu.activation_roswish(input_data, alpha=alpha2, beta=beta2)

        sigmoid_bx2 = 1.0 / (1.0 + np.exp(-beta2 * input_data))
        expected2 = (input_data + alpha2) * sigmoid_bx2 - 0.5 * alpha2

        np.testing.assert_allclose(output2, expected2, rtol=1e-5)

    def test_activation_swiglu(self, gpu):
        """Test SwiGLU activation: x1 * silu(x2) where input is [x1, x2]"""
        # SwiGLU expects input of shape (..., 2*hidden_dim)
        # It splits into x1 and x2, then computes x1 * silu(x2)
        hidden_dim = 4
        input_data = np.random.randn(8, 2 * hidden_dim).astype(np.float32)

        output = gpu.activation_swiglu(input_data)

        # Manual computation
        x1 = input_data[:, :hidden_dim]
        x2 = input_data[:, hidden_dim:]
        silu_x2 = x2 / (1.0 + np.exp(-x2))  # SiLU
        expected = x1 * silu_x2

        np.testing.assert_allclose(output, expected, rtol=1e-5)

        # Test 1D case
        input_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)  # 2*hidden_dim=4
        output_1d = gpu.activation_swiglu(input_1d)

        x1_1d = input_1d[:2]
        x2_1d = input_1d[2:]
        silu_x2_1d = x2_1d / (1.0 + np.exp(-x2_1d))
        expected_1d = x1_1d * silu_x2_1d

        np.testing.assert_allclose(output_1d, expected_1d, rtol=1e-5)

    def test_layernorm(self, gpu):
        """Test layer normalization"""
        input_data = np.random.randn(100).astype(np.float32)

        output = gpu.layernorm(input_data)

        # Output should be normalized (mean ~0, std ~1)
        assert abs(output.mean()) < 0.1
        assert abs(output.std() - 1.0) < 0.1


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestFAISSOperations:
    """Test FAISS operations on GPU"""

    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_faiss_compute_distances_l2(self, gpu):
        """Test L2 distance computation"""
        queries = np.random.randn(5, 128).astype(np.float32)
        database = np.random.randn(100, 128).astype(np.float32)

        distances = gpu.faiss_compute_distances(queries, database, distance_type="l2")

        assert distances.shape == (5, 100)
        assert np.all(distances >= 0)

        # Verify against CPU computation
        expected = np.sqrt(np.sum((queries[0:1] - database) ** 2, axis=1))
        np.testing.assert_allclose(distances[0], expected, rtol=1e-4)

    def test_faiss_compute_distances_cosine(self, gpu):
        """Test cosine distance computation"""
        queries = np.random.randn(3, 64).astype(np.float32)
        database = np.random.randn(50, 64).astype(np.float32)

        distances = gpu.faiss_compute_distances(queries, database, distance_type="cosine")

        assert distances.shape == (3, 50)
        assert np.all(distances >= 0)

    def test_faiss_topk(self, gpu):
        """Test top-k selection"""
        distances = np.random.randn(10, 1000).astype(np.float32)
        k = 20

        topk_indices, topk_distances = gpu.faiss_topk(distances, k)

        assert topk_indices.shape == (10, k)
        assert topk_distances.shape == (10, k)
        assert topk_indices.dtype == np.uint32

        # Verify top-k correctness
        for i in range(10):
            sorted_dists = np.sort(distances[i])
            np.testing.assert_allclose(np.sort(topk_distances[i]), sorted_dists[:k], rtol=1e-4)

    def test_faiss_end_to_end(self, gpu):
        """Test end-to-end FAISS pipeline"""
        np.random.seed(42)
        queries = np.random.randn(10, 256).astype(np.float32)
        database = np.random.randn(1000, 256).astype(np.float32)
        k = 10

        distances = gpu.faiss_compute_distances(queries, database, distance_type="cosine")
        topk_indices, topk_distances = gpu.faiss_topk(distances, k)

        assert topk_indices.shape == (10, k)

        # Verify indices match distances
        for i in range(10):
            for j in range(k):
                idx = topk_indices[i, j]
                dist = topk_distances[i, j]
                expected = distances[i, idx]
                assert abs(dist - expected) < 1e-4
