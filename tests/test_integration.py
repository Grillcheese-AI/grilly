"""
Integration tests for Grilly SDK
"""

import numpy as np
import pytest

try:
    from grilly import Compute, SNNCompute
    from grilly.backend.base import VULKAN_AVAILABLE
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


class TestSNNIntegration:
    """Integration tests for SNN"""

    def test_snn_full_pipeline(self):
        """Test full SNN pipeline"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)

        # Process multiple embeddings
        embeddings = [np.random.randn(384).astype(np.float32) for _ in range(5)]

        results = []
        for emb in embeddings:
            result = snn.process(emb)
            results.append(result)
            snn.reset()

        assert len(results) == 5
        for result in results:
            assert "spike_activity" in result
            assert "firing_rate" in result

    def test_snn_temporal_dynamics(self):
        """Test SNN over multiple timesteps"""
        snn = SNNCompute(n_neurons=10, use_vulkan=False)
        snn.reset()

        # Apply constant input over time
        input_current = np.ones(10, dtype=np.float32) * 50.0
        spike_history = []

        for t in range(50):
            spikes = snn.forward(input_current)
            spike_history.append(spikes.sum())

        total_spikes = sum(spike_history)

        # Should have some activity
        assert total_spikes > 0


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestGPUIntegration:
    """Integration tests for GPU operations"""

    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_multiple_operations_sequence(self, gpu):
        """Test sequence of different operations"""
        # LIF step
        input_current = np.random.randn(100).astype(np.float32)
        membrane = np.zeros(100, dtype=np.float32)
        refractory = np.zeros(100, dtype=np.float32)

        mem_out, ref_out, spikes = gpu.lif_step(input_current, membrane, refractory)

        assert spikes.shape == (100,)

        # Activation
        output = gpu.activation_relu(mem_out)

        assert output.shape == (100,)

        # FAISS
        queries = np.random.randn(5, 128).astype(np.float32)
        database = np.random.randn(100, 128).astype(np.float32)

        distances = gpu.faiss_compute_distances(queries, database)
        topk_indices, topk_distances = gpu.faiss_topk(distances, k=10)

        assert topk_indices.shape == (5, 10)

    def test_large_batch_processing(self, gpu):
        """Test processing large batches"""
        n = 10000
        input_current = np.random.randn(n).astype(np.float32) * 0.3
        membrane = np.zeros(n, dtype=np.float32)
        refractory = np.zeros(n, dtype=np.float32)

        mem_out, ref_out, spikes = gpu.lif_step(input_current, membrane, refractory)

        spike_count = spikes.sum()

        assert spikes.shape == (n,)
        assert 0 <= spike_count <= n
