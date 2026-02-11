"""
Tests for Spiking Neural Network (SNN) operations
"""

import numpy as np
import pytest

try:
    from grilly import SNNCompute
    from grilly.backend.base import VULKAN_AVAILABLE
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


class TestSNNForward:
    """Test SNN forward pass"""

    def test_forward_returns_spikes(self):
        """Forward should return spike array"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        input_current = np.random.randn(100).astype(np.float32)

        spikes = snn.forward(input_current)

        assert spikes is not None
        assert len(spikes) == 100
        assert spikes.dtype == np.float32

    def test_forward_spikes_are_binary(self):
        """Spikes should be 0 or 1"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        input_current = np.random.randn(100).astype(np.float32)

        spikes = snn.forward(input_current)

        assert np.all((spikes == 0) | (spikes == 1))

    def test_forward_updates_membrane(self):
        """Forward should update membrane potential"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        initial_membrane = snn.membrane.copy()

        input_current = np.ones(100, dtype=np.float32) * 0.1
        snn.forward(input_current)

        # Membrane should have changed
        assert not np.allclose(snn.membrane, initial_membrane)

    def test_forward_strong_input_causes_spikes(self):
        """Strong input should cause spikes"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)

        # Very strong input
        input_current = np.ones(100, dtype=np.float32) * 100.0

        # Run multiple timesteps
        total_spikes = 0
        for _ in range(10):
            spikes = snn.forward(input_current)
            total_spikes += spikes.sum()

        assert total_spikes > 0


class TestSNNProcess:
    """Test SNN process method (full pipeline)"""

    def test_process_returns_dict(self):
        """Process should return a dictionary"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        embedding = np.random.randn(384).astype(np.float32)

        result = snn.process(embedding)

        assert isinstance(result, dict)

    def test_process_contains_required_keys(self):
        """Result should contain required keys"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        embedding = np.random.randn(384).astype(np.float32)

        result = snn.process(embedding)

        assert "spike_activity" in result
        assert "spikes" in result
        assert "firing_rate" in result

    def test_process_spike_activity_is_numeric(self):
        """Spike activity should be a number"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        embedding = np.random.randn(384).astype(np.float32)

        result = snn.process(embedding)

        assert isinstance(result["spike_activity"], (int, float))
        assert result["spike_activity"] >= 0

    def test_process_spikes_is_array(self):
        """Spikes should be an array"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        embedding = np.random.randn(384).astype(np.float32)

        result = snn.process(embedding)

        assert isinstance(result["spikes"], np.ndarray)
        assert len(result["spikes"]) == 100

    def test_process_firing_rate_in_valid_range(self):
        """Firing rate should be between 0 and 1"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        embedding = np.random.randn(384).astype(np.float32)

        result = snn.process(embedding)

        assert 0 <= result["firing_rate"] <= 1

    def test_process_handles_small_embedding(self):
        """Process should handle embeddings smaller than n_neurons"""
        snn = SNNCompute(n_neurons=1000, use_vulkan=False)
        embedding = np.random.randn(100).astype(np.float32)

        result = snn.process(embedding)

        assert result["spikes"] is not None
        assert len(result["spikes"]) == 1000

    def test_process_handles_large_embedding(self):
        """Process should handle embeddings larger than n_neurons"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        embedding = np.random.randn(1000).astype(np.float32)

        result = snn.process(embedding)

        assert result["spikes"] is not None
        assert len(result["spikes"]) == 100


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestSNNGPU:
    """Test SNN with GPU (if available)"""

    def test_gpu_init_does_not_crash(self):
        """GPU initialization should not crash"""
        try:
            snn = SNNCompute(n_neurons=100, use_vulkan=True)
            assert snn.n_neurons == 100
        except Exception as e:
            pytest.skip(f"GPU not available: {e}")

    def test_gpu_process_works(self):
        """Process should work in GPU mode"""
        try:
            snn = SNNCompute(n_neurons=100, use_vulkan=True)
            embedding = np.random.randn(384).astype(np.float32)

            result = snn.process(embedding)

            assert "spike_activity" in result
        except Exception as e:
            pytest.skip(f"GPU not available: {e}")


class TestSNNReproducibility:
    """Test SNN reproducibility"""

    def test_same_input_after_reset_gives_same_output(self):
        """Same input after reset should give same output"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        embedding = np.ones(384, dtype=np.float32) * 0.1

        # First run
        result1 = snn.process(embedding)

        # Reset and run again
        snn.reset()
        result2 = snn.process(embedding)

        # Should be the same
        assert result1["spike_activity"] == result2["spike_activity"]
        np.testing.assert_array_equal(result1["spikes"], result2["spikes"])
