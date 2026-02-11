"""
Tests for learning operations
"""

import numpy as np
import pytest

try:
    from grilly import Compute, Learning
    from grilly.backend.base import VULKAN_AVAILABLE
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestLearningOperations:
    """Test learning operations on GPU"""

    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_whitening_transform(self, gpu):
        """Test whitening transform"""
        # Create correlated data
        n_samples = 100
        n_features = 64
        data = np.random.randn(n_samples, n_features).astype(np.float32)

        # Add correlation
        data = data @ np.random.randn(n_features, n_features).astype(np.float32)

        # Initialize running statistics
        running_mean = np.zeros(n_features, dtype=np.float32)
        running_var = np.ones(n_features, dtype=np.float32)

        whitened, new_mean, new_var = gpu.whitening_transform(data, running_mean, running_var)

        assert whitened.shape == data.shape
        assert np.all(np.isfinite(whitened))
        assert new_mean.shape == (n_features,)
        assert new_var.shape == (n_features,)

    def test_nlms_predict(self, gpu):
        """Test NLMS prediction"""
        n_features = 32
        n_filters = 4

        weights = np.random.randn(n_filters, n_features).astype(np.float32)
        input_signal = np.random.randn(n_features).astype(np.float32)

        predictions = gpu.nlms_predict(weights, input_signal)

        assert predictions.shape == (n_filters,)
        assert np.all(np.isfinite(predictions))

    def test_nlms_update(self, gpu):
        """Test NLMS weight update"""
        n_features = 32

        # NLMS update expects: features, prediction, target, weights, bias, learning_rate, ...
        features = np.random.randn(n_features).astype(np.float32)
        weights = np.random.randn(n_features).astype(np.float32)
        bias = 0.0
        prediction = float(np.dot(features, weights) + bias)
        target = prediction + 0.1  # Small error

        updated_weights, updated_bias, updated_lr, error = gpu.nlms_update(
            features, prediction, target, weights, bias, learning_rate=0.5
        )

        assert updated_weights.shape == weights.shape
        assert np.all(np.isfinite(updated_weights))
        assert np.isfinite(updated_bias)
        assert np.isfinite(error)
