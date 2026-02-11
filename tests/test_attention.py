"""
Tests for attention operations
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
class TestAttentionOperations:
    """Test attention operations on GPU"""

    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_attention_scores(self, gpu):
        """Test attention score computation"""
        batch_size = 2
        seq_len = 10
        num_heads = 1
        head_dim = 64

        # Shape: (batch, seq_len, num_heads * head_dim)
        q = np.random.randn(batch_size, seq_len, num_heads * head_dim).astype(np.float32)
        k = np.random.randn(batch_size, seq_len, num_heads * head_dim).astype(np.float32)

        scores = gpu.attention_scores(q, k, num_heads, head_dim)

        # Output shape: (batch, num_heads, seq_len, seq_len)
        assert scores.shape == (batch_size, num_heads, seq_len, seq_len)
        assert np.all(np.isfinite(scores))

    def test_attention_output(self, gpu):
        """Test attention output computation"""
        batch_size = 2
        seq_len = 10
        num_heads = 1
        head_dim = 64

        # Attention weights: (batch, num_heads, seq_len, seq_len)
        weights = np.random.rand(batch_size, num_heads, seq_len, seq_len).astype(np.float32)
        # Normalize weights
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-8)
        # Values: (batch, seq_len, num_heads * head_dim)
        values = np.random.randn(batch_size, seq_len, num_heads * head_dim).astype(np.float32)

        output = gpu.attention_output(weights, values, num_heads, head_dim)

        # Output shape: (batch, seq_len, num_heads, head_dim)
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
        assert np.all(np.isfinite(output))
