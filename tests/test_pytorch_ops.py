"""
Tests for PyTorch Operations

Tests common PyTorch operations using Vulkan backend
"""

import numpy as np
import pytest

try:
    from grilly.utils.pytorch_compat import tensor, to_numpy
    from grilly.utils.pytorch_ops import (
        add,
        avg_pool2d,
        batch_norm,
        bmm,
        conv2d,
        cross_entropy_loss,
        dropout,
        flatten,
        gelu,
        layer_norm,
        matmul,
        max_pool2d,
        mse_loss,
        mul,
        relu,
        reshape,
        sigmoid,
        softmax,
        squeeze,
        tanh,
        transpose,
        unsqueeze,
    )

    PYTORCH_OPS_AVAILABLE = True
except ImportError:
    PYTORCH_OPS_AVAILABLE = False


@pytest.mark.gpu
@pytest.mark.skipif(not PYTORCH_OPS_AVAILABLE, reason="PyTorch ops not available")
class TestBasicOperations:
    """Test basic operations"""

    def test_add(self):
        """Test add operation"""
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        result = add(a, b)
        expected = np.array([5, 7, 9], dtype=np.float32)
        np.testing.assert_array_equal(to_numpy(result), expected)

    def test_mul(self):
        """Test multiply operation"""
        a = np.array([2, 3, 4], dtype=np.float32)
        b = np.array([2, 2, 2], dtype=np.float32)
        result = mul(a, b)
        expected = np.array([4, 6, 8], dtype=np.float32)
        np.testing.assert_array_equal(to_numpy(result), expected)

    def test_matmul(self):
        """Test matrix multiplication"""
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        result = matmul(a, b)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_almost_equal(to_numpy(result), expected)

    def test_bmm(self):
        """Test batch matrix multiplication"""
        a = np.random.randn(2, 3, 4).astype(np.float32)
        b = np.random.randn(2, 4, 5).astype(np.float32)
        result = bmm(a, b)
        assert result.shape == (2, 3, 5)


@pytest.mark.gpu
@pytest.mark.skipif(not PYTORCH_OPS_AVAILABLE, reason="PyTorch ops not available")
class TestActivations:
    """Test activation functions"""

    def test_relu(self):
        """Test ReLU activation"""
        x = np.array([-1, 0, 1, 2], dtype=np.float32)
        result = relu(x)
        expected = np.array([0, 0, 1, 2], dtype=np.float32)
        np.testing.assert_array_equal(to_numpy(result), expected)

    def test_gelu(self):
        """Test GELU activation"""
        x = np.array([0.0, 1.0], dtype=np.float32)
        result = gelu(x)
        assert result.shape == x.shape
        # GELU(0) should be approximately 0
        assert abs(to_numpy(result)[0]) < 0.01

    def test_softmax(self):
        """Test softmax activation"""
        x = np.array([[1, 2, 3]], dtype=np.float32)
        result = softmax(x, dim=-1)
        result_np = to_numpy(result)
        # Should sum to 1
        assert abs(np.sum(result_np) - 1.0) < 0.01
        # Should be positive
        assert np.all(result_np > 0)

    def test_sigmoid(self):
        """Test sigmoid activation"""
        x = np.array([0.0], dtype=np.float32)
        result = sigmoid(x)
        # sigmoid(0) = 0.5
        assert abs(to_numpy(result)[0] - 0.5) < 0.01

    def test_tanh(self):
        """Test tanh activation"""
        x = np.array([0.0], dtype=np.float32)
        result = tanh(x)
        # tanh(0) = 0
        assert abs(to_numpy(result)[0]) < 0.01


@pytest.mark.gpu
@pytest.mark.skipif(not PYTORCH_OPS_AVAILABLE, reason="PyTorch ops not available")
class TestNormalization:
    """Test normalization operations"""

    def test_layer_norm(self):
        """Test layer normalization"""
        x = np.random.randn(2, 10).astype(np.float32)
        weight = np.ones(10, dtype=np.float32)
        bias = np.zeros(10, dtype=np.float32)
        result = layer_norm(x, (10,), weight, bias)
        result_np = to_numpy(result)
        # Mean should be approximately 0
        assert np.allclose(np.mean(result_np, axis=-1), 0, atol=0.1)

    def test_batch_norm(self):
        """Test batch normalization"""
        x = np.random.randn(2, 3, 4, 4).astype(np.float32)
        running_mean = np.zeros(3, dtype=np.float32)
        running_var = np.ones(3, dtype=np.float32)
        weight = np.ones(3, dtype=np.float32)
        bias = np.zeros(3, dtype=np.float32)
        result = batch_norm(x, running_mean, running_var, weight, bias, training=True)
        assert result.shape == x.shape


@pytest.mark.skipif(not PYTORCH_OPS_AVAILABLE, reason="PyTorch ops not available")
class TestPooling:
    """Test pooling operations"""

    def test_max_pool2d(self):
        """Test 2D max pooling"""
        x = np.random.randn(1, 1, 4, 4).astype(np.float32)
        result = max_pool2d(x, kernel_size=2, stride=2)
        assert result.shape == (1, 1, 2, 2)

    def test_avg_pool2d(self):
        """Test 2D average pooling"""
        x = np.random.randn(1, 1, 4, 4).astype(np.float32)
        result = avg_pool2d(x, kernel_size=2, stride=2)
        assert result.shape == (1, 1, 2, 2)


@pytest.mark.skipif(not PYTORCH_OPS_AVAILABLE, reason="PyTorch ops not available")
class TestLossFunctions:
    """Test loss functions"""

    def test_mse_loss(self):
        """Test MSE loss"""
        pred = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        target = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        loss = mse_loss(pred, target)
        # Should be 0 for identical inputs
        assert abs(to_numpy(loss)) < 0.01

    def test_cross_entropy_loss(self):
        """Test cross entropy loss"""
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        target = np.array([2], dtype=np.int32)  # Class 2
        loss = cross_entropy_loss(logits, target)
        # Loss should be positive
        assert to_numpy(loss) > 0


@pytest.mark.skipif(not PYTORCH_OPS_AVAILABLE, reason="PyTorch ops not available")
class TestUtilityFunctions:
    """Test utility functions"""

    def test_flatten(self):
        """Test flatten function"""
        x = np.random.randn(2, 3, 4).astype(np.float32)
        result = flatten(x, start_dim=1, end_dim=2)
        assert result.shape == (2, 12)

    def test_reshape(self):
        """Test reshape function"""
        x = np.random.randn(2, 3, 4).astype(np.float32)
        result = reshape(x, (2, 12))
        assert result.shape == (2, 12)

    def test_transpose(self):
        """Test transpose function"""
        x = np.random.randn(2, 3).astype(np.float32)
        result = transpose(x, 0, 1)
        assert result.shape == (3, 2)

    def test_unsqueeze(self):
        """Test unsqueeze function"""
        x = np.random.randn(2, 3).astype(np.float32)
        result = unsqueeze(x, 0)
        assert result.shape == (1, 2, 3)

    def test_squeeze(self):
        """Test squeeze function"""
        x = np.random.randn(1, 2, 3).astype(np.float32)
        result = squeeze(x, 0)
        assert result.shape == (2, 3)
