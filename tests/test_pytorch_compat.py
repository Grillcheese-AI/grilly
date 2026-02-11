"""
Tests for PyTorch Compatibility Layer

Tests PyTorch-like tensor operations using Vulkan backend
"""

import numpy as np
import pytest

try:
    from grilly.utils.pytorch_compat import (
        Tensor,
        arange,
        cat,
        from_numpy,
        ones,
        randn,
        stack,
        tensor,
        to_numpy,
        zeros,
    )

    PYTORCH_COMPAT_AVAILABLE = True
except ImportError:
    PYTORCH_COMPAT_AVAILABLE = False


@pytest.mark.skipif(not PYTORCH_COMPAT_AVAILABLE, reason="PyTorch compat not available")
class TestTensor:
    """Test Tensor class"""

    def test_tensor_creation_from_numpy(self):
        """Test creating tensor from numpy array"""
        arr = np.random.randn(10, 20).astype(np.float32)
        t = Tensor(arr)
        assert t.shape == (10, 20)
        assert t.dtype == np.float32
        np.testing.assert_array_equal(t.numpy(), arr)

    def test_tensor_creation_from_list(self):
        """Test creating tensor from list"""
        data = [[1, 2, 3], [4, 5, 6]]
        t = tensor(data)
        assert t.shape == (2, 3)

    def test_tensor_numpy(self):
        """Test converting tensor to numpy"""
        arr = np.random.randn(5, 5).astype(np.float32)
        t = Tensor(arr)
        result = t.numpy()
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_tensor_cpu(self):
        """Test moving tensor to CPU"""
        arr = np.random.randn(10, 10).astype(np.float32)
        t = Tensor(arr, device="vulkan")
        t_cpu = t.cpu()
        assert t_cpu.device == "cpu"
        np.testing.assert_array_equal(t_cpu.numpy(), arr)

    def test_tensor_add(self):
        """Test tensor addition"""
        a = tensor(np.array([1, 2, 3], dtype=np.float32))
        b = tensor(np.array([4, 5, 6], dtype=np.float32))
        c = a + b
        np.testing.assert_array_equal(c.numpy(), np.array([5, 7, 9], dtype=np.float32))

    def test_tensor_sub(self):
        """Test tensor subtraction"""
        a = tensor(np.array([5, 6, 7], dtype=np.float32))
        b = tensor(np.array([1, 2, 3], dtype=np.float32))
        c = a - b
        np.testing.assert_array_equal(c.numpy(), np.array([4, 4, 4], dtype=np.float32))

    def test_tensor_mul(self):
        """Test tensor multiplication"""
        a = tensor(np.array([2, 3, 4], dtype=np.float32))
        b = tensor(np.array([2, 2, 2], dtype=np.float32))
        c = a * b
        np.testing.assert_array_equal(c.numpy(), np.array([4, 6, 8], dtype=np.float32))

    def test_tensor_matmul(self):
        """Test tensor matrix multiplication"""
        a = tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b = tensor(np.array([[5, 6], [7, 8]], dtype=np.float32))
        c = a @ b
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_almost_equal(c.numpy(), expected)


@pytest.mark.skipif(not PYTORCH_COMPAT_AVAILABLE, reason="PyTorch compat not available")
class TestTensorFunctions:
    """Test tensor creation functions"""

    def test_zeros(self):
        """Test zeros function"""
        t = zeros((5, 10))
        assert t.shape == (5, 10)
        np.testing.assert_array_equal(t.numpy(), np.zeros((5, 10), dtype=np.float32))

    def test_ones(self):
        """Test ones function"""
        t = ones((3, 4))
        assert t.shape == (3, 4)
        np.testing.assert_array_equal(t.numpy(), np.ones((3, 4), dtype=np.float32))

    def test_randn(self):
        """Test randn function"""
        t = randn(10, 20)
        assert t.shape == (10, 20)
        assert t.dtype == np.float32

    def test_arange(self):
        """Test arange function"""
        t = arange(0, 10)
        expected = np.arange(0, 10, dtype=np.float32)
        np.testing.assert_array_equal(t.numpy(), expected)

    def test_cat(self):
        """Test cat function"""
        a = tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b = tensor(np.array([[5, 6], [7, 8]], dtype=np.float32))
        c = cat([a, b], dim=0)
        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        np.testing.assert_array_equal(c.numpy(), expected)

    def test_stack(self):
        """Test stack function"""
        a = tensor(np.array([1, 2, 3], dtype=np.float32))
        b = tensor(np.array([4, 5, 6], dtype=np.float32))
        c = stack([a, b], dim=0)
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        np.testing.assert_array_equal(c.numpy(), expected)

    def test_from_numpy(self):
        """Test from_numpy function"""
        arr = np.random.randn(10, 20).astype(np.float32)
        t = from_numpy(arr)
        assert isinstance(t, Tensor)
        np.testing.assert_array_equal(t.numpy(), arr)

    def test_to_numpy(self):
        """Test to_numpy function"""
        arr = np.random.randn(5, 5).astype(np.float32)
        t = tensor(arr)
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)
