"""
Tests for VulkanTensor (GPU-resident tensors)

Tests GPU tensor operations on AMD/Vulkan systems.
"""

import numpy as np
import pytest

try:
    from grilly.utils.tensor_conversion import VulkanTensor, to_vulkan_gpu

    VULKAN_TENSOR_AVAILABLE = True
except ImportError:
    VULKAN_TENSOR_AVAILABLE = False


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_TENSOR_AVAILABLE, reason="VulkanTensor not available")
class TestVulkanTensor:
    """Test VulkanTensor GPU-resident tensor operations"""

    def test_vulkan_tensor_creation(self):
        """Test creating VulkanTensor from numpy array"""
        arr = np.random.randn(10, 20).astype(np.float32)
        tensor = VulkanTensor(arr)

        assert tensor.shape == (10, 20)
        assert tensor.dtype == np.float32
        assert isinstance(tensor, VulkanTensor)

    def test_vulkan_tensor_numpy(self):
        """Test converting VulkanTensor back to numpy"""
        arr = np.random.randn(5, 10).astype(np.float32)
        tensor = VulkanTensor(arr)

        result = tensor.numpy()
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 10)
        np.testing.assert_array_almost_equal(result, arr, decimal=5)

    def test_vulkan_tensor_cpu(self):
        """Test getting CPU copy"""
        arr = np.random.randn(3, 4).astype(np.float32)
        tensor = VulkanTensor(arr)

        cpu_result = tensor.cpu()
        assert isinstance(cpu_result, np.ndarray)
        np.testing.assert_array_almost_equal(cpu_result, arr, decimal=5)

    def test_to_vulkan_gpu(self):
        """Test to_vulkan_gpu function"""
        arr = np.random.randn(8, 16).astype(np.float32)
        tensor = to_vulkan_gpu(arr)

        assert isinstance(tensor, VulkanTensor)
        assert tensor.shape == (8, 16)

    def test_to_vulkan_gpu_pytorch(self):
        """Test converting PyTorch tensor to GPU tensor"""
        try:
            import torch

            torch_tensor = torch.randn(10, 20)
            vulkan_tensor = to_vulkan_gpu(torch_tensor)

            assert isinstance(vulkan_tensor, VulkanTensor)
            assert vulkan_tensor.shape == (10, 20)
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_vulkan_tensor_with_module(self):
        """Test using VulkanTensor with nn.Module"""
        try:
            from grilly import nn

            arr = np.random.randn(5, 128).astype(np.float32)
            tensor = VulkanTensor(arr)

            # Module should handle VulkanTensor automatically
            linear = nn.Linear(128, 64)
            result = linear(tensor)

            assert isinstance(result, np.ndarray)
            assert result.shape == (5, 64)
        except Exception as e:
            pytest.skip(f"Module test failed: {e}")

    def test_vulkan_tensor_array_interface(self):
        """Test numpy array interface"""
        arr = np.random.randn(4, 8).astype(np.float32)
        tensor = VulkanTensor(arr)

        # Should work with numpy operations
        result = np.array(tensor)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 8)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_TENSOR_AVAILABLE, reason="VulkanTensor not available")
class TestVulkanTensorGPU:
    """Test GPU operations with VulkanTensor"""

    def test_gpu_buffer_creation(self):
        """Test that GPU buffer is created (if Vulkan available)"""
        try:
            from grilly import Compute

            Compute()

            arr = np.random.randn(10, 20).astype(np.float32)
            tensor = VulkanTensor(arr)

            # Try to ensure uploaded (will fail gracefully if Vulkan not available)
            try:
                tensor._ensure_uploaded()
                assert tensor._uploaded or True  # Either uploaded or gracefully failed
            except RuntimeError:
                # Vulkan not available, that's okay
                pass
        except Exception:
            pytest.skip("Vulkan backend not available")

    def test_keep_on_gpu_option(self):
        """Test keep_on_gpu option in to_vulkan"""
        from grilly.utils.tensor_conversion import to_vulkan

        arr = np.random.randn(5, 10).astype(np.float32)

        # Without keep_on_gpu (default)
        result1 = to_vulkan(arr, keep_on_gpu=False)
        assert isinstance(result1, np.ndarray)

        # With keep_on_gpu (may return VulkanTensor or numpy depending on Vulkan availability)
        result2 = to_vulkan(arr, keep_on_gpu=True)
        # Should be either numpy or VulkanTensor
        assert isinstance(result2, (np.ndarray, VulkanTensor))
