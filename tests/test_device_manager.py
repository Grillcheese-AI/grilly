"""
Tests for Device Manager

Tests multi-backend device management (Vulkan, CUDA, CPU)
"""

import numpy as np
import pytest

from grilly.utils.device_manager import (
    DeviceManager,
    get_cuda_backend,
    get_device_manager,
    get_torch,
    get_vulkan_backend,
)


class TestDeviceManager:
    """Test DeviceManager class"""

    def test_device_manager_initialization(self):
        """Test device manager can be initialized"""
        manager = DeviceManager()
        assert manager is not None
        assert manager._current == "vulkan"

    def test_set_device_vulkan(self):
        """Test setting device to Vulkan"""
        manager = DeviceManager()
        manager.set_device("vulkan")
        assert manager.get_device() == "vulkan"

    def test_set_device_cpu(self):
        """Test setting device to CPU"""
        manager = DeviceManager()
        manager.set_device("cpu")
        assert manager.get_device() == "cpu"

    def test_set_device_cuda(self):
        """Test setting device to CUDA (if available)"""
        manager = DeviceManager()
        try:
            manager.set_device("cuda")
            assert manager.get_device() == "cuda"
        except RuntimeError:
            # CUDA not available, skip
            pytest.skip("CUDA not available")

    def test_to_vulkan_numpy(self):
        """Test converting numpy array to Vulkan format"""
        manager = DeviceManager()
        arr = np.random.randn(10, 20).astype(np.float32)
        result = manager.to_vulkan(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_to_vulkan_pytorch(self):
        """Test converting PyTorch tensor to numpy"""
        manager = DeviceManager()
        try:
            import torch

            tensor = torch.randn(10, 20)
            result = manager.to_vulkan(tensor)
            assert isinstance(result, np.ndarray)
            assert result.shape == (10, 20)
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_to_cuda(self):
        """Test converting numpy to CUDA tensor"""
        manager = DeviceManager()
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
            arr = np.random.randn(10, 20).astype(np.float32)
            result = manager.to_cuda(arr)
            assert isinstance(result, torch.Tensor)
            assert result.device.type == "cuda"
        except (ImportError, RuntimeError, AssertionError) as e:
            if "CUDA" in str(e) or "not compiled" in str(e) or "not available" in str(e).lower():
                pytest.skip(f"CUDA/PyTorch not available: {e}")
            raise

    def test_device_count_vulkan(self):
        """Test getting Vulkan device count"""
        manager = DeviceManager()
        count = manager.device_count("vulkan")
        assert count >= 1

    def test_device_count_cpu(self):
        """Test getting CPU device count"""
        manager = DeviceManager()
        count = manager.device_count("cpu")
        assert count >= 1

    def test_synchronize_vulkan(self):
        """Test Vulkan synchronization (should not raise)"""
        manager = DeviceManager()
        manager.synchronize("vulkan")  # Should not raise

    def test_synchronize_cpu(self):
        """Test CPU synchronization (should not raise)"""
        manager = DeviceManager()
        manager.synchronize("cpu")  # Should not raise


class TestDeviceManagerGlobal:
    """Test global device manager functions"""

    def test_get_device_manager(self):
        """Test getting global device manager"""
        manager = get_device_manager()
        assert isinstance(manager, DeviceManager)

    def test_get_vulkan_backend(self):
        """Test getting Vulkan backend"""
        try:
            backend = get_vulkan_backend()
            assert backend is not None
        except RuntimeError:
            pytest.skip("Vulkan not available")

    def test_get_cuda_backend(self):
        """Test getting CUDA backend (if available)"""
        try:
            backend = get_cuda_backend()
            assert backend is not None
        except (ImportError, RuntimeError):
            pytest.skip("CUDA/PyTorch not available")

    def test_get_torch(self):
        """Test getting PyTorch module (if available)"""
        try:
            torch = get_torch()
            assert torch is not None
        except ImportError:
            pytest.skip("PyTorch not available")
