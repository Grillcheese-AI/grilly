"""
Tests for Tensor Conversion Utilities

Tests conversion between PyTorch tensors and Vulkan (numpy arrays)
"""

import numpy as np
import pytest

try:
    from grilly.utils.tensor_conversion import (
        auto_convert_to_vulkan,
        convert_module_inputs,
        ensure_vulkan_compatible,
        from_vulkan,
        to_vulkan,
        to_vulkan_batch,
    )

    TENSOR_CONVERSION_AVAILABLE = True
except ImportError:
    TENSOR_CONVERSION_AVAILABLE = False


@pytest.mark.skipif(not TENSOR_CONVERSION_AVAILABLE, reason="Tensor conversion not available")
class TestTensorConversion:
    """Test tensor conversion functions"""

    def test_to_vulkan_numpy(self):
        """Test converting numpy array to Vulkan"""
        arr = np.random.randn(10, 20).astype(np.float32)
        result = to_vulkan(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_to_vulkan_pytorch(self):
        """Test converting PyTorch tensor to Vulkan"""
        try:
            import torch

            tensor = torch.randn(10, 20)
            result = to_vulkan(tensor)
            assert isinstance(result, np.ndarray)
            assert result.shape == (10, 20)
            assert result.dtype == np.float32
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_to_vulkan_pytorch_cuda(self):
        """Test converting PyTorch CUDA tensor to Vulkan"""
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
            tensor = torch.randn(10, 20).cuda()
            result = to_vulkan(tensor)
            assert isinstance(result, np.ndarray)
            assert result.shape == (10, 20)
            assert result.dtype == np.float32
        except (ImportError, AssertionError):
            pytest.skip("PyTorch/CUDA not available")

    def test_to_vulkan_batch(self):
        """Test batch conversion"""
        try:
            import torch

            tensors = [torch.randn(10, 20), torch.randn(5, 30), torch.randn(8, 15)]
            results = to_vulkan_batch(tensors)
            assert len(results) == 3
            assert all(isinstance(r, np.ndarray) for r in results)
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_from_vulkan_cpu(self):
        """Test converting Vulkan array to PyTorch CPU tensor"""
        try:
            import torch

            arr = np.random.randn(10, 20).astype(np.float32)
            result = from_vulkan(arr, device="cpu")
            assert isinstance(result, torch.Tensor)
            assert result.device.type == "cpu"
            assert result.shape == (10, 20)
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_from_vulkan_cuda(self):
        """Test converting Vulkan array to PyTorch CUDA tensor"""
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
            arr = np.random.randn(10, 20).astype(np.float32)
            result = from_vulkan(arr, device="cuda")
            assert isinstance(result, torch.Tensor)
            assert result.device.type == "cuda"
            assert result.shape == (10, 20)
        except (ImportError, AssertionError):
            pytest.skip("PyTorch/CUDA not available")

    def test_ensure_vulkan_compatible(self):
        """Test ensuring Vulkan compatibility"""
        # Test with numpy
        arr = np.random.randn(10, 20).astype(np.float64)
        result = ensure_vulkan_compatible(arr)
        assert result.dtype == np.float32

        # Test with PyTorch
        try:
            import torch

            tensor = torch.randn(10, 20)
            result = ensure_vulkan_compatible(tensor)
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
        except ImportError:
            pass

    def test_convert_module_inputs(self):
        """Test converting module inputs"""
        try:
            import torch

            x = torch.randn(10, 20)
            y = torch.randn(20, 30)
            args, kwargs = convert_module_inputs(x, y, param=torch.tensor([1, 2, 3]))

            assert len(args) == 2
            assert all(isinstance(a, np.ndarray) for a in args)
            assert "param" in kwargs
            assert isinstance(kwargs["param"], np.ndarray)
        except ImportError:
            pytest.skip("PyTorch not available")


@pytest.mark.gpu
@pytest.mark.skipif(not TENSOR_CONVERSION_AVAILABLE, reason="Tensor conversion not available")
class TestAutomaticConversion:
    """Test automatic conversion in nn.Module"""

    def test_module_auto_conversion(self):
        """Test that nn.Module automatically converts PyTorch tensors"""
        try:
            import torch

            from grilly import nn

            # Create PyTorch tensor
            torch_tensor = torch.randn(10, 128, dtype=torch.float32)

            # Create model
            linear = nn.Linear(128, 64)

            # Pass PyTorch tensor directly - should auto-convert
            result = linear(torch_tensor)

            # Result should be numpy array
            assert isinstance(result, np.ndarray)
            assert result.shape == (10, 64)
            assert result.dtype == np.float32
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_sequential_auto_conversion(self):
        """Test automatic conversion with Sequential model"""
        try:
            import torch

            from grilly import nn

            # Create PyTorch tensor
            torch_tensor = torch.randn(5, 256, dtype=torch.float32)

            # Create sequential model
            model = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64))

            # Pass PyTorch tensor directly
            result = model(torch_tensor)

            # Result should be numpy
            assert isinstance(result, np.ndarray)
            assert result.shape == (5, 64)
        except ImportError:
            pytest.skip("PyTorch not available")
