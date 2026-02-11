"""
Integration Tests for Vulkan-Only Operations (AMD Compatible)

Tests that verify core functionality works on AMD GPUs (Vulkan only, no CUDA)
"""

import numpy as np
import pytest

try:
    from grilly import functional, nn
    from grilly.backend.compute import VulkanCompute
    from grilly.utils.device_manager import get_device_manager

    GRILLY_AVAILABLE = True
except ImportError:
    GRILLY_AVAILABLE = False


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_AVAILABLE, reason="Grilly not available")
class TestVulkanCore:
    """Test core Vulkan functionality"""

    def test_vulkan_compute_initialization(self):
        """Test Vulkan compute backend can be initialized"""
        try:
            compute = VulkanCompute()
            assert compute is not None
        except RuntimeError as e:
            if "Vulkan" in str(e) or "not available" in str(e).lower():
                pytest.skip(f"Vulkan not available: {e}")
            raise

    def test_linear_layer_vulkan(self):
        """Test linear layer works with Vulkan"""
        try:
            linear = nn.Linear(128, 64)
            x = np.random.randn(10, 128).astype(np.float32)
            result = linear(x)
            assert result.shape == (10, 64)
            assert isinstance(result, np.ndarray)
        except RuntimeError as e:
            if "Vulkan" in str(e) or "not available" in str(e).lower():
                pytest.skip(f"Vulkan not available: {e}")
            raise

    def test_activation_vulkan(self):
        """Test activation functions work with Vulkan"""
        try:
            x = np.random.randn(10, 20).astype(np.float32)
            result = functional.relu(x)
            assert result.shape == x.shape
            assert np.all(result >= 0)  # ReLU should be non-negative
        except RuntimeError as e:
            if "Vulkan" in str(e) or "not available" in str(e).lower():
                pytest.skip(f"Vulkan not available: {e}")
            raise

    def test_sequential_model_vulkan(self):
        """Test sequential model works with Vulkan"""
        try:
            model = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10)
            )
            x = np.random.randn(5, 128).astype(np.float32)
            result = model(x)
            assert result.shape == (5, 10)
        except RuntimeError as e:
            if "Vulkan" in str(e) or "not available" in str(e).lower():
                pytest.skip(f"Vulkan not available: {e}")
            raise


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_AVAILABLE, reason="Grilly not available")
class TestDeviceManagerVulkan:
    """Test device manager with Vulkan only"""

    def test_device_manager_vulkan(self):
        """Test device manager works with Vulkan"""
        try:
            manager = get_device_manager()
            manager.set_device("vulkan")
            assert manager.get_device() == "vulkan"
        except RuntimeError as e:
            if "Vulkan" in str(e) or "not available" in str(e).lower():
                pytest.skip(f"Vulkan not available: {e}")
            raise

    def test_vulkan_backend_access(self):
        """Test accessing Vulkan backend"""
        try:
            from grilly.utils.device_manager import get_vulkan_backend

            backend = get_vulkan_backend()
            assert backend is not None
        except RuntimeError as e:
            if "Vulkan" in str(e) or "not available" in str(e).lower():
                pytest.skip(f"Vulkan not available: {e}")
            raise


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_AVAILABLE, reason="Grilly not available")
class TestPyTorchOpsVulkan:
    """Test PyTorch operations with Vulkan backend"""

    def test_pytorch_ops_vulkan(self):
        """Test PyTorch operations work with Vulkan"""
        try:
            from grilly.utils.pytorch_compat import tensor
            from grilly.utils.pytorch_ops import add, matmul, mul, relu

            a = tensor(np.array([1, 2, 3], dtype=np.float32))
            b = tensor(np.array([4, 5, 6], dtype=np.float32))
            c = add(a, b)
            assert c.shape == (3,)
        except ImportError:
            pytest.skip("PyTorch ops not available")
        except RuntimeError as e:
            if "Vulkan" in str(e) or "not available" in str(e).lower():
                pytest.skip(f"Vulkan not available: {e}")
            raise


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_AVAILABLE, reason="Grilly not available")
class TestEndToEndVulkan:
    """End-to-end tests using only Vulkan (AMD compatible)"""

    def test_full_pipeline_vulkan(self):
        """Test full pipeline using only Vulkan operations"""
        try:
            # Create model
            model = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            )

            # Create input
            x = np.random.randn(32, 256).astype(np.float32)

            # Forward pass
            output = model(x)

            # Verify output
            assert output.shape == (32, 10)
            assert isinstance(output, np.ndarray)
            assert output.dtype == np.float32
        except RuntimeError as e:
            if "Vulkan" in str(e) or "not available" in str(e).lower():
                pytest.skip(f"Vulkan not available: {e}")
            raise

    def test_memory_operations_vulkan(self):
        """Test memory operations with Vulkan"""
        try:
            from grilly.nn.memory import MemoryRead

            # Create memory
            memory_keys = np.random.randn(100, 128).astype(np.float32)
            memory_values = np.random.randn(100, 256).astype(np.float32)

            # Create queries
            queries = np.random.randn(10, 128).astype(np.float32)

            # Read from memory
            memory_read = MemoryRead(128, 256, num_memories=100)
            # MemoryRead.forward takes only queries, uses internal memory
            memory_read.memory_keys = memory_keys
            memory_read.memory_values = memory_values
            result = memory_read(queries)

            assert result.shape == (10, 256)
        except RuntimeError as e:
            if "Vulkan" in str(e) or "not available" in str(e).lower():
                pytest.skip(f"Vulkan not available: {e}")
            raise
