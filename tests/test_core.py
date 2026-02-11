"""
Tests for core Grilly functionality and initialization
"""

import numpy as np
import pytest

try:
    import grilly
    from grilly.backend.base import VULKAN_AVAILABLE

    GRILLY_AVAILABLE = True
except ImportError:
    GRILLY_AVAILABLE = False
    VULKAN_AVAILABLE = False


class TestGrillyImports:
    """Test that grilly can be imported correctly"""

    def test_import_grilly(self):
        """Test basic grilly import"""
        assert GRILLY_AVAILABLE, "grilly package not available"
        assert hasattr(grilly, "VULKAN_AVAILABLE")

    def test_import_compute(self):
        """Test Compute class import"""
        from grilly import Compute, VulkanCompute

        assert Compute is VulkanCompute

    def test_import_snn_compute(self):
        """Test SNNCompute import"""
        from grilly import SNNCompute

        assert SNNCompute is not None

    def test_import_learning(self):
        """Test Learning class import"""
        from grilly import Learning, VulkanLearning

        assert Learning is VulkanLearning

    def test_vulkan_available_flag(self):
        """Test VULKAN_AVAILABLE flag"""
        from grilly.backend.base import VULKAN_AVAILABLE

        assert isinstance(VULKAN_AVAILABLE, bool)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestComputeInitialization:
    """Test Compute backend initialization"""

    def test_compute_init(self):
        """Test Compute initialization"""
        from grilly import Compute

        backend = Compute()

        assert backend.device is not None
        assert backend.queue is not None
        assert len(backend.shaders) > 0

    def test_compute_shaders_loaded(self):
        """Test that shaders are loaded"""
        from grilly import Compute

        backend = Compute()

        assert isinstance(backend.shaders, dict)
        assert len(backend.shaders) > 0

    def test_compute_cleanup(self):
        """Test Compute cleanup"""
        from grilly import Compute

        backend = Compute()
        backend.cleanup()

        # After cleanup, device should be destroyed
        # (exact behavior depends on implementation)


class TestSNNComputeInitialization:
    """Test SNNCompute initialization"""

    def test_snn_init_default(self):
        """Test SNNCompute with default parameters"""
        from grilly import SNNCompute

        snn = SNNCompute(use_vulkan=False)

        assert snn.n_neurons > 0
        assert snn.membrane is not None
        assert len(snn.membrane) == snn.n_neurons

    def test_snn_init_custom_neurons(self):
        """Test SNNCompute with custom neuron count"""
        from grilly import SNNCompute

        snn = SNNCompute(n_neurons=500, use_vulkan=False)

        assert snn.n_neurons == 500
        assert len(snn.membrane) == 500

    def test_snn_init_membrane_zero(self):
        """Test that membrane starts at zero"""
        from grilly import SNNCompute

        snn = SNNCompute(n_neurons=100, use_vulkan=False)

        assert np.all(snn.membrane == 0)
        assert np.all(snn.refractory == 0)

    def test_snn_reset(self):
        """Test SNNCompute reset"""
        from grilly import SNNCompute

        snn = SNNCompute(n_neurons=100, use_vulkan=False)

        # Modify state
        snn.membrane = np.ones(100, dtype=np.float32)
        snn.refractory = np.ones(100, dtype=np.float32)

        # Reset
        snn.reset()

        assert np.all(snn.membrane == 0)
        assert np.all(snn.refractory == 0)
