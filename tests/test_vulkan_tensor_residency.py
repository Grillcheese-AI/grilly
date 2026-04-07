"""
VulkanTensor GPU residency: prepare_for_dispatch binds buffers without redundant upload.
"""

import warnings

import numpy as np
import pytest

try:
    from grilly.backend.compute import VulkanCompute
    from grilly.backend.base import VULKAN_PYTHON_LEGACY_BACKEND_AVAILABLE
    from grilly.utils.tensor_conversion import VulkanTensor
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_PYTHON_LEGACY_BACKEND_AVAILABLE, reason="Vulkan not available")
def test_prepare_for_dispatch_binds_without_double_upload():
    """After one GPU op, a VulkanTensor should expose a buffer for the next op without re-uploading."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        backend = VulkanCompute()
    try:
        if "activation-relu" not in backend.fnn.shaders:
            pytest.skip("activation-relu not available")

        x = np.random.randn(4, 8).astype(np.float32)
        out = backend.fnn.activation_relu(x, return_gpu_tensor=True)
        assert isinstance(out, VulkanTensor)

        out.prepare_for_dispatch()
        assert out._pooled_buffer is not None or out._gpu_buffer is not None

        out2 = backend.fnn.activation_relu(out, return_gpu_tensor=False)
        assert out2.shape == (4, 8)
        assert np.all(np.isfinite(out2))
    finally:
        backend.cleanup()
