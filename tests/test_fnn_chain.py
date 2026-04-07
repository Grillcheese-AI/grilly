"""FnnChainRecorder: many dispatches, one submit (GPU)."""

import warnings

import numpy as np
import pytest

try:
    from grilly.backend.base import VULKAN_PYTHON_LEGACY_BACKEND_AVAILABLE
    from grilly.backend.compute import VulkanCompute
    from grilly.backend.fnn_chain import ChainBufferHandle
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_PYTHON_LEGACY_BACKEND_AVAILABLE, reason="Vulkan not available")
def test_fnn_chain_linear_relu_read():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        backend = VulkanCompute()
    try:
        if "fnn-linear" not in backend.fnn.shaders or "activation-relu" not in backend.fnn.shaders:
            pytest.skip("fnn-linear or activation-relu not available")

        rng = np.random.default_rng(0)
        x = rng.standard_normal((3, 8), dtype=np.float32)
        w1 = rng.standard_normal((16, 8), dtype=np.float32)
        b1 = rng.standard_normal((16,), dtype=np.float32)
        w2 = rng.standard_normal((4, 16), dtype=np.float32)

        with backend.record_commands(fnn_chain=True) as rec:
            h = rec.linear(x, w1, b1)
            assert isinstance(h, ChainBufferHandle)
            h = rec.relu(h)
            h = rec.linear(h, w2, None)
            out = rec.read(h)

        ref = np.maximum(0, x @ w1.T + b1) @ w2.T
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
    finally:
        backend.cleanup()


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_PYTHON_LEGACY_BACKEND_AVAILABLE, reason="Vulkan not available")
def test_fnn_chain_read_multiple_moe_fanout():
    """Parallel expert linears: one submit, multiple downloads (MoE pattern)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        backend = VulkanCompute()
    try:
        if "fnn-linear" not in backend.fnn.shaders:
            pytest.skip("fnn-linear not available")

        rng = np.random.default_rng(1)
        x = rng.standard_normal((2, 8), dtype=np.float32)
        w0 = rng.standard_normal((4, 8), dtype=np.float32)
        w1 = rng.standard_normal((4, 8), dtype=np.float32)
        w2 = rng.standard_normal((4, 8), dtype=np.float32)
        w3 = rng.standard_normal((4, 8), dtype=np.float32)

        with backend.record_commands(fnn_chain=True) as rec:
            h0 = rec.linear(x, w0, None)
            h1 = rec.linear(x, w1, None)
            h2 = rec.linear(x, w2, None)
            h3 = rec.linear(x, w3, None)
            results = rec.read_multiple([h0, h1, h2, h3])

        assert len(results) == 4
        refs = [x @ w0.T, x @ w1.T, x @ w2.T, x @ w3.T]
        for got, ref in zip(results, refs):
            np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)
    finally:
        backend.cleanup()
