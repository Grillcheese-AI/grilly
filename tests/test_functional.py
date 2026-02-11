"""Tests for functional API modules."""

import numpy as np
import pytest

try:
    from grilly import Compute
    from grilly.backend.base import VULKAN_AVAILABLE
    from grilly.functional import dropout, linear
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestFunctionalLinear:
    """Tests for functional.linear."""

    @pytest.fixture
    def backend(self):
        c = Compute()
        yield c
        if hasattr(c, "cleanup"):
            c.cleanup()

    def test_linear_basic(self, backend):
        """linear produces correct shape."""
        x = np.random.randn(4, 32).astype(np.float32)
        w = np.random.randn(64, 32).astype(np.float32)
        b = np.zeros(64, dtype=np.float32)
        out = linear(x, w, b)
        assert out.shape == (4, 64)

    def test_linear_no_bias(self, backend):
        """linear works without bias."""
        x = np.random.randn(2, 16).astype(np.float32)
        w = np.random.randn(8, 16).astype(np.float32)
        out = linear(x, w)
        assert out.shape == (2, 8)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestFunctionalDropout:
    """Tests for functional.dropout."""

    def test_dropout_training_false(self):
        """dropout with training=False returns input unchanged."""
        x = np.random.randn(4, 32).astype(np.float32)
        out = dropout(x, p=0.5, training=False)
        assert np.array_equal(out, x)

    def test_dropout_p_zero(self):
        """dropout with p=0 returns input unchanged."""
        x = np.random.randn(4, 32).astype(np.float32)
        out = dropout(x, p=0.0, training=True)
        assert np.array_equal(out, x)

    def test_dropout_training(self):
        """dropout with training=True produces different output."""
        x = np.ones((4, 32), dtype=np.float32)
        out = dropout(x, p=0.5, training=True)
        assert out.shape == x.shape
        # With p=0.5, some values should be zeroed
        assert out.dtype == np.float32


class TestFunctionalDropoutNoGPU:
    """Tests for functional.dropout that don't require GPU."""

    def test_dropout_training_false_no_gpu(self):
        """dropout training=False works without GPU."""
        x = np.random.randn(4, 32).astype(np.float32)
        out = dropout(x, p=0.5, training=False)
        assert np.array_equal(out, x)

    def test_dropout_p_zero_no_gpu(self):
        """dropout p=0 works without GPU."""
        x = np.random.randn(4, 32).astype(np.float32)
        out = dropout(x, p=0.0, training=True)
        assert np.array_equal(out, x)
