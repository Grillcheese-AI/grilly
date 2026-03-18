"""
Tests for C++ Tensor and Parameter classes.
Validates lazy GPU/CPU sync, numpy round-trips, reshape, and gradient tracking.
"""

import numpy as np
import pytest

try:
    from grilly_core import DType, Parameter, Tensor
    import grilly_core

    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    grilly_core = None

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ backend not available")


class TestTensorCreation:
    """Test Tensor construction and factory methods."""

    def test_from_numpy(self):
        """Tensor.from_numpy should preserve data and shape."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = Tensor.from_numpy(arr)
        assert t.shape == [2, 2]
        assert t.numel == 4
        assert t.ndim == 2
        np.testing.assert_array_equal(t.numpy(), arr)

    def test_from_numpy_1d(self):
        """Tensor.from_numpy works with 1D arrays."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = Tensor.from_numpy(arr)
        assert t.shape == [3]
        assert t.numel == 3
        np.testing.assert_array_equal(t.numpy(), arr)

    def test_from_numpy_3d(self):
        """Tensor.from_numpy works with 3D arrays."""
        arr = np.random.randn(2, 3, 4).astype(np.float32)
        t = Tensor.from_numpy(arr)
        assert t.shape == [2, 3, 4]
        assert t.numel == 24
        np.testing.assert_allclose(t.numpy(), arr)

    def test_zeros(self):
        """Tensor.zeros should create all-zero tensor."""
        t = Tensor.zeros([3, 4])
        assert t.shape == [3, 4]
        np.testing.assert_array_equal(t.numpy(), np.zeros((3, 4), dtype=np.float32))

    def test_empty(self):
        """Tensor.empty should create tensor with correct shape."""
        t = Tensor.empty([5, 6])
        assert t.shape == [5, 6]
        assert t.numel == 30
        # Empty doesn't guarantee values, just shape

    def test_shape_constructor(self):
        """Tensor(shape) should create zero-filled tensor."""
        t = Tensor([2, 3])
        assert t.shape == [2, 3]
        assert t.numel == 6

    def test_dtype_default_float32(self):
        """Default dtype should be Float32."""
        t = Tensor.zeros([2])
        assert t.dtype == DType.Float32

    def test_nbytes(self):
        """nbytes should be numel * sizeof(float32)."""
        t = Tensor.zeros([10])
        assert t.nbytes == 40  # 10 * 4 bytes


class TestTensorNumpyRoundTrip:
    """Test numpy conversion round-trips."""

    def test_roundtrip_preserves_data(self):
        """from_numpy -> numpy should preserve exact values."""
        arr = np.random.randn(10, 20).astype(np.float32)
        t = Tensor.from_numpy(arr)
        recovered = t.numpy()
        np.testing.assert_array_equal(arr, recovered)

    def test_roundtrip_large_tensor(self):
        """Round-trip works with large tensors."""
        arr = np.random.randn(100, 128).astype(np.float32)
        t = Tensor.from_numpy(arr)
        np.testing.assert_array_equal(t.numpy(), arr)

    def test_roundtrip_preserves_shape(self):
        """Shape should be preserved through round-trip."""
        for shape in [(5,), (3, 4), (2, 3, 4), (1, 1, 1, 1)]:
            arr = np.ones(shape, dtype=np.float32)
            t = Tensor.from_numpy(arr)
            assert t.numpy().shape == shape


class TestTensorReshape:
    """Test reshape and view operations."""

    def test_reshape_basic(self):
        """Reshape should change shape without changing data."""
        arr = np.arange(12, dtype=np.float32)
        t = Tensor.from_numpy(arr)
        t2 = t.reshape([3, 4])
        assert t2.shape == [3, 4]
        np.testing.assert_array_equal(t2.numpy().ravel(), arr)

    def test_reshape_with_neg1(self):
        """Reshape with -1 should infer the dimension."""
        t = Tensor.from_numpy(np.arange(12, dtype=np.float32))
        t2 = t.reshape([3, -1])
        assert t2.shape == [3, 4]

        t3 = t.reshape([-1, 6])
        assert t3.shape == [2, 6]

    def test_reshape_preserves_data(self):
        """Data should be the same after reshape."""
        arr = np.random.randn(2, 3, 4).astype(np.float32)
        t = Tensor.from_numpy(arr)
        t2 = t.reshape([6, 4])
        np.testing.assert_array_equal(t2.numpy().ravel(), arr.ravel())

    def test_reshape_wrong_size_raises(self):
        """Reshape to incompatible shape should raise."""
        t = Tensor.from_numpy(np.arange(12, dtype=np.float32))
        with pytest.raises(Exception):
            t.reshape([5, 5])

    def test_view_is_reshape(self):
        """view() should behave identically to reshape()."""
        t = Tensor.from_numpy(np.arange(12, dtype=np.float32))
        t2 = t.view([3, 4])
        assert t2.shape == [3, 4]


class TestTensorState:
    """Test CPU/GPU state tracking."""

    def test_starts_on_cpu(self):
        """Freshly created tensor should be on CPU."""
        t = Tensor.from_numpy(np.ones(5, dtype=np.float32))
        assert t.on_cpu is True
        assert t.on_gpu is False

    def test_valid(self):
        """Tensor with data should be valid."""
        t = Tensor.from_numpy(np.ones(5, dtype=np.float32))
        assert t.valid is True

    def test_requires_grad_default_false(self):
        """requires_grad should default to False for Tensor."""
        t = Tensor.from_numpy(np.ones(5, dtype=np.float32))
        assert t.requires_grad is False

    def test_set_requires_grad(self):
        """requires_grad should be settable."""
        t = Tensor.from_numpy(np.ones(5, dtype=np.float32))
        t.requires_grad = True
        assert t.requires_grad is True

    def test_repr(self):
        """repr should contain shape info."""
        t = Tensor.from_numpy(np.ones((2, 3), dtype=np.float32))
        r = repr(t)
        assert "Tensor" in r
        assert "2" in r
        assert "3" in r


class TestParameter:
    """Test Parameter class (extends Tensor with gradient tracking)."""

    def test_parameter_from_tensor(self):
        """Parameter wraps a Tensor with requires_grad=True."""
        t = Tensor.from_numpy(np.ones((3, 4), dtype=np.float32))
        p = Parameter(t)
        assert p.requires_grad is True
        assert p.shape == [3, 4]

    def test_parameter_from_shape(self):
        """Parameter can be created from shape."""
        p = Parameter([5, 6])
        assert p.shape == [5, 6]
        assert p.requires_grad is True

    def test_parameter_requires_grad_false(self):
        """Parameter can disable requires_grad."""
        t = Tensor.from_numpy(np.ones(5, dtype=np.float32))
        p = Parameter(t, requires_grad=False)
        assert p.requires_grad is False

    def test_parameter_zero_grad(self):
        """zero_grad should zero the gradient tensor."""
        p = Parameter(Tensor.from_numpy(np.ones(5, dtype=np.float32)))
        # Access grad to lazily create it
        g = p.grad()
        assert g.numel == 5
        # Grad should be zero
        np.testing.assert_array_equal(g.numpy(), np.zeros(5, dtype=np.float32))

    def test_parameter_has_grad(self):
        """has_grad should report whether gradient exists."""
        p = Parameter(Tensor.from_numpy(np.ones(5, dtype=np.float32)))
        # Before accessing grad
        assert p.has_grad() is False
        # After accessing grad
        p.grad()
        assert p.has_grad() is True

    def test_parameter_inherits_tensor_ops(self):
        """Parameter should support all Tensor operations."""
        p = Parameter(Tensor.from_numpy(np.arange(6, dtype=np.float32)))
        p2 = p.reshape([2, 3])
        assert p2.shape == [2, 3]
        np.testing.assert_array_equal(p2.numpy().ravel(), np.arange(6, dtype=np.float32))


class TestSplitBindings:
    """Test that the split binding registration functions are working."""

    def test_to_tensor_function_exists(self):
        """to_tensor() helper should be exposed in grilly_core."""
        assert hasattr(grilly_core, "to_tensor")

    def test_to_tensor_roundtrip(self):
        """to_tensor should produce a valid Tensor from numpy."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = grilly_core.to_tensor(arr)
        assert t.shape == [3]
        np.testing.assert_array_almost_equal(t.numpy(), arr)

    def test_module_class_exists(self):
        """Module class should be exposed for Python subclassing."""
        assert hasattr(grilly_core, "Module")

    def test_create_backend_function_exists(self):
        """create_backend() should be exposed."""
        assert hasattr(grilly_core, "create_backend")

    def test_compute_backend_class_exists(self):
        """ComputeBackend class should be exposed."""
        assert hasattr(grilly_core, "ComputeBackend")

    def test_device_class_exists(self):
        """Device (GrillyCoreContext) should be exposed."""
        assert hasattr(grilly_core, "Device")

    def test_opgraph_class_exists(self):
        """OpGraph should be exposed."""
        assert hasattr(grilly_core, "OpGraph")

    def test_snn_classes_exist(self):
        """SNN node classes should be exposed."""
        for cls_name in ["IFNode", "LIFNode", "ParametricLIFNode",
                         "BaseNode", "MemoryModule"]:
            assert hasattr(grilly_core, cls_name), f"Missing: {cls_name}"

    def test_optimizer_classes_exist(self):
        """Optimizer classes should be exposed."""
        for cls_name in ["Optimizer", "Adam", "AdamW", "SGD"]:
            assert hasattr(grilly_core, cls_name), f"Missing: {cls_name}"

    def test_dataloader_classes_exist(self):
        """DataLoader and Batch should be exposed."""
        assert hasattr(grilly_core, "DataLoader")
        assert hasattr(grilly_core, "Batch")

    def test_surrogate_classes_exist(self):
        """SurrogateFunction and SurrogateType should be exposed."""
        assert hasattr(grilly_core, "SurrogateFunction")
        assert hasattr(grilly_core, "SurrogateType")

    def test_container_classes_exist(self):
        """Container classes should be exposed."""
        for cls_name in ["MultiStepContainer", "SeqToANNContainer", "Flatten"]:
            assert hasattr(grilly_core, cls_name), f"Missing: {cls_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
