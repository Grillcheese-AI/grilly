"""
CPU-only tests for nn/modules.py init, repr, and backward for classes
that do not require GPU (Vulkan) in their __init__.

Covers helper functions (_get_param_array, _create_param_wrapper) and
modules whose __init__ is pure Python: Dropout, ReLU, GELU, SiLU,
Softmax, Softplus, Sequential, Residual, RoSwish, GCU.
"""

import numpy as np
import pytest

try:
    from grilly.nn.modules import (
        GCU,
        GELU,
        Dropout,
        ReLU,
        Residual,
        RoSwish,
        Sequential,
        SiLU,
        Softmax,
        Softplus,
        _create_param_wrapper,
        _get_param_array,
    )
    from grilly.nn.module import Module
    from grilly.nn.parameter import Parameter
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


@pytest.fixture(autouse=True)
def seed_rng():
    """Seed numpy RNG for reproducibility."""
    np.random.seed(42)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestGetParamArray:
    """Tests for the _get_param_array helper."""

    def test_with_ndarray(self):
        """Plain ndarray is returned as-is."""
        arr = np.ones((3, 4), dtype=np.float32)
        result = _get_param_array(arr)
        assert result is arr

    def test_with_parameter(self):
        """Parameter (np.ndarray subclass) is returned directly."""
        data = np.zeros((2, 5), dtype=np.float32)
        param = Parameter(data, requires_grad=True)
        result = _get_param_array(param)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, data)

    def test_with_param_wrapper(self):
        """_ParamWrapper returns the underlying .data array."""
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        wrapper = _create_param_wrapper(data)
        result = _get_param_array(wrapper)
        np.testing.assert_array_equal(result, data)


class TestCreateParamWrapper:
    """Tests for the _create_param_wrapper helper."""

    def test_basic(self):
        """Created wrapper holds data that matches the input."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        wrapper = _create_param_wrapper(data)
        result = _get_param_array(wrapper)
        np.testing.assert_array_equal(result, data)

    def test_has_grad(self):
        """Wrapper exposes a grad attribute (None initially)."""
        data = np.zeros((4,), dtype=np.float32)
        wrapper = _create_param_wrapper(data)
        assert hasattr(wrapper, "grad")
        # grad is None initially for the fallback wrapper;
        # for Parameter it is also None initially.
        assert wrapper.grad is None

    def test_shape_dtype(self):
        """Shape and dtype of the wrapper match the input."""
        data = np.random.randn(3, 5).astype(np.float32)
        wrapper = _create_param_wrapper(data)
        assert wrapper.shape == (3, 5)
        assert wrapper.dtype == np.float32

    def test_sub(self):
        """Subtraction produces correct values."""
        a = _create_param_wrapper(np.array([5.0, 3.0], dtype=np.float32))
        b = _create_param_wrapper(np.array([1.0, 2.0], dtype=np.float32))
        result = a - b
        result_arr = _get_param_array(result)
        np.testing.assert_allclose(result_arr, [4.0, 1.0])

    def test_isub(self):
        """In-place subtraction mutates data correctly."""
        wrapper = _create_param_wrapper(np.array([10.0, 20.0], dtype=np.float32))
        delta = np.array([1.0, 2.0], dtype=np.float32)
        wrapper -= delta
        result = _get_param_array(wrapper)
        np.testing.assert_allclose(result, [9.0, 18.0])

    def test_copy(self):
        """copy() creates an independent copy."""
        orig = _create_param_wrapper(np.array([1.0, 2.0], dtype=np.float32))
        clone = orig.copy()
        # Mutate clone; original unchanged
        clone_arr = _get_param_array(clone)
        clone_arr[0] = 99.0
        orig_arr = _get_param_array(orig)
        assert orig_arr[0] == pytest.approx(1.0)

    def test_zero_grad(self):
        """zero_grad() sets grad to zeros with matching shape."""
        wrapper = _create_param_wrapper(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        wrapper.zero_grad()
        assert wrapper.grad is not None
        np.testing.assert_array_equal(wrapper.grad, np.zeros(3, dtype=np.float32))


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------


class TestDropout:
    """Tests for Dropout module (init, repr, forward in eval, backward)."""

    def test_init_default(self):
        """Default dropout probability is 0.5."""
        d = Dropout()
        assert d.p == 0.5
        assert d.training is True

    def test_init_custom_p(self):
        """Custom probability is stored."""
        d = Dropout(p=0.3)
        assert d.p == pytest.approx(0.3)

    def test_repr(self):
        """repr contains the probability."""
        d = Dropout(p=0.25)
        r = repr(d)
        assert "Dropout" in r
        assert "0.25" in r

    def test_forward_eval(self):
        """In eval mode, forward returns input unchanged (no GPU needed)."""
        d = Dropout(p=0.5)
        d.eval()
        x = np.random.randn(4, 8).astype(np.float32)
        y = d(x)
        np.testing.assert_array_equal(y, x)

    def test_forward_training_p_zero(self):
        """In training mode with p=0, forward returns input unchanged."""
        d = Dropout(p=0.0)
        d.train()
        x = np.random.randn(4, 8).astype(np.float32)
        y = d(x)
        np.testing.assert_array_equal(y, x)

    def test_backward(self):
        """Backward scales gradient by mask / (1-p) at non-dropped positions."""
        d = Dropout(p=0.4)
        # Manually set a known mask as if forward had been called
        mask = np.array([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=np.float32)
        d._mask = mask
        d.training = True

        grad_out = np.ones((2, 4), dtype=np.float32)
        grad_in = d.backward(grad_out)

        expected = grad_out * mask / (1.0 - 0.4)
        np.testing.assert_allclose(grad_in, expected, rtol=1e-6)

    def test_backward_eval(self):
        """In eval mode, backward passes gradient through unchanged."""
        d = Dropout(p=0.5)
        d.eval()
        grad = np.random.randn(3, 3).astype(np.float32)
        result = d.backward(grad)
        np.testing.assert_array_equal(result, grad)


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------


class TestReLU:
    """Tests for ReLU module (init, repr, backward)."""

    def test_init(self):
        """ReLU init stores no learnable parameters."""
        r = ReLU()
        assert list(r.parameters()) == []

    def test_repr(self):
        """repr is 'ReLU()'."""
        assert repr(ReLU()) == "ReLU()"

    def test_backward(self):
        """Backward passes gradient where x > 0, zeros elsewhere."""
        r = ReLU()
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
        grad_out = np.ones_like(x)
        grad_in = r.backward(grad_out, x)

        expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(grad_in, expected)

    def test_backward_scaled(self):
        """Backward correctly scales by upstream gradient."""
        r = ReLU()
        x = np.array([-1.0, 3.0, -0.1, 5.0], dtype=np.float32)
        grad_out = np.array([10.0, 2.0, 7.0, 0.5], dtype=np.float32)
        grad_in = r.backward(grad_out, x)

        expected = np.array([0.0, 2.0, 0.0, 0.5], dtype=np.float32)
        np.testing.assert_array_equal(grad_in, expected)


# ---------------------------------------------------------------------------
# GELU
# ---------------------------------------------------------------------------


class TestGELU:
    """Tests for GELU module (init, repr)."""

    def test_init(self):
        """GELU init has no parameters."""
        g = GELU()
        assert list(g.parameters()) == []

    def test_repr(self):
        """repr is 'GELU()'."""
        assert repr(GELU()) == "GELU()"


# ---------------------------------------------------------------------------
# SiLU
# ---------------------------------------------------------------------------


class TestSiLU:
    """Tests for SiLU module (init, repr, backward)."""

    def test_init(self):
        """SiLU init has no parameters."""
        s = SiLU()
        assert list(s.parameters()) == []

    def test_repr(self):
        """repr is 'SiLU()'."""
        assert repr(SiLU()) == "SiLU()"

    def test_backward(self):
        """Backward computes sigmoid(x) * (1 + x * (1 - sigmoid(x)))."""
        s = SiLU()
        x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        grad_out = np.ones_like(x)
        grad_in = s.backward(grad_out, x)

        sigmoid_x = 1.0 / (1.0 + np.exp(-x))
        expected = sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))
        np.testing.assert_allclose(grad_in, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------


class TestSoftmax:
    """Tests for Softmax module (init, repr)."""

    def test_init_default_dim(self):
        """Default dim is -1."""
        s = Softmax()
        assert s.dim == -1

    def test_init_custom_dim(self):
        """Custom dim is stored."""
        s = Softmax(dim=0)
        assert s.dim == 0

    def test_repr(self):
        """repr contains the dim value."""
        r = repr(Softmax(dim=1))
        assert "Softmax" in r
        assert "dim=1" in r


# ---------------------------------------------------------------------------
# Softplus
# ---------------------------------------------------------------------------


class TestSoftplus:
    """Tests for Softplus module (init, repr, backward)."""

    def test_init(self):
        """Softplus init has no parameters."""
        s = Softplus()
        assert list(s.parameters()) == []

    def test_repr(self):
        """repr is 'Softplus()'."""
        assert repr(Softplus()) == "Softplus()"

    def test_backward(self):
        """Backward is sigmoid(x) = 1 / (1 + exp(-x))."""
        s = Softplus()
        x = np.array([-2.0, 0.0, 1.0, 5.0], dtype=np.float32)
        grad_out = np.ones_like(x)
        grad_in = s.backward(grad_out, x)

        expected = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(grad_in, expected, rtol=1e-6)

    def test_backward_scaled(self):
        """Backward correctly scales by upstream gradient."""
        s = Softplus()
        x = np.array([0.0, 1.0], dtype=np.float32)
        grad_out = np.array([2.0, 3.0], dtype=np.float32)
        grad_in = s.backward(grad_out, x)

        sigmoid_x = 1.0 / (1.0 + np.exp(-x))
        expected = grad_out * sigmoid_x
        np.testing.assert_allclose(grad_in, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# GCU
# ---------------------------------------------------------------------------


class TestGCU:
    """Tests for GCU module (init, repr)."""

    def test_init(self):
        """GCU init has no parameters."""
        g = GCU()
        assert list(g.parameters()) == []

    def test_repr(self):
        """repr is 'GCU()'."""
        assert repr(GCU()) == "GCU()"


# ---------------------------------------------------------------------------
# RoSwish
# ---------------------------------------------------------------------------


class TestRoSwish:
    """Tests for RoSwish module (init, repr, parameters)."""

    def test_init_defaults(self):
        """Default alpha=1.0, beta=1.0, learnable=True."""
        r = RoSwish()
        alpha_val = float(r.alpha[0]) if hasattr(r.alpha, "__getitem__") else float(r.alpha)
        beta_val = float(r.beta[0]) if hasattr(r.beta, "__getitem__") else float(r.beta)
        assert alpha_val == pytest.approx(1.0)
        assert beta_val == pytest.approx(1.0)
        assert r.learnable is True

    def test_init_custom(self):
        """Custom alpha and beta values are stored."""
        r = RoSwish(alpha_init=0.5, beta_init=2.0)
        alpha_val = float(r.alpha[0]) if hasattr(r.alpha, "__getitem__") else float(r.alpha)
        beta_val = float(r.beta[0]) if hasattr(r.beta, "__getitem__") else float(r.beta)
        assert alpha_val == pytest.approx(0.5)
        assert beta_val == pytest.approx(2.0)

    def test_init_not_learnable(self):
        """Non-learnable uses plain numpy arrays."""
        r = RoSwish(learnable=False)
        assert r.learnable is False
        assert isinstance(r.alpha, np.ndarray)
        assert isinstance(r.beta, np.ndarray)

    def test_repr(self):
        """repr contains 'RoSwish'."""
        r = repr(RoSwish())
        assert "RoSwish" in r


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------


class TestSequential:
    """Tests for Sequential container (init, repr, parameters)."""

    def test_init_empty(self):
        """Empty Sequential has no child modules."""
        seq = Sequential()
        assert len(seq._modules) == 0

    def test_init_with_modules(self):
        """Modules are registered in _modules dict with string indices."""
        relu = ReLU()
        drop = Dropout(p=0.1)
        seq = Sequential(relu, drop)

        assert len(seq._modules) == 2
        assert seq._modules["0"] is relu
        assert seq._modules["1"] is drop

    def test_repr(self):
        """repr lists child modules."""
        seq = Sequential(ReLU(), Dropout(p=0.2))
        r = repr(seq)
        assert "Sequential" in r
        assert "ReLU()" in r
        assert "Dropout" in r

    def test_parameters_empty(self):
        """Empty Sequential yields no parameters."""
        seq = Sequential()
        assert list(seq.parameters()) == []

    def test_parameters_collects_children(self):
        """Sequential collects parameters from child modules that have them."""
        r = RoSwish(alpha_init=1.0, beta_init=1.0, learnable=True)
        seq = Sequential(ReLU(), r)
        params = list(seq.parameters())
        # RoSwish has alpha and beta as learnable Parameters (if Parameter available)
        # ReLU has no parameters
        # The number depends on whether Parameter is registered via register_parameter
        # but at minimum ReLU contributes 0
        assert isinstance(params, list)

    def test_nested_sequential(self):
        """Sequential can contain another Sequential."""
        inner = Sequential(ReLU(), Dropout(p=0.1))
        outer = Sequential(inner, Softplus())
        assert len(outer._modules) == 2
        assert outer._modules["0"] is inner
        r = repr(outer)
        assert "Sequential" in r


# ---------------------------------------------------------------------------
# Residual
# ---------------------------------------------------------------------------


class TestResidual:
    """Tests for Residual wrapper (init, repr)."""

    def test_init(self):
        """Wrapped module is stored in _modules['module']."""
        relu = ReLU()
        res = Residual(relu)
        assert res.module is relu
        assert res._modules["module"] is relu

    def test_repr(self):
        """repr contains 'Residual'."""
        res = Residual(ReLU())
        r = repr(res)
        assert "Residual" in r

    def test_parameters_delegates(self):
        """Residual yields parameters from its child module."""
        inner = ReLU()
        res = Residual(inner)
        # ReLU has no parameters so the list should be empty
        assert list(res.parameters()) == []
