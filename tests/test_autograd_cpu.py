"""
Comprehensive CPU-only tests for nn/autograd.py.

Tests the full autograd system: Variable creation, gradient context managers,
factory functions, arithmetic (forward + backward), activations, reductions,
statistical ops, shape ops, comparisons, trigonometric functions, losses,
the where op, custom Function subclass, and backward integration chains.
"""

import numpy as np
import pytest

try:
    from grilly.nn.autograd import (
        Function,
        FunctionCtx,
        Variable,
        abs,
        acos,
        add,
        arange,
        asin,
        atan,
        atan2,
        bce_loss,
        bce_with_logits_loss,
        clamp,
        clone,
        concat,
        contiguous,
        cos,
        cross_entropy,
        div,
        elu,
        enable_grad,
        eq,
        exp,
        expand,
        eye,
        flatten,
        full,
        ge,
        gelu,
        gt,
        index,
        is_grad_enabled,
        kl_div_loss,
        l1_loss,
        le,
        leaky_relu,
        linspace,
        log,
        lt,
        matmul,
        max,
        mean,
        min,
        mse_loss,
        mul,
        ne,
        neg,
        nll_loss,
        no_grad,
        norm,
        ones,
        permute,
        pow,
        rand,
        randn,
        relu,
        repeat,
        reshape,
        sigmoid,
        silu,
        sin,
        smooth_l1_loss,
        softmax,
        softplus,
        sqrt,
        squeeze,
        stack,
        std,
        sub,
        sum,
        tan,
        tanh,
        tensor,
        transpose,
        unsqueeze,
        var,
        view,
        where,
        zeros,
    )
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def seed_rng():
    """Seed the RNG before every test for reproducibility."""
    np.random.seed(42)


# ---------------------------------------------------------------------------
# Numerical gradient helper
# ---------------------------------------------------------------------------


def _numerical_grad(fn, x, eps=1e-3):
    """Compute numerical gradient of a scalar-valued function w.r.t. Variable x.

    Works entirely in float32 to match the autograd computation. Uses a
    slightly larger default eps (1e-3) to reduce float32 rounding noise.
    """
    x_data = x.data.copy()
    grad = np.zeros_like(x_data)
    flat = x_data.flatten()
    for i in range(len(flat)):
        old = flat[i]
        flat[i] = np.float32(old + eps)
        loss_plus = float(fn(Variable(flat.reshape(x.shape).copy())).data)
        flat[i] = np.float32(old - eps)
        loss_minus = float(fn(Variable(flat.reshape(x.shape).copy())).data)
        actual_delta = float(np.float32(old + eps)) - float(np.float32(old - eps))
        flat[i] = old
        grad.flat[i] = (loss_plus - loss_minus) / actual_delta
    return grad


# ===========================================================================
# 1. TestVariableCreation
# ===========================================================================


class TestVariableCreation:
    """Test Variable construction from different inputs and basic properties."""

    def test_from_list(self):
        v = Variable([1.0, 2.0, 3.0])
        assert v.data.dtype == np.float32
        assert v.shape == (3,)
        np.testing.assert_allclose(v.data, [1.0, 2.0, 3.0])

    def test_from_scalar(self):
        v = Variable(5.0)
        assert v.shape == ()
        assert v.item() == pytest.approx(5.0)

    def test_from_int_scalar(self):
        v = Variable(3)
        assert v.data.dtype == np.float32
        assert v.item() == pytest.approx(3.0)

    def test_from_ndarray(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        v = Variable(arr)
        assert v.data.dtype == np.float32
        assert v.shape == (2, 2)

    def test_from_variable(self):
        v1 = Variable([1.0, 2.0])
        v2 = Variable(v1)
        np.testing.assert_allclose(v2.data, v1.data)

    def test_shape_property(self):
        v = Variable(np.zeros((2, 3, 4), dtype=np.float32))
        assert v.shape == (2, 3, 4)

    def test_ndim_property(self):
        v = Variable(np.zeros((2, 3), dtype=np.float32))
        assert v.ndim == 2

    def test_item_scalar(self):
        v = Variable(7.5)
        assert v.item() == pytest.approx(7.5)

    def test_numpy_returns_copy(self):
        v = Variable([1.0, 2.0])
        arr = v.numpy()
        arr[0] = 999.0
        assert v.data[0] == pytest.approx(1.0), "numpy() should return a copy"

    def test_detach(self):
        v = Variable([1.0, 2.0], requires_grad=True)
        d = v.detach()
        assert d.requires_grad is False
        assert d.grad_fn is None
        np.testing.assert_allclose(d.data, v.data)
        # Mutating detached should not affect original
        d.data[0] = 999.0
        assert v.data[0] == pytest.approx(1.0)

    def test_zero_grad(self):
        v = Variable([1.0], requires_grad=True)
        v.grad = np.array([42.0], dtype=np.float32)
        v.zero_grad()
        assert v.grad is None

    def test_is_leaf_true(self):
        v = Variable([1.0], requires_grad=True)
        assert v.is_leaf is True

    def test_is_leaf_false_after_op(self):
        a = Variable([1.0, 2.0], requires_grad=True)
        b = a + 1
        assert b.is_leaf is False

    def test_repr(self):
        v = Variable([1.0], requires_grad=True)
        r = repr(v)
        assert "Variable" in r
        assert "requires_grad=True" in r

    def test_str(self):
        v = Variable([1.0, 2.0])
        s = str(v)
        assert "1." in s

    def test_requires_grad_default_false(self):
        v = Variable([1.0])
        assert v.requires_grad is False


# ===========================================================================
# 2. TestGradContext
# ===========================================================================


class TestGradContext:
    """Test no_grad, enable_grad, and is_grad_enabled."""

    def test_is_grad_enabled_default(self):
        assert is_grad_enabled() is True

    def test_no_grad_disables(self):
        with no_grad():
            assert is_grad_enabled() is False
        assert is_grad_enabled() is True

    def test_enable_grad_enables(self):
        with no_grad():
            with enable_grad():
                assert is_grad_enabled() is True
            assert is_grad_enabled() is False
        assert is_grad_enabled() is True

    def test_nesting_no_grad(self):
        with no_grad():
            with no_grad():
                assert is_grad_enabled() is False
            assert is_grad_enabled() is False
        assert is_grad_enabled() is True

    def test_no_grad_prevents_graph_building(self):
        a = Variable([1.0, 2.0], requires_grad=True)
        with no_grad():
            b = a * 2
        # b should not have a grad_fn because graph building was disabled
        assert b.grad_fn is None


# ===========================================================================
# 3. TestFactoryFunctions
# ===========================================================================


class TestFactoryFunctions:
    """Test tensor, zeros, ones, randn, rand, linspace, arange, eye, full."""

    def test_tensor(self):
        t = tensor([1.0, 2.0], requires_grad=True)
        assert isinstance(t, Variable)
        assert t.requires_grad is True

    def test_zeros(self):
        z = zeros((2, 3))
        assert z.shape == (2, 3)
        assert np.all(z.data == 0)
        assert z.data.dtype == np.float32

    def test_ones(self):
        o = ones((3, 2))
        assert o.shape == (3, 2)
        assert np.all(o.data == 1)

    def test_randn_shape(self):
        r = randn(4, 5, requires_grad=True)
        assert r.shape == (4, 5)
        assert r.requires_grad is True

    def test_randn_scalar(self):
        r = randn()
        assert r.shape == ()

    def test_rand_shape(self):
        r = rand(3, 4)
        assert r.shape == (3, 4)
        assert np.all(r.data >= 0) and np.all(r.data < 1)

    def test_rand_scalar(self):
        r = rand()
        assert r.shape == ()

    def test_linspace(self):
        l = linspace(0, 1, 5)
        assert l.shape == (5,)
        np.testing.assert_allclose(l.data, [0.0, 0.25, 0.5, 0.75, 1.0], atol=1e-6)

    def test_arange_one_arg(self):
        a = arange(5)
        assert a.shape == (5,)
        np.testing.assert_allclose(a.data, [0, 1, 2, 3, 4], atol=1e-6)

    def test_arange_three_arg(self):
        a = arange(1, 7, 2)
        np.testing.assert_allclose(a.data, [1, 3, 5], atol=1e-6)

    def test_eye_square(self):
        e = eye(3)
        assert e.shape == (3, 3)
        np.testing.assert_allclose(e.data, np.eye(3, dtype=np.float32))

    def test_eye_rectangular(self):
        e = eye(2, 4)
        assert e.shape == (2, 4)
        expected = np.eye(2, 4, dtype=np.float32)
        np.testing.assert_allclose(e.data, expected)

    def test_full(self):
        f = full((2, 3), 7.0)
        assert f.shape == (2, 3)
        assert np.all(f.data == 7.0)


# ===========================================================================
# 4. TestArithmeticForward
# ===========================================================================


class TestArithmeticForward:
    """Test forward pass of add, sub, mul, div, neg, pow and broadcast variants."""

    def test_add(self):
        a = Variable([1.0, 2.0, 3.0])
        b = Variable([4.0, 5.0, 6.0])
        c = add(a, b)
        np.testing.assert_allclose(c.data, [5.0, 7.0, 9.0])

    def test_add_scalar(self):
        a = Variable([1.0, 2.0])
        c = add(a, 10.0)
        np.testing.assert_allclose(c.data, [11.0, 12.0])

    def test_add_broadcast(self):
        a = Variable(np.ones((2, 3), dtype=np.float32))
        b = Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        c = add(a, b)
        expected = np.ones((2, 3)) + np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(c.data, expected.astype(np.float32))

    def test_sub(self):
        a = Variable([5.0, 6.0])
        b = Variable([1.0, 2.0])
        c = sub(a, b)
        np.testing.assert_allclose(c.data, [4.0, 4.0])

    def test_sub_scalar(self):
        a = Variable([5.0])
        c = sub(a, 3.0)
        np.testing.assert_allclose(c.data, [2.0])

    def test_mul(self):
        a = Variable([2.0, 3.0])
        b = Variable([4.0, 5.0])
        c = mul(a, b)
        np.testing.assert_allclose(c.data, [8.0, 15.0])

    def test_mul_scalar(self):
        a = Variable([2.0, 3.0])
        c = mul(a, 3.0)
        np.testing.assert_allclose(c.data, [6.0, 9.0])

    def test_mul_broadcast(self):
        a = Variable(np.ones((2, 3), dtype=np.float32) * 2)
        b = Variable(np.array([[3.0]], dtype=np.float32))
        c = mul(a, b)
        np.testing.assert_allclose(c.data, np.full((2, 3), 6.0, dtype=np.float32))

    def test_div(self):
        a = Variable([6.0, 8.0])
        b = Variable([2.0, 4.0])
        c = div(a, b)
        np.testing.assert_allclose(c.data, [3.0, 2.0])

    def test_div_scalar(self):
        a = Variable([6.0, 9.0])
        c = div(a, 3.0)
        np.testing.assert_allclose(c.data, [2.0, 3.0])

    def test_neg(self):
        a = Variable([1.0, -2.0, 3.0])
        c = neg(a)
        np.testing.assert_allclose(c.data, [-1.0, 2.0, -3.0])

    def test_pow_int(self):
        a = Variable([2.0, 3.0])
        c = pow(a, 3)
        np.testing.assert_allclose(c.data, [8.0, 27.0])

    def test_pow_float(self):
        a = Variable([4.0, 9.0])
        c = pow(a, 0.5)
        np.testing.assert_allclose(c.data, [2.0, 3.0], atol=1e-5)

    def test_matmul_2d(self):
        a = Variable(np.ones((2, 3), dtype=np.float32))
        b = Variable(np.ones((3, 4), dtype=np.float32))
        c = matmul(a, b)
        assert c.shape == (2, 4)
        np.testing.assert_allclose(c.data, np.full((2, 4), 3.0, dtype=np.float32))

    def test_matmul_vec(self):
        a = Variable([1.0, 2.0, 3.0])
        b = Variable([4.0, 5.0, 6.0])
        c = matmul(a, b)
        assert c.item() == pytest.approx(32.0)


# ===========================================================================
# 5. TestArithmeticOperators
# ===========================================================================


class TestArithmeticOperators:
    """Test operator overloads (+, -, *, /, **, -x, @) and reverse ops."""

    def test_add_op(self):
        a = Variable([1.0, 2.0])
        b = Variable([3.0, 4.0])
        c = a + b
        np.testing.assert_allclose(c.data, [4.0, 6.0])

    def test_sub_op(self):
        a = Variable([5.0, 6.0])
        b = Variable([1.0, 2.0])
        c = a - b
        np.testing.assert_allclose(c.data, [4.0, 4.0])

    def test_mul_op(self):
        a = Variable([2.0, 3.0])
        b = Variable([4.0, 5.0])
        c = a * b
        np.testing.assert_allclose(c.data, [8.0, 15.0])

    def test_div_op(self):
        a = Variable([6.0, 8.0])
        b = Variable([2.0, 4.0])
        c = a / b
        np.testing.assert_allclose(c.data, [3.0, 2.0])

    def test_pow_op(self):
        a = Variable([2.0, 3.0])
        c = a ** 2
        np.testing.assert_allclose(c.data, [4.0, 9.0])

    def test_neg_op(self):
        a = Variable([1.0, -2.0])
        c = -a
        np.testing.assert_allclose(c.data, [-1.0, 2.0])

    def test_matmul_op(self):
        a = Variable(np.eye(2, dtype=np.float32))
        b = Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        c = a @ b
        np.testing.assert_allclose(c.data, b.data)

    def test_radd(self):
        a = Variable([1.0, 2.0])
        c = 5.0 + a
        np.testing.assert_allclose(c.data, [6.0, 7.0])

    def test_rsub(self):
        a = Variable([1.0, 2.0])
        c = 10.0 - a
        np.testing.assert_allclose(c.data, [9.0, 8.0])

    def test_rmul(self):
        a = Variable([2.0, 3.0])
        c = 3.0 * a
        np.testing.assert_allclose(c.data, [6.0, 9.0])

    def test_rtruediv(self):
        a = Variable([2.0, 4.0])
        c = 8.0 / a
        np.testing.assert_allclose(c.data, [4.0, 2.0])


# ===========================================================================
# 6. TestArithmeticBackward
# ===========================================================================


class TestArithmeticBackward:
    """Backward tests for arithmetic ops with numerical gradient checks."""

    def test_add_backward(self):
        a = Variable([1.0, 2.0, 3.0], requires_grad=True)
        b = Variable([4.0, 5.0, 6.0], requires_grad=True)
        c = add(a, b)
        loss = sum(c)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(a.grad, np.ones(3, dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(b.grad, np.ones(3, dtype=np.float32), atol=1e-5)

    def test_sub_backward(self):
        a = Variable([1.0, 2.0], requires_grad=True)
        b = Variable([3.0, 4.0], requires_grad=True)
        c = sub(a, b)
        loss = sum(c)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(a.grad, [1.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(b.grad, [-1.0, -1.0], atol=1e-5)

    def test_mul_backward(self):
        a = Variable([2.0, 3.0], requires_grad=True)
        b = Variable([4.0, 5.0], requires_grad=True)
        c = mul(a, b)
        loss = sum(c)
        loss.backward(use_gpu=False)
        # d(a*b)/da = b, d(a*b)/db = a
        np.testing.assert_allclose(a.grad, [4.0, 5.0], atol=1e-5)
        np.testing.assert_allclose(b.grad, [2.0, 3.0], atol=1e-5)

    def test_div_backward(self):
        a = Variable([6.0, 8.0], requires_grad=True)
        b = Variable([2.0, 4.0], requires_grad=True)
        c = div(a, b)
        loss = sum(c)
        loss.backward(use_gpu=False)
        # d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        np.testing.assert_allclose(a.grad, [0.5, 0.25], atol=1e-5)
        np.testing.assert_allclose(b.grad, [-1.5, -0.5], atol=1e-5)

    def test_neg_backward(self):
        a = Variable([1.0, -2.0], requires_grad=True)
        c = neg(a)
        loss = sum(c)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(a.grad, [-1.0, -1.0], atol=1e-5)

    def test_pow_backward_numerical(self):
        x = Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(pow(v, 3))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_matmul_backward(self):
        a = Variable(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
        b = Variable(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        c = matmul(a, b)
        loss = sum(c)
        loss.backward(use_gpu=False)
        # d(sum(A@B))/dA = ones @ B^T, d(sum(A@B))/dB = A^T @ ones
        grad_out = np.ones((2, 4), dtype=np.float32)
        expected_a = grad_out @ b.data.T
        expected_b = a.data.T @ grad_out
        np.testing.assert_allclose(a.grad, expected_a, atol=1e-5)
        np.testing.assert_allclose(b.grad, expected_b, atol=1e-5)

    def test_add_broadcast_backward(self):
        a = Variable(np.ones((2, 3), dtype=np.float32), requires_grad=True)
        b = Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        c = add(a, b)
        loss = sum(c)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(a.grad, np.ones((2, 3), dtype=np.float32), atol=1e-5)
        # b is broadcast over dim 0 (size 2), so its grad should sum over that dim
        np.testing.assert_allclose(b.grad, [2.0, 2.0, 2.0], atol=1e-5)

    def test_mul_scalar_backward(self):
        x = Variable([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 5.0
        loss = sum(y)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, [5.0, 5.0, 5.0], atol=1e-5)


# ===========================================================================
# 7. TestActivationForward
# ===========================================================================


class TestActivationForward:
    """Test forward pass for all activation functions."""

    def test_relu(self):
        x = Variable([-1.0, 0.0, 1.0, 2.0])
        y = relu(x)
        np.testing.assert_allclose(y.data, [0.0, 0.0, 1.0, 2.0])

    def test_sigmoid(self):
        x = Variable([0.0])
        y = sigmoid(x)
        assert y.item() == pytest.approx(0.5)

    def test_tanh(self):
        x = Variable([0.0])
        y = tanh(x)
        assert y.item() == pytest.approx(0.0)

    def test_exp(self):
        x = Variable([0.0, 1.0])
        y = exp(x)
        np.testing.assert_allclose(y.data, [1.0, np.e], atol=1e-5)

    def test_log(self):
        x = Variable([1.0, np.e])
        y = log(x)
        np.testing.assert_allclose(y.data, [0.0, 1.0], atol=1e-5)

    def test_sqrt(self):
        x = Variable([4.0, 9.0, 16.0])
        y = sqrt(x)
        np.testing.assert_allclose(y.data, [2.0, 3.0, 4.0], atol=1e-5)

    def test_abs(self):
        x = Variable([-3.0, 0.0, 2.0])
        y = abs(x)
        np.testing.assert_allclose(y.data, [3.0, 0.0, 2.0])

    def test_clamp_min_only(self):
        x = Variable([-2.0, 0.0, 3.0])
        y = clamp(x, min_val=0.0)
        np.testing.assert_allclose(y.data, [0.0, 0.0, 3.0])

    def test_clamp_max_only(self):
        x = Variable([-2.0, 0.0, 3.0])
        y = clamp(x, max_val=1.0)
        np.testing.assert_allclose(y.data, [-2.0, 0.0, 1.0])

    def test_clamp_both(self):
        x = Variable([-2.0, 0.5, 3.0])
        y = clamp(x, min_val=-1.0, max_val=1.0)
        np.testing.assert_allclose(y.data, [-1.0, 0.5, 1.0])

    def test_gelu(self):
        x = Variable([0.0])
        y = gelu(x)
        assert y.item() == pytest.approx(0.0, abs=1e-5)

    def test_gelu_positive(self):
        x = Variable([2.0])
        y = gelu(x)
        # GELU(2) ~= 1.9545 (approx)
        assert y.item() > 1.9

    def test_silu(self):
        x = Variable([0.0])
        y = silu(x)
        # SiLU(0) = 0 * sigmoid(0) = 0
        assert y.item() == pytest.approx(0.0)

    def test_leaky_relu(self):
        x = Variable([-1.0, 0.0, 1.0])
        y = leaky_relu(x, negative_slope=0.1)
        np.testing.assert_allclose(y.data, [-0.1, 0.0, 1.0])

    def test_elu(self):
        x = Variable([-1.0, 0.0, 1.0])
        y = elu(x, alpha=1.0)
        expected = np.where(
            np.array([-1.0, 0.0, 1.0]) >= 0,
            [-1.0, 0.0, 1.0],
            1.0 * (np.exp([-1.0, 0.0, 1.0]) - 1),
        )
        np.testing.assert_allclose(y.data, expected, atol=1e-5)

    def test_softplus(self):
        x = Variable([0.0])
        y = softplus(x)
        # softplus(0) = log(2)
        assert y.item() == pytest.approx(np.log(2), abs=1e-5)

    def test_softplus_large(self):
        x = Variable([100.0])
        y = softplus(x, threshold=20.0)
        # For large x, softplus ~ x
        assert y.item() == pytest.approx(100.0, abs=1e-3)

    def test_softmax(self):
        x = Variable([1.0, 2.0, 3.0])
        y = softmax(x, dim=-1)
        assert y.item is not None  # ensure it's a Variable
        assert np.allclose(np.sum(y.data), 1.0, atol=1e-5)
        # values should be increasing
        assert y.data[0] < y.data[1] < y.data[2]

    def test_softmax_2d(self):
        x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        y = softmax(x, dim=-1)
        row_sums = np.sum(y.data, axis=-1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0], atol=1e-5)


# ===========================================================================
# 8. TestActivationBackward
# ===========================================================================


class TestActivationBackward:
    """Gradient checks for all activation functions."""

    def test_relu_backward(self):
        x = Variable(np.array([-1.0, 0.5, 2.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(relu(v))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_sigmoid_backward(self):
        x = Variable(np.array([0.0, 1.0, -1.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(sigmoid(v))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_tanh_backward(self):
        x = Variable(np.array([0.0, 0.5, -0.5], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(tanh(v))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_exp_backward(self):
        x = Variable(np.array([0.0, 1.0, -1.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(exp(v))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_log_backward(self):
        x = Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(log(v))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_sqrt_backward(self):
        x = Variable(np.array([1.0, 4.0, 9.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(sqrt(v))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_abs_backward(self):
        # Avoid exact zero where grad is undefined
        x = Variable(np.array([-2.0, 1.0, 3.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(abs(v))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_clamp_backward(self):
        x = Variable(np.array([-2.0, 0.5, 3.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(clamp(v, min_val=-1.0, max_val=1.0))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_gelu_backward(self):
        x = Variable(np.array([-1.0, 0.0, 1.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(gelu(v))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_silu_backward(self):
        x = Variable(np.array([-1.0, 0.0, 1.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(silu(v))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_leaky_relu_backward(self):
        x = Variable(np.array([-1.0, 0.5, 2.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(leaky_relu(v, negative_slope=0.1))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_elu_backward(self):
        x = Variable(np.array([-1.0, 0.5, 2.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(elu(v, alpha=1.0))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_softplus_backward(self):
        x = Variable(np.array([-1.0, 0.0, 1.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(softplus(v))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_softmax_backward(self):
        x = Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            s = softmax(v, dim=-1)
            return sum(s * Variable([1.0, 0.0, 0.0]))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)


# ===========================================================================
# 9. TestReductions
# ===========================================================================


class TestReductions:
    """Test sum, mean, max, min reductions and their backward passes."""

    def test_sum_global(self):
        x = Variable([1.0, 2.0, 3.0])
        s = sum(x)
        assert s.item() == pytest.approx(6.0)

    def test_sum_dim(self):
        x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        s = sum(x, dim=0)
        np.testing.assert_allclose(s.data, [4.0, 6.0])

    def test_sum_keepdims(self):
        x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        s = sum(x, dim=1, keepdims=True)
        assert s.shape == (2, 1)
        np.testing.assert_allclose(s.data, [[3.0], [7.0]])

    def test_mean_global(self):
        x = Variable([2.0, 4.0, 6.0])
        m = mean(x)
        assert m.item() == pytest.approx(4.0)

    def test_mean_dim(self):
        x = Variable(np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32))
        m = mean(x, dim=1)
        np.testing.assert_allclose(m.data, [2.0, 6.0])

    def test_mean_keepdims(self):
        x = Variable(np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32))
        m = mean(x, dim=0, keepdims=True)
        assert m.shape == (1, 2)
        np.testing.assert_allclose(m.data, [[3.0, 5.0]])

    def test_max_global(self):
        x = Variable([1.0, 5.0, 3.0])
        m = max(x)
        assert m.data == pytest.approx(5.0)

    def test_max_dim(self):
        x = Variable(np.array([[1.0, 3.0], [4.0, 2.0]], dtype=np.float32))
        m = max(x, dim=1)
        np.testing.assert_allclose(m.data, [3.0, 4.0])

    def test_min_global(self):
        x = Variable([1.0, 5.0, 3.0])
        m = min(x)
        assert m.data == pytest.approx(1.0)

    def test_min_dim(self):
        x = Variable(np.array([[1.0, 3.0], [4.0, 2.0]], dtype=np.float32))
        m = min(x, dim=1)
        np.testing.assert_allclose(m.data, [1.0, 2.0])

    def test_sum_backward_global(self):
        x = Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        loss = sum(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, [1.0, 1.0, 1.0], atol=1e-5)

    def test_sum_backward_dim(self):
        x = Variable(np.ones((2, 3), dtype=np.float32), requires_grad=True)
        s = sum(x, dim=1)  # shape (2,)
        loss = sum(s)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, np.ones((2, 3), dtype=np.float32), atol=1e-5)

    def test_mean_backward_global(self):
        x = Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        loss = mean(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, np.full(3, 1.0 / 3, dtype=np.float32), atol=1e-5)

    def test_mean_backward_dim(self):
        x = Variable(np.ones((2, 3), dtype=np.float32), requires_grad=True)
        m = mean(x, dim=1)  # shape (2,)
        loss = sum(m)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(
            x.grad, np.full((2, 3), 1.0 / 3, dtype=np.float32), atol=1e-5
        )

    def test_max_backward_global(self):
        x = Variable(np.array([1.0, 5.0, 3.0], dtype=np.float32), requires_grad=True)
        m = max(x)
        m.backward(use_gpu=False)
        # Gradient should be 1 at the max position (index 1) and 0 elsewhere
        expected = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(x.grad, expected, atol=1e-5)

    def test_min_backward_global(self):
        x = Variable(np.array([3.0, 1.0, 5.0], dtype=np.float32), requires_grad=True)
        m = min(x)
        m.backward(use_gpu=False)
        expected = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(x.grad, expected, atol=1e-5)


# ===========================================================================
# 10. TestStatistical
# ===========================================================================


class TestStatistical:
    """Test var, std, norm functions and their backward passes."""

    def test_var_global(self):
        x = Variable([1.0, 2.0, 3.0, 4.0, 5.0])
        v = var(x)
        expected = np.var([1, 2, 3, 4, 5], ddof=1)
        assert v.item() == pytest.approx(float(expected), abs=1e-4)

    def test_var_dim(self):
        x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        v = var(x, dim=0)
        expected = np.var([[1, 2], [3, 4]], axis=0, ddof=1).astype(np.float32)
        np.testing.assert_allclose(v.data, expected, atol=1e-5)

    def test_var_unbiased_false(self):
        x = Variable([1.0, 2.0, 3.0])
        v = var(x, unbiased=False)
        expected = np.var([1, 2, 3], ddof=0)
        assert v.item() == pytest.approx(float(expected), abs=1e-5)

    def test_std_global(self):
        x = Variable([1.0, 2.0, 3.0, 4.0, 5.0])
        s = std(x)
        expected = np.std([1, 2, 3, 4, 5], ddof=1)
        assert s.item() == pytest.approx(float(expected), abs=1e-4)

    def test_std_dim(self):
        x = Variable(np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32))
        s = std(x, dim=1)
        expected = np.std([[1, 3], [5, 7]], axis=1, ddof=1).astype(np.float32)
        np.testing.assert_allclose(s.data, expected, atol=1e-5)

    def test_std_unbiased_false(self):
        x = Variable([2.0, 4.0, 6.0])
        s = std(x, unbiased=False)
        expected = np.std([2, 4, 6], ddof=0)
        assert s.item() == pytest.approx(float(expected), abs=1e-5)

    def test_norm_l2_global(self):
        x = Variable([3.0, 4.0])
        n = norm(x, p=2)
        assert n.item() == pytest.approx(5.0, abs=1e-5)

    def test_norm_l1_global(self):
        x = Variable([-3.0, 4.0])
        n = norm(x, p=1)
        assert n.item() == pytest.approx(7.0, abs=1e-5)

    def test_norm_linf(self):
        x = Variable([-3.0, 4.0, 1.0])
        n = norm(x, p=float("inf"))
        assert n.item() == pytest.approx(4.0, abs=1e-5)

    def test_norm_p3(self):
        x = Variable([1.0, 2.0, 3.0])
        n = norm(x, p=3)
        expected = (1**3 + 2**3 + 3**3) ** (1 / 3)
        assert n.item() == pytest.approx(float(expected), abs=1e-4)

    def test_var_backward(self):
        x = Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return var(v)

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_std_backward(self):
        x = Variable(np.array([1.0, 3.0, 5.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return std(v)

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_norm_l2_backward(self):
        x = Variable(np.array([3.0, 4.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return norm(v, p=2)

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_norm_l1_backward(self):
        x = Variable(np.array([-3.0, 4.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return norm(v, p=1)

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)


# ===========================================================================
# 11. TestShapeOps
# ===========================================================================


class TestShapeOps:
    """Test reshape, transpose, squeeze, unsqueeze, flatten, view, expand,
    repeat, permute, contiguous, clone, concat, stack, index."""

    def test_reshape(self):
        x = Variable(np.arange(6, dtype=np.float32))
        y = reshape(x, (2, 3))
        assert y.shape == (2, 3)

    def test_reshape_method(self):
        x = Variable(np.arange(6, dtype=np.float32))
        y = x.reshape(2, 3)
        assert y.shape == (2, 3)

    def test_transpose_default(self):
        x = Variable(np.arange(6, dtype=np.float32).reshape(2, 3))
        y = transpose(x)
        assert y.shape == (3, 2)

    def test_transpose_T_property(self):
        x = Variable(np.arange(6, dtype=np.float32).reshape(2, 3))
        y = x.T
        assert y.shape == (3, 2)

    def test_transpose_dims(self):
        x = Variable(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
        y = transpose(x, (2, 0, 1))
        assert y.shape == (4, 2, 3)

    def test_squeeze_all(self):
        x = Variable(np.zeros((1, 3, 1), dtype=np.float32))
        y = squeeze(x)
        assert y.shape == (3,)

    def test_squeeze_dim(self):
        x = Variable(np.zeros((1, 3, 1), dtype=np.float32))
        y = squeeze(x, dim=0)
        assert y.shape == (3, 1)

    def test_unsqueeze(self):
        x = Variable(np.zeros((3, 4), dtype=np.float32))
        y = unsqueeze(x, 0)
        assert y.shape == (1, 3, 4)

    def test_unsqueeze_method(self):
        x = Variable(np.zeros((3, 4), dtype=np.float32))
        y = x.unsqueeze(1)
        assert y.shape == (3, 1, 4)

    def test_flatten(self):
        x = Variable(np.zeros((2, 3, 4), dtype=np.float32))
        y = flatten(x)
        assert y.shape == (24,)

    def test_flatten_partial(self):
        x = Variable(np.zeros((2, 3, 4), dtype=np.float32))
        y = flatten(x, start_dim=1)
        assert y.shape == (2, 12)

    def test_view(self):
        x = Variable(np.arange(6, dtype=np.float32))
        y = view(x, 2, 3)
        assert y.shape == (2, 3)

    def test_view_method(self):
        x = Variable(np.arange(6, dtype=np.float32))
        y = x.view(3, 2)
        assert y.shape == (3, 2)

    def test_expand(self):
        x = Variable(np.array([[1.0], [2.0], [3.0]], dtype=np.float32))
        y = expand(x, 3, 4)
        assert y.shape == (3, 4)
        np.testing.assert_allclose(y.data[0], [1.0, 1.0, 1.0, 1.0])

    def test_expand_method(self):
        x = Variable(np.ones((1, 3), dtype=np.float32))
        y = x.expand(4, 3)
        assert y.shape == (4, 3)

    def test_repeat(self):
        x = Variable(np.array([[1.0, 2.0]], dtype=np.float32))
        y = repeat(x, 3, 2)
        assert y.shape == (3, 4)

    def test_repeat_method(self):
        x = Variable(np.ones((2, 3), dtype=np.float32))
        y = x.repeat(2, 1)
        assert y.shape == (4, 3)

    def test_permute(self):
        x = Variable(np.zeros((2, 3, 4), dtype=np.float32))
        y = permute(x, 2, 0, 1)
        assert y.shape == (4, 2, 3)

    def test_permute_method(self):
        x = Variable(np.zeros((2, 3, 4), dtype=np.float32))
        y = x.permute(1, 2, 0)
        assert y.shape == (3, 4, 2)

    def test_contiguous(self):
        x = Variable(np.arange(6, dtype=np.float32).reshape(2, 3))
        y = contiguous(x)
        assert y.shape == (2, 3)
        np.testing.assert_allclose(y.data, x.data)

    def test_clone(self):
        x = Variable([1.0, 2.0, 3.0])
        y = clone(x)
        np.testing.assert_allclose(y.data, x.data)
        y.data[0] = 999.0
        assert x.data[0] == pytest.approx(1.0), "clone should create independent copy"

    def test_concat(self):
        a = Variable(np.array([[1.0, 2.0]], dtype=np.float32))
        b = Variable(np.array([[3.0, 4.0]], dtype=np.float32))
        c = concat([a, b], dim=0)
        assert c.shape == (2, 2)
        np.testing.assert_allclose(c.data, [[1.0, 2.0], [3.0, 4.0]])

    def test_concat_dim1(self):
        a = Variable(np.array([[1.0], [2.0]], dtype=np.float32))
        b = Variable(np.array([[3.0], [4.0]], dtype=np.float32))
        c = concat([a, b], dim=1)
        assert c.shape == (2, 2)

    def test_stack(self):
        a = Variable([1.0, 2.0])
        b = Variable([3.0, 4.0])
        c = stack([a, b], dim=0)
        assert c.shape == (2, 2)
        np.testing.assert_allclose(c.data, [[1.0, 2.0], [3.0, 4.0]])

    def test_stack_dim1(self):
        a = Variable([1.0, 2.0])
        b = Variable([3.0, 4.0])
        c = stack([a, b], dim=1)
        assert c.shape == (2, 2)
        np.testing.assert_allclose(c.data, [[1.0, 3.0], [2.0, 4.0]])

    def test_index_slice(self):
        x = Variable(np.arange(6, dtype=np.float32).reshape(2, 3))
        y = index(x, (slice(None), slice(0, 2)))
        assert y.shape == (2, 2)

    def test_index_int(self):
        x = Variable(np.arange(6, dtype=np.float32).reshape(2, 3))
        y = index(x, 0)
        assert y.shape == (3,)
        np.testing.assert_allclose(y.data, [0.0, 1.0, 2.0])

    def test_getitem_operator(self):
        x = Variable(np.arange(6, dtype=np.float32).reshape(2, 3))
        y = x[1]
        assert y.shape == (3,)
        np.testing.assert_allclose(y.data, [3.0, 4.0, 5.0])


# ===========================================================================
# 12. TestShapeOpsBackward
# ===========================================================================


class TestShapeOpsBackward:
    """Backward tests for shape ops."""

    def test_reshape_backward(self):
        x = Variable(np.arange(6, dtype=np.float32), requires_grad=True)
        y = reshape(x, (2, 3))
        loss = sum(y)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, np.ones(6, dtype=np.float32), atol=1e-5)

    def test_transpose_backward(self):
        x = Variable(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
        y = transpose(x)
        loss = sum(y)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, np.ones((2, 3), dtype=np.float32), atol=1e-5)

    def test_squeeze_backward(self):
        x = Variable(np.random.randn(1, 3, 1).astype(np.float32), requires_grad=True)
        y = squeeze(x)
        loss = sum(y)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, np.ones((1, 3, 1), dtype=np.float32), atol=1e-5)

    def test_unsqueeze_backward(self):
        x = Variable(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        y = unsqueeze(x, 0)
        loss = sum(y)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, np.ones((3, 4), dtype=np.float32), atol=1e-5)

    def test_concat_backward(self):
        a = Variable(np.array([[1.0, 2.0]], dtype=np.float32), requires_grad=True)
        b = Variable(np.array([[3.0, 4.0]], dtype=np.float32), requires_grad=True)
        c = concat([a, b], dim=0)
        loss = sum(c)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(a.grad, np.ones((1, 2), dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(b.grad, np.ones((1, 2), dtype=np.float32), atol=1e-5)

    def test_stack_backward(self):
        a = Variable(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
        b = Variable(np.array([3.0, 4.0], dtype=np.float32), requires_grad=True)
        c = stack([a, b], dim=0)
        loss = sum(c)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(a.grad, np.ones(2, dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(b.grad, np.ones(2, dtype=np.float32), atol=1e-5)

    def test_flatten_backward(self):
        x = Variable(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
        y = flatten(x, start_dim=1)
        loss = sum(y)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, np.ones((2, 3, 4), dtype=np.float32), atol=1e-5)

    def test_expand_backward(self):
        x = Variable(np.array([[1.0], [2.0]], dtype=np.float32), requires_grad=True)
        y = expand(x, 2, 3)
        loss = sum(y)
        loss.backward(use_gpu=False)
        # Each row is broadcast 3 times, so grad is summed over dim 1
        expected = np.full((2, 1), 3.0, dtype=np.float32)
        np.testing.assert_allclose(x.grad, expected, atol=1e-5)

    def test_repeat_backward(self):
        x = Variable(np.array([[1.0, 2.0]], dtype=np.float32), requires_grad=True)
        y = repeat(x, 2, 1)
        loss = sum(y)
        loss.backward(use_gpu=False)
        # Repeated 2 times along dim 0, so grad is accumulated
        expected = np.full((1, 2), 2.0, dtype=np.float32)
        np.testing.assert_allclose(x.grad, expected, atol=1e-5)

    def test_clone_backward(self):
        x = Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        y = clone(x)
        loss = sum(y)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, np.ones(3, dtype=np.float32), atol=1e-5)

    def test_contiguous_backward(self):
        x = Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        y = contiguous(x)
        loss = sum(y)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, np.ones(3, dtype=np.float32), atol=1e-5)

    def test_index_backward(self):
        x = Variable(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
        y = x[0]  # first row
        loss = sum(y)
        loss.backward(use_gpu=False)
        expected = np.zeros((2, 3), dtype=np.float32)
        expected[0, :] = 1.0
        np.testing.assert_allclose(x.grad, expected, atol=1e-5)


# ===========================================================================
# 13. TestComparisons
# ===========================================================================


class TestComparisons:
    """Test comparison functions (eq, ne, lt, le, gt, ge) and Variable operators."""

    def test_eq(self):
        a = Variable([1.0, 2.0, 3.0])
        b = Variable([1.0, 0.0, 3.0])
        result = eq(a, b)
        np.testing.assert_allclose(result.data, [1.0, 0.0, 1.0])

    def test_ne(self):
        a = Variable([1.0, 2.0, 3.0])
        b = Variable([1.0, 0.0, 3.0])
        result = ne(a, b)
        np.testing.assert_allclose(result.data, [0.0, 1.0, 0.0])

    def test_lt_func(self):
        a = Variable([1.0, 2.0, 3.0])
        result = lt(a, 2.0)
        np.testing.assert_allclose(result.data, [1.0, 0.0, 0.0])

    def test_le_func(self):
        a = Variable([1.0, 2.0, 3.0])
        result = le(a, 2.0)
        np.testing.assert_allclose(result.data, [1.0, 1.0, 0.0])

    def test_gt_func(self):
        a = Variable([1.0, 2.0, 3.0])
        result = gt(a, 2.0)
        np.testing.assert_allclose(result.data, [0.0, 0.0, 1.0])

    def test_ge_func(self):
        a = Variable([1.0, 2.0, 3.0])
        result = ge(a, 2.0)
        np.testing.assert_allclose(result.data, [0.0, 1.0, 1.0])

    def test_lt_operator(self):
        a = Variable([1.0, 2.0, 3.0])
        # operator returns numpy bool array, not Variable
        result = a < 2.0
        np.testing.assert_array_equal(result, [True, False, False])

    def test_le_operator(self):
        a = Variable([1.0, 2.0, 3.0])
        result = a <= 2.0
        np.testing.assert_array_equal(result, [True, True, False])

    def test_gt_operator(self):
        a = Variable([1.0, 2.0, 3.0])
        result = a > 2.0
        np.testing.assert_array_equal(result, [False, False, True])

    def test_ge_operator(self):
        a = Variable([1.0, 2.0, 3.0])
        result = a >= 2.0
        np.testing.assert_array_equal(result, [False, True, True])

    def test_lt_operator_variable(self):
        a = Variable([1.0, 3.0])
        b = Variable([2.0, 2.0])
        result = a < b
        np.testing.assert_array_equal(result, [True, False])

    def test_comparisons_no_grad(self):
        """Comparison results should never require grad."""
        a = Variable([1.0, 2.0], requires_grad=True)
        result = eq(a, 1.0)
        assert result.requires_grad is False


# ===========================================================================
# 14. TestTrigonometric
# ===========================================================================


class TestTrigonometric:
    """Test sin, cos, tan, asin, acos, atan, atan2 forward + backward."""

    def test_sin_forward(self):
        x = Variable([0.0, np.pi / 2, np.pi])
        y = sin(x)
        np.testing.assert_allclose(y.data, [0.0, 1.0, 0.0], atol=1e-5)

    def test_cos_forward(self):
        x = Variable([0.0, np.pi / 2, np.pi])
        y = cos(x)
        np.testing.assert_allclose(y.data, [1.0, 0.0, -1.0], atol=1e-5)

    def test_tan_forward(self):
        x = Variable([0.0, np.pi / 4])
        y = tan(x)
        np.testing.assert_allclose(y.data, [0.0, 1.0], atol=1e-5)

    def test_asin_forward(self):
        x = Variable([0.0, 0.5, 1.0])
        y = asin(x)
        np.testing.assert_allclose(y.data, np.arcsin([0.0, 0.5, 1.0]).astype(np.float32), atol=1e-5)

    def test_acos_forward(self):
        x = Variable([0.0, 0.5, 1.0])
        y = acos(x)
        np.testing.assert_allclose(y.data, np.arccos([0.0, 0.5, 1.0]).astype(np.float32), atol=1e-5)

    def test_atan_forward(self):
        x = Variable([0.0, 1.0, -1.0])
        y = atan(x)
        np.testing.assert_allclose(y.data, np.arctan([0.0, 1.0, -1.0]).astype(np.float32), atol=1e-5)

    def test_atan2_forward(self):
        y_val = Variable([1.0, -1.0])
        x_val = Variable([1.0, 1.0])
        result = atan2(y_val, x_val)
        expected = np.arctan2([1.0, -1.0], [1.0, 1.0]).astype(np.float32)
        np.testing.assert_allclose(result.data, expected, atol=1e-5)

    def test_sin_backward(self):
        x = Variable(np.array([0.5, 1.0, 1.5], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(sin(v))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_cos_backward(self):
        x = Variable(np.array([0.5, 1.0, 1.5], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(cos(v))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_tan_backward(self):
        x = Variable(np.array([0.1, 0.2, 0.3], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(tan(v))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_asin_backward(self):
        x = Variable(np.array([0.1, 0.3, 0.5], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(asin(v))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_acos_backward(self):
        x = Variable(np.array([0.1, 0.3, 0.5], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(acos(v))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_atan_backward(self):
        x = Variable(np.array([0.5, 1.0, 2.0], dtype=np.float32), requires_grad=True)

        def fn(v):
            return sum(atan(v))

        expected = _numerical_grad(fn, x)
        loss = fn(x)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, expected, atol=1e-3)

    def test_atan2_backward(self):
        y_var = Variable(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
        x_var = Variable(np.array([3.0, 4.0], dtype=np.float32), requires_grad=True)

        def fn_y(v):
            return sum(atan2(v, Variable([3.0, 4.0])))

        def fn_x(v):
            return sum(atan2(Variable([1.0, 2.0]), v))

        expected_y = _numerical_grad(fn_y, y_var)
        expected_x = _numerical_grad(fn_x, x_var)

        result = atan2(y_var, x_var)
        loss = sum(result)
        loss.backward(use_gpu=False)

        np.testing.assert_allclose(y_var.grad, expected_y, atol=1e-3)
        np.testing.assert_allclose(x_var.grad, expected_x, atol=1e-3)

    def test_trig_methods(self):
        """Test that trig methods on Variable work."""
        x = Variable([0.5], requires_grad=True)
        assert sin(x).data[0] == pytest.approx(x.sin().data[0])
        assert cos(x).data[0] == pytest.approx(x.cos().data[0])
        assert tan(x).data[0] == pytest.approx(x.tan().data[0])
        assert asin(x).data[0] == pytest.approx(x.asin().data[0])
        assert acos(x).data[0] == pytest.approx(x.acos().data[0])
        assert atan(x).data[0] == pytest.approx(x.atan().data[0])


# ===========================================================================
# 15. TestLossesAutograd
# ===========================================================================


class TestLossesAutograd:
    """Test loss functions: cross_entropy, mse, l1, smooth_l1, bce, bce_with_logits,
    nll, kl_div. Each tests forward value and backward gradient."""

    def test_cross_entropy_forward_hard_targets(self):
        logits = Variable(np.array([[2.0, 1.0, 0.1]], dtype=np.float32))
        targets = np.array([0])
        loss = cross_entropy(logits, targets)
        # loss > 0
        assert loss.item() > 0
        # Check it is a scalar
        assert loss.shape == ()

    def test_cross_entropy_backward(self):
        logits = Variable(
            np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], dtype=np.float32),
            requires_grad=True,
        )
        targets = np.array([0, 1])

        def fn(v):
            return cross_entropy(v, targets)

        expected = _numerical_grad(fn, logits)
        loss = fn(logits)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(logits.grad, expected, atol=1e-3)

    def test_cross_entropy_soft_targets(self):
        logits = Variable(
            np.array([[2.0, 1.0, 0.1]], dtype=np.float32),
            requires_grad=True,
        )
        targets = np.array([[0.7, 0.2, 0.1]], dtype=np.float32)
        loss = cross_entropy(logits, targets)
        assert loss.item() > 0
        loss.backward(use_gpu=False)
        assert logits.grad is not None

    def test_mse_loss_forward(self):
        pred = Variable([1.0, 2.0, 3.0])
        target = Variable([1.5, 2.5, 3.5])
        loss = mse_loss(pred, target)
        expected = np.mean((np.array([1.0, 2.0, 3.0]) - np.array([1.5, 2.5, 3.5])) ** 2)
        assert loss.item() == pytest.approx(float(expected), abs=1e-5)

    def test_mse_loss_backward(self):
        pred = Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        Variable([1.5, 2.5, 3.5])

        def fn(v):
            return mse_loss(v, Variable([1.5, 2.5, 3.5]))

        expected = _numerical_grad(fn, pred)
        loss = fn(pred)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(pred.grad, expected, atol=1e-3)

    def test_l1_loss_mean(self):
        pred = Variable([1.0, 2.0, 3.0])
        target = Variable([1.5, 2.5, 3.5])
        loss = l1_loss(pred, target, reduction="mean")
        assert loss.item() == pytest.approx(0.5, abs=1e-5)

    def test_l1_loss_sum(self):
        pred = Variable([1.0, 2.0, 3.0])
        target = Variable([1.5, 2.5, 3.5])
        loss = l1_loss(pred, target, reduction="sum")
        assert loss.item() == pytest.approx(1.5, abs=1e-5)

    def test_l1_loss_backward(self):
        pred = Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        Variable([1.5, 2.5, 2.0])

        def fn(v):
            return l1_loss(v, Variable([1.5, 2.5, 2.0]), reduction="mean")

        expected = _numerical_grad(fn, pred)
        loss = fn(pred)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(pred.grad, expected, atol=1e-3)

    def test_smooth_l1_loss_forward(self):
        pred = Variable([1.0, 2.0, 3.0])
        target = Variable([1.3, 2.0, 5.0])
        loss = smooth_l1_loss(pred, target, beta=1.0, reduction="mean")
        assert loss.item() > 0

    def test_smooth_l1_loss_backward(self):
        pred = Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        tgt = [1.3, 2.0, 5.0]

        def fn(v):
            return smooth_l1_loss(v, Variable(tgt), beta=1.0, reduction="mean")

        expected = _numerical_grad(fn, pred)
        loss = fn(pred)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(pred.grad, expected, atol=1e-3)

    def test_bce_loss_forward(self):
        pred = Variable([0.8, 0.4, 0.1])
        target = Variable([1.0, 0.0, 0.0])
        loss = bce_loss(pred, target, reduction="mean")
        assert loss.item() > 0

    def test_bce_loss_backward(self):
        pred = Variable(np.array([0.8, 0.4, 0.1], dtype=np.float32), requires_grad=True)
        tgt = [1.0, 0.0, 0.0]

        def fn(v):
            return bce_loss(v, Variable(tgt), reduction="mean")

        expected = _numerical_grad(fn, pred)
        loss = fn(pred)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(pred.grad, expected, atol=1e-3)

    def test_bce_with_logits_loss_forward(self):
        logits = Variable([1.0, -1.0, 0.0])
        target = Variable([1.0, 0.0, 0.5])
        loss = bce_with_logits_loss(logits, target, reduction="mean")
        assert loss.item() > 0

    def test_bce_with_logits_loss_backward(self):
        logits = Variable(np.array([1.0, -1.0, 0.0], dtype=np.float32), requires_grad=True)
        tgt = [1.0, 0.0, 0.5]

        def fn(v):
            return bce_with_logits_loss(v, Variable(tgt), reduction="mean")

        expected = _numerical_grad(fn, logits)
        loss = fn(logits)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(logits.grad, expected, atol=1e-3)

    def test_nll_loss_forward(self):
        # log-probabilities
        log_probs_data = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32))
        log_probs = Variable(log_probs_data)
        target = np.array([0, 1])
        loss = nll_loss(log_probs, target, reduction="mean")
        expected = -np.mean([np.log(0.7), np.log(0.8)])
        assert loss.item() == pytest.approx(float(expected), abs=1e-4)

    def test_nll_loss_backward(self):
        log_probs = Variable(
            np.log(
                np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
            ),
            requires_grad=True,
        )
        target = np.array([0, 1])

        def fn(v):
            return nll_loss(v, target, reduction="mean")

        expected = _numerical_grad(fn, log_probs)
        loss = fn(log_probs)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(log_probs.grad, expected, atol=1e-3)

    def test_kl_div_loss_forward(self):
        log_probs = Variable(np.log(np.array([[0.5, 0.5]], dtype=np.float32)))
        target = Variable(np.array([[0.4, 0.6]], dtype=np.float32))
        loss = kl_div_loss(log_probs, target, reduction="mean")
        assert loss.item() > 0

    def test_kl_div_loss_backward(self):
        log_probs = Variable(
            np.log(np.array([[0.5, 0.5]], dtype=np.float32)),
            requires_grad=True,
        )
        target_data = np.array([[0.4, 0.6]], dtype=np.float32)

        def fn(v):
            return kl_div_loss(v, Variable(target_data), reduction="mean")

        expected = _numerical_grad(fn, log_probs)
        loss = fn(log_probs)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(log_probs.grad, expected, atol=1e-3)


# ===========================================================================
# 16. TestWhereOp
# ===========================================================================


class TestWhereOp:
    """Test the where(condition, x, y) operation."""

    def test_where_forward(self):
        cond = np.array([True, False, True])
        x = Variable([1.0, 2.0, 3.0])
        y = Variable([4.0, 5.0, 6.0])
        result = where(cond, x, y)
        np.testing.assert_allclose(result.data, [1.0, 5.0, 3.0])

    def test_where_with_variable_condition(self):
        cond = Variable([1.0, 0.0, 1.0])
        x = Variable([10.0, 20.0, 30.0])
        y = Variable([40.0, 50.0, 60.0])
        result = where(cond, x, y)
        np.testing.assert_allclose(result.data, [10.0, 50.0, 30.0])

    def test_where_backward(self):
        cond = np.array([True, False, True])
        x = Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        y = Variable(np.array([4.0, 5.0, 6.0], dtype=np.float32), requires_grad=True)
        result = where(cond, x, y)
        loss = sum(result)
        loss.backward(use_gpu=False)
        # x gets grad only where cond is True
        np.testing.assert_allclose(x.grad, [1.0, 0.0, 1.0], atol=1e-5)
        # y gets grad only where cond is False
        np.testing.assert_allclose(y.grad, [0.0, 1.0, 0.0], atol=1e-5)


# ===========================================================================
# 17. TestCustomFunction
# ===========================================================================


class TestCustomFunction:
    """Test creating a custom Function subclass."""

    def test_custom_relu(self):
        class MyReLU(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return np.maximum(x.data, 0)

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                return grad_output * (x.data > 0).astype(np.float32)

        x = Variable([1.0, -2.0, 3.0, -0.5], requires_grad=True)
        y = MyReLU.apply(x)
        np.testing.assert_allclose(y.data, [1.0, 0.0, 3.0, 0.0])

        loss = sum(y)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, [1.0, 0.0, 1.0, 0.0], atol=1e-5)

    def test_custom_square(self):
        class Square(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.data ** 2

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                return 2 * x.data * grad_output

        x = Variable([1.0, 2.0, 3.0], requires_grad=True)
        y = Square.apply(x)
        np.testing.assert_allclose(y.data, [1.0, 4.0, 9.0])

        loss = sum(y)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, [2.0, 4.0, 6.0], atol=1e-5)

    def test_custom_function_via_call(self):
        """Function subclasses can be invoked via MyFunc(args) due to FunctionMeta."""

        class Scale(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.data * 3.0

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * 3.0

        x = Variable([1.0, 2.0], requires_grad=True)
        y = Scale(x)
        np.testing.assert_allclose(y.data, [3.0, 6.0])

    def test_function_ctx_needs_input_grad(self):
        """Test that needs_input_grad returns correct booleans."""

        class Dummy(Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                ctx._test_needs_grad = ctx.needs_input_grad
                return x.data + y.data

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, grad_output

        x = Variable([1.0], requires_grad=True)
        y = Variable([2.0], requires_grad=False)
        result = Dummy.apply(x, y)
        # We cannot easily access ctx._test_needs_grad from here,
        # so just verify forward and backward work
        loss = sum(result)
        loss.backward(use_gpu=False)
        assert x.grad is not None

    def test_custom_function_no_grad_input(self):
        """If no input requires grad, the result should not either."""

        class AddOne(Function):
            @staticmethod
            def forward(ctx, x):
                return x.data + 1

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        x = Variable([1.0, 2.0], requires_grad=False)
        y = AddOne.apply(x)
        assert y.requires_grad is False
        assert y.grad_fn is None


# ===========================================================================
# 18. TestBackwardIntegration
# ===========================================================================


class TestBackwardIntegration:
    """Integration tests: chained ops, gradient accumulation, no_grad early exit."""

    def test_chained_ops_mul_add_relu_sum(self):
        """Test backward through: sum(relu(a * x + b))."""
        x = Variable(np.array([1.0, -2.0, 3.0], dtype=np.float32), requires_grad=True)
        a = Variable(np.array([2.0, 2.0, 2.0], dtype=np.float32), requires_grad=True)
        b = Variable(np.array([0.5, 0.5, 0.5], dtype=np.float32), requires_grad=True)

        y = mul(a, x)         # [2, -4, 6]
        z = add(y, b)         # [2.5, -3.5, 6.5]
        r = relu(z)           # [2.5, 0, 6.5]
        loss = sum(r)         # 9.0
        loss.backward(use_gpu=False)

        # relu mask: [True, False, True] => [1, 0, 1]
        # d(sum(relu(a*x + b)))/dx = a * relu_mask = [2, 0, 2]
        np.testing.assert_allclose(x.grad, [2.0, 0.0, 2.0], atol=1e-5)
        # d/da = x * relu_mask = [1, 0, 3]
        np.testing.assert_allclose(a.grad, [1.0, 0.0, 3.0], atol=1e-5)
        # d/db = relu_mask = [1, 0, 1]
        np.testing.assert_allclose(b.grad, [1.0, 0.0, 1.0], atol=1e-5)

    def test_gradient_accumulation(self):
        """Gradients should accumulate over multiple backward calls."""
        x = Variable(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)

        # First backward
        y1 = sum(x * 2)
        y1.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, [2.0, 2.0], atol=1e-5)

        # Second backward (do NOT zero_grad) => accumulate
        y2 = sum(x * 3)
        y2.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, [5.0, 5.0], atol=1e-5)

    def test_gradient_accumulation_with_zero_grad(self):
        """zero_grad should clear gradients before next backward."""
        x = Variable(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)

        y1 = sum(x * 2)
        y1.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, [2.0, 2.0], atol=1e-5)

        x.zero_grad()

        y2 = sum(x * 3)
        y2.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, [3.0, 3.0], atol=1e-5)

    def test_no_grad_early_exit(self):
        """Variable with requires_grad=False should not get gradients."""
        x = Variable(np.array([1.0, 2.0], dtype=np.float32), requires_grad=False)
        y = sum(x * 2)
        y.backward(use_gpu=False)
        assert x.grad is None

    def test_backward_no_requires_grad_returns(self):
        """backward on a Variable with requires_grad=False does nothing."""
        x = Variable([1.0], requires_grad=False)
        x.backward(use_gpu=False)  # should not raise
        assert x.grad is None

    def test_complex_chain_numerical(self):
        """Test a complex chain: sigmoid(matmul(x, w) + b).sum() with numerical check."""
        x = Variable(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
        w = Variable(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        b = Variable(np.random.randn(4).astype(np.float32), requires_grad=True)

        def fn_x(v):
            return sum(sigmoid(add(matmul(v, Variable(w.data)), Variable(b.data))))

        def fn_w(v):
            return sum(sigmoid(add(matmul(Variable(x.data), v), Variable(b.data))))

        def fn_b(v):
            return sum(sigmoid(add(matmul(Variable(x.data), Variable(w.data)), v)))

        # Analytical backward
        h = matmul(x, w)
        z = add(h, b)
        s = sigmoid(z)
        loss = sum(s)
        loss.backward(use_gpu=False)

        # Numerical backward
        expected_x = _numerical_grad(fn_x, x)
        expected_w = _numerical_grad(fn_w, w)
        expected_b = _numerical_grad(fn_b, b)

        np.testing.assert_allclose(x.grad, expected_x, atol=1e-3)
        np.testing.assert_allclose(w.grad, expected_w, atol=1e-3)
        np.testing.assert_allclose(b.grad, expected_b, atol=1e-3)

    def test_diamond_graph(self):
        """Test backward through a diamond-shaped computation graph.
        x -> y1 -> z
        x -> y2 -> z
        Both paths should contribute gradients to x.
        """
        x = Variable(np.array([2.0, 3.0], dtype=np.float32), requires_grad=True)
        y1 = x * 2    # [4, 6]
        y2 = x * 3    # [6, 9]
        z = y1 + y2   # [10, 15]
        loss = sum(z)  # 25
        loss.backward(use_gpu=False)
        # d/dx = 2 + 3 = 5
        np.testing.assert_allclose(x.grad, [5.0, 5.0], atol=1e-5)

    def test_multi_output_use(self):
        """Test that using a variable in multiple places accumulates gradients."""
        x = Variable(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
        y = x + x + x  # equivalent to 3*x
        loss = sum(y)
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, [3.0, 3.0], atol=1e-5)

    def test_long_chain(self):
        """Backward through a long chain of additions."""
        x = Variable(np.array([1.0], dtype=np.float32), requires_grad=True)
        y = x
        for _ in range(20):
            y = y + x
        loss = sum(y)
        loss.backward(use_gpu=False)
        # y = x + x*20 iterations = 21*x
        assert x.grad[0] == pytest.approx(21.0, abs=1e-4)

    def test_detach_breaks_graph(self):
        """detach() should break the computation graph."""
        x = Variable(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
        y = x * 2
        z = y.detach()  # break graph
        w = z + 1       # z has no grad, so w has no grad_fn from x
        loss = sum(w)
        loss.backward(use_gpu=False)
        # x should get no gradient because detach broke the chain
        assert x.grad is None

    def test_method_chaining(self):
        """Test method-style calls: x.relu().sum().backward()."""
        x = Variable(np.array([-1.0, 0.5, 2.0], dtype=np.float32), requires_grad=True)
        loss = x.relu().sum()
        loss.backward(use_gpu=False)
        np.testing.assert_allclose(x.grad, [0.0, 1.0, 1.0], atol=1e-5)

    def test_scalar_loss_backward(self):
        """backward() on a scalar loss should work without explicit grad_output."""
        x = Variable(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        loss = mean(x ** 2)
        loss.backward(use_gpu=False)
        # d/dx mean(x^2) = 2x / n
        expected = 2 * x.data / 3
        np.testing.assert_allclose(x.grad, expected, atol=1e-5)
