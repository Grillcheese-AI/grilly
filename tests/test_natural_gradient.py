"""Tests for NaturalGradient optimizer (CPU fallback path)."""

import numpy as np
import pytest

try:
    from grilly.nn.parameter import Parameter
    from grilly.optim.natural_gradient import NaturalGradient
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


@pytest.fixture(autouse=True)
def seed_rng():
    """Seed numpy RNG for reproducibility."""
    np.random.seed(42)


def _make_param(shape, grad_shape=None):
    """Create a Parameter with random data and gradient."""
    p = Parameter(np.random.randn(*shape).astype(np.float32))
    if grad_shape is not None:
        p.grad = np.random.randn(*grad_shape).astype(np.float32)
    else:
        p.grad = np.random.randn(*shape).astype(np.float32)
    return p


class TestNaturalGradientInit:
    """Tests for NaturalGradient optimizer initialization."""

    def test_init_basic(self):
        """Creates optimizer with params and custom lr."""
        param = Parameter(np.random.randn(10, 5).astype(np.float32))
        opt = NaturalGradient([param], lr=0.05, use_gpu=False)
        assert opt.defaults["lr"] == 0.05
        assert len(opt.param_groups) == 1
        assert len(opt.param_groups[0]["params"]) == 1

    def test_init_defaults(self):
        """Default lr=1e-3 and fisher_momentum=0.9."""
        param = Parameter(np.random.randn(4, 4).astype(np.float32))
        opt = NaturalGradient([param], use_gpu=False)
        assert opt.defaults["lr"] == 1e-3
        assert opt.defaults["fisher_momentum"] == 0.9

    def test_init_empty_params(self):
        """Empty parameter list raises ValueError from base Optimizer."""
        with pytest.raises(ValueError, match="empty parameter list"):
            NaturalGradient([], use_gpu=False)


class TestNaturalGradientStep:
    """Tests for the NaturalGradient.step() CPU fallback path."""

    def test_step_initializes_fisher(self):
        """After the first step, Fisher dict has an entry for each param."""
        p1 = _make_param((6, 3))
        p2 = _make_param((4,))
        opt = NaturalGradient([p1, p2], lr=0.01, use_gpu=False)

        opt.step()

        for p in [p1, p2]:
            state = opt.state[id(p)]
            assert "fisher" in state
            assert state["fisher"].shape == p.shape

    def test_step_updates_params(self):
        """Parameters change after a step."""
        param = Parameter(np.random.randn(10, 5).astype(np.float32))
        param.grad = np.random.randn(10, 5).astype(np.float32)
        opt = NaturalGradient([param], lr=0.01, use_gpu=False)

        original = np.array(param).copy()
        opt.step()

        assert not np.allclose(param, original), "Parameters should change after step"

    def test_step_fisher_update(self):
        """Verify Fisher = momentum * old_fisher + (1 - momentum) * grad^2."""
        param = Parameter(np.ones((3, 3), dtype=np.float32))
        grad = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
        )
        param.grad = grad.copy()

        momentum = 0.8
        opt = NaturalGradient([param], lr=0.001, fisher_momentum=momentum, use_gpu=False)

        # Before step, fisher is initialized to ones * 1e-6
        old_fisher = np.ones_like(param, dtype=np.float32) * 1e-6

        opt.step()

        expected_fisher = momentum * old_fisher + (1.0 - momentum) * (grad ** 2)
        actual_fisher = opt.state[id(param)]["fisher"]

        np.testing.assert_allclose(actual_fisher, expected_fisher, rtol=1e-5)

    def test_step_natural_gradient_direction(self):
        """Verify update is approximately lr * grad / (fisher + eps)."""
        param = Parameter(np.ones((4,), dtype=np.float32) * 5.0)
        grad = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        param.grad = grad.copy()

        lr = 0.01
        momentum = 0.9
        eps = 1e-8
        opt = NaturalGradient([param], lr=lr, fisher_momentum=momentum, use_gpu=False)

        original = np.array(param).copy()
        opt.step()

        # Compute expected update manually
        old_fisher = np.ones(4, dtype=np.float32) * 1e-6
        fisher = momentum * old_fisher + (1.0 - momentum) * (grad ** 2)
        natural_grad = grad / (fisher + eps)
        expected = original - lr * natural_grad

        np.testing.assert_allclose(
            np.asarray(param, dtype=np.float32),
            expected,
            rtol=1e-5,
            err_msg="Parameter update should match manual natural gradient computation",
        )

    def test_step_clears_grad(self):
        """After step, param.grad is zeroed (via zero_grad) or set to None."""
        param = _make_param((5, 5))
        opt = NaturalGradient([param], lr=0.01, use_gpu=False)

        opt.step()

        # Parameter.zero_grad() fills with zeros; otherwise grad is set to None
        if param.grad is not None:
            assert np.all(param.grad == 0), "Gradient should be zeroed after step"

    def test_step_skips_none_grad(self):
        """Param with grad=None is skipped without error."""
        p1 = Parameter(np.random.randn(3, 3).astype(np.float32))
        p1.grad = None  # No gradient

        p2 = _make_param((3, 3))
        original_p1 = np.array(p1).copy()

        opt = NaturalGradient([p1, p2], lr=0.01, use_gpu=False)
        opt.step()  # Should not raise

        # p1 should be unchanged (skipped)
        np.testing.assert_array_equal(
            np.asarray(p1, dtype=np.float32),
            original_p1,
            err_msg="Parameter with None grad should be unchanged",
        )

    def test_multiple_steps(self):
        """Multiple steps converge the Fisher estimate toward grad^2."""
        param = Parameter(np.ones((4,), dtype=np.float32))
        constant_grad = np.array([2.0, 3.0, 1.0, 4.0], dtype=np.float32)

        momentum = 0.9
        opt = NaturalGradient([param], lr=1e-4, fisher_momentum=momentum, use_gpu=False)

        # Run many steps with constant gradient to let Fisher converge
        for _ in range(200):
            param.grad = constant_grad.copy()
            opt.step()

        # With constant gradients, Fisher should converge to grad^2
        # fisher_inf = (1 - m) * g^2 / (1 - m) = g^2 (geometric series limit)
        fisher = opt.state[id(param)]["fisher"]
        np.testing.assert_allclose(
            fisher,
            constant_grad ** 2,
            rtol=0.05,
            err_msg="Fisher should converge to grad^2 with constant gradients",
        )

    def test_learning_rate(self):
        """Larger lr produces a bigger parameter update."""
        grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Small lr
        p_small = Parameter(np.ones((3,), dtype=np.float32) * 10.0)
        p_small.grad = grad.copy()
        opt_small = NaturalGradient([p_small], lr=0.001, use_gpu=False)
        original_small = np.array(p_small).copy()
        opt_small.step()
        delta_small = np.abs(np.asarray(p_small, dtype=np.float32) - original_small)

        # Large lr
        p_large = Parameter(np.ones((3,), dtype=np.float32) * 10.0)
        p_large.grad = grad.copy()
        opt_large = NaturalGradient([p_large], lr=0.1, use_gpu=False)
        original_large = np.array(p_large).copy()
        opt_large.step()
        delta_large = np.abs(np.asarray(p_large, dtype=np.float32) - original_large)

        assert np.all(delta_large > delta_small), (
            "Larger learning rate should produce bigger parameter updates"
        )
