"""
Tests for C++ optimizers: Adam, AdamW, SGD.
Mirrors test_adamw.py and test_optim_sgd_nlms.py but exercises
the C++ implementations with batched GPU dispatch.
"""

import numpy as np
import pytest

try:
    from grilly_core import SGD, Adam, AdamW, Parameter, Tensor

    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ backend not available")


# ═══════════════════════════════════════════════════════════════════════
# Adam
# ═══════════════════════════════════════════════════════════════════════


class TestAdamBasic:
    """Basic Adam optimizer tests."""

    def test_adam_init(self):
        """Adam should initialize with default hyperparameters."""
        p = Parameter(Tensor.from_numpy(np.ones(5, dtype=np.float32)))
        opt = Adam([p], lr=0.001)
        assert opt.param_count == 1
        assert opt.lr == pytest.approx(0.001)

    def test_adam_step_with_gradient(self):
        """Adam.step() should update parameters when gradient exists."""
        p = Parameter(Tensor.from_numpy(np.ones((3, 3), dtype=np.float32)))
        # Manually set gradient
        g = p.grad()
        g.numpy()
        # Fill gradient with 0.1
        new_grad = Tensor.from_numpy(np.ones((3, 3), dtype=np.float32) * 0.1)
        p.set_grad(new_grad)

        opt = Adam([p], lr=0.01)
        param_before = p.numpy().copy()
        opt.step()
        param_after = p.numpy()

        # Parameters should have changed
        assert not np.allclose(param_before, param_after)
        # With positive gradient, parameters should decrease
        assert np.all(param_after < param_before)

    def test_adam_zero_grad(self):
        """zero_grad should zero all parameter gradients."""
        p = Parameter(Tensor.from_numpy(np.ones(5, dtype=np.float32)))
        p.set_grad(Tensor.from_numpy(np.ones(5, dtype=np.float32)))

        opt = Adam([p])
        opt.zero_grad()
        np.testing.assert_array_equal(p.grad().numpy(), np.zeros(5, dtype=np.float32))

    def test_adam_multiple_params(self):
        """Adam should handle multiple parameters."""
        p1 = Parameter(Tensor.from_numpy(np.ones((3, 3), dtype=np.float32)))
        p2 = Parameter(Tensor.from_numpy(np.ones((5,), dtype=np.float32)))

        p1.set_grad(Tensor.from_numpy(np.ones((3, 3), dtype=np.float32) * 0.1))
        p2.set_grad(Tensor.from_numpy(np.ones(5, dtype=np.float32) * 0.2))

        opt = Adam([p1, p2], lr=0.01)
        assert opt.param_count == 2

        p1_before = p1.numpy().copy()
        p2_before = p2.numpy().copy()
        opt.step()

        assert not np.allclose(p1.numpy(), p1_before)
        assert not np.allclose(p2.numpy(), p2_before)

    def test_adam_set_lr(self):
        """Learning rate should be adjustable."""
        p = Parameter(Tensor.from_numpy(np.ones(5, dtype=np.float32)))
        opt = Adam([p], lr=0.001)
        assert opt.lr == pytest.approx(0.001)
        opt.lr = 0.01
        assert opt.lr == pytest.approx(0.01)

    def test_adam_multiple_steps(self):
        """Adam should converge over multiple steps."""
        p = Parameter(Tensor.from_numpy(np.ones(5, dtype=np.float32) * 5.0))
        opt = Adam([p], lr=0.1)

        for _ in range(10):
            # Gradient pointing toward zero
            p.set_grad(Tensor.from_numpy(p.numpy().copy() * 0.1))
            opt.step()

        # Parameters should have decreased
        assert np.all(p.numpy() < 5.0)

    def test_adam_no_nan_inf(self):
        """Adam should not produce NaN or Inf."""
        p = Parameter(Tensor.from_numpy(np.ones(5, dtype=np.float32)))
        p.set_grad(Tensor.from_numpy(np.ones(5, dtype=np.float32) * 1000.0))

        opt = Adam([p], lr=0.001)
        opt.step()

        assert not np.any(np.isnan(p.numpy()))
        assert not np.any(np.isinf(p.numpy()))

    def test_adam_zero_gradient_no_crash(self):
        """Adam with zero gradient should not crash."""
        p = Parameter(Tensor.from_numpy(np.ones(5, dtype=np.float32)))
        p.set_grad(Tensor.from_numpy(np.zeros(5, dtype=np.float32)))

        opt = Adam([p], lr=0.01)
        opt.step()  # Should not crash
        assert not np.any(np.isnan(p.numpy()))


# ═══════════════════════════════════════════════════════════════════════
# AdamW
# ═══════════════════════════════════════════════════════════════════════


class TestAdamWBasic:
    """Basic AdamW tests (decoupled weight decay)."""

    def test_adamw_init(self):
        """AdamW should initialize correctly."""
        p = Parameter(Tensor.from_numpy(np.ones(5, dtype=np.float32)))
        opt = AdamW([p], lr=0.001, weight_decay=0.01)
        assert opt.param_count == 1
        assert opt.lr == pytest.approx(0.001)

    def test_adamw_step(self):
        """AdamW should perform parameter update."""
        p = Parameter(Tensor.from_numpy(np.ones((5, 5), dtype=np.float32)))
        p.set_grad(Tensor.from_numpy(np.ones((5, 5), dtype=np.float32) * 0.1))

        opt = AdamW([p], lr=0.01, weight_decay=0.0)
        before = p.numpy().copy()
        opt.step()
        after = p.numpy()

        assert not np.allclose(before, after)
        assert np.all(after < before)

    def test_adamw_weight_decay_shrinks_params(self):
        """With weight decay and zero grad, params should shrink."""
        p = Parameter(Tensor.from_numpy(np.ones((5, 5), dtype=np.float32) * 2.0))
        p.set_grad(Tensor.from_numpy(np.zeros((5, 5), dtype=np.float32)))

        opt = AdamW([p], lr=0.01, weight_decay=0.1)
        before = p.numpy().copy()
        opt.step()
        after = p.numpy()

        # Decoupled weight decay: p *= (1 - lr * wd) = (1 - 0.001) = 0.999
        assert np.all(np.abs(after) < np.abs(before))

    def test_adamw_no_nan(self):
        """AdamW should not produce NaN."""
        p = Parameter(Tensor.from_numpy(np.ones(10, dtype=np.float32)))
        p.set_grad(Tensor.from_numpy(np.ones(10, dtype=np.float32) * 100.0))

        opt = AdamW([p], lr=0.001, weight_decay=0.01)
        opt.step()
        assert not np.any(np.isnan(p.numpy()))


# ═══════════════════════════════════════════════════════════════════════
# SGD
# ═══════════════════════════════════════════════════════════════════════


class TestSGDBasic:
    """Basic SGD tests."""

    def test_sgd_init(self):
        """SGD should initialize correctly."""
        p = Parameter(Tensor.from_numpy(np.ones(5, dtype=np.float32)))
        opt = SGD([p], lr=0.01)
        assert opt.param_count == 1
        assert opt.lr == pytest.approx(0.01)

    def test_sgd_step(self):
        """SGD should update parameters: p -= lr * grad."""
        p = Parameter(Tensor.from_numpy(np.ones(5, dtype=np.float32)))
        grad = np.ones(5, dtype=np.float32) * 0.5
        p.set_grad(Tensor.from_numpy(grad))

        opt = SGD([p], lr=0.1)
        opt.step()

        # p = 1.0 - 0.1 * 0.5 = 0.95
        np.testing.assert_allclose(p.numpy(), 0.95, atol=1e-6)

    def test_sgd_with_weight_decay(self):
        """SGD with weight decay should shrink parameters."""
        p = Parameter(Tensor.from_numpy(np.ones(5, dtype=np.float32) * 2.0))
        p.set_grad(Tensor.from_numpy(np.zeros(5, dtype=np.float32)))

        opt = SGD([p], lr=0.01, weight_decay=0.1)
        before = p.numpy().copy()
        opt.step()

        # grad_effective = 0 + 0.1 * 2.0 = 0.2
        # p = 2.0 - 0.01 * 0.2 = 1.998
        assert np.all(p.numpy() < before)

    def test_sgd_set_lr(self):
        """SGD learning rate should be adjustable."""
        p = Parameter(Tensor.from_numpy(np.ones(5, dtype=np.float32)))
        opt = SGD([p], lr=0.01)
        opt.lr = 0.1
        assert opt.lr == pytest.approx(0.1)

    def test_sgd_multiple_params(self):
        """SGD should handle multiple parameters."""
        p1 = Parameter(Tensor.from_numpy(np.ones(3, dtype=np.float32)))
        p2 = Parameter(Tensor.from_numpy(np.ones(5, dtype=np.float32) * 2.0))
        p1.set_grad(Tensor.from_numpy(np.ones(3, dtype=np.float32)))
        p2.set_grad(Tensor.from_numpy(np.ones(5, dtype=np.float32)))

        opt = SGD([p1, p2], lr=0.1)
        opt.step()

        np.testing.assert_allclose(p1.numpy(), 0.9, atol=1e-6)
        np.testing.assert_allclose(p2.numpy(), 1.9, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Optimizer Integration
# ═══════════════════════════════════════════════════════════════════════


class TestOptimizerIntegration:
    """Integration tests: optimizer + module parameters."""

    def test_adam_with_module_parameters(self):
        """Adam should work with parameters from a Module."""
        from grilly_core import LIFNode, Module

        class Net(Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "weight", Parameter(Tensor.from_numpy(np.ones((4, 8), dtype=np.float32)))
                )

            def forward(self, x):
                return x

        net = Net()
        params = net.parameters()
        opt = Adam(params, lr=0.01)
        assert opt.param_count == 1

    def test_optimizer_with_parametric_lif(self):
        """Optimizer should update ParametricLIFNode's tau parameter."""
        from grilly_core import ParametricLIFNode

        node = ParametricLIFNode(init_tau=2.0)
        params = node.parameters()
        opt = Adam(params, lr=0.01)

        # Set gradient on tau
        for p in params:
            p.set_grad(Tensor.from_numpy(np.array([0.1], dtype=np.float32)))

        tau_before = params[0].numpy().copy()
        opt.step()
        tau_after = params[0].numpy()

        assert not np.allclose(tau_before, tau_after), "Tau should have been updated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
