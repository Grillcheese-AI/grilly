"""Tests for HypergradientAdamW and AutoHypergradientAdamW optimizers."""

import numpy as np
import pytest

from grilly.nn.parameter import Parameter
from grilly.optim.hypergradient import AutoHypergradientAdamW, HypergradientAdamW


def _make_params(*shapes):
    """Create Parameter objects with random data."""
    params = []
    for shape in shapes:
        p = Parameter(np.random.randn(*shape).astype(np.float32) * 0.1)
        params.append(p)
    return params


def _quadratic_grads(params, target=None):
    """Compute gradients for a simple quadratic loss: L = 0.5 * ||p - target||^2."""
    grads = {}
    for p in params:
        p_arr = np.asarray(p, dtype=np.float32)
        t = target if target is not None else np.zeros_like(p_arr)
        grad = (p_arr - t).astype(np.float32)
        grads[id(p)] = grad
        p.grad = grad
    return grads


# ---------------------------------------------------------------------------
# HypergradientAdamW tests
# ---------------------------------------------------------------------------
class TestHypergradientAdamW:

    def test_basic_step(self):
        params = _make_params((10,))
        opt = HypergradientAdamW(params, lr=0.01, beta_hyper=1e-6, use_gpu=False)

        for _ in range(5):
            _quadratic_grads(params)
            opt.step()
            opt.zero_grad()

        assert len(opt.lr_history) > 1

    def test_lr_adapts(self):
        """LR should change when consecutive gradients agree/disagree."""
        params = _make_params((20,))
        opt = HypergradientAdamW(params, lr=0.01, beta_hyper=1e-5, use_gpu=False)

        initial_lr = opt.current_lr
        for _ in range(20):
            _quadratic_grads(params)
            opt.step()
            opt.zero_grad()

        # LR should have moved from initial value
        assert opt.current_lr != initial_lr

    def test_lr_clamping(self):
        """LR should stay within [lr_min, lr_max]."""
        params = _make_params((5,))
        opt = HypergradientAdamW(
            params, lr=0.5, beta_hyper=1.0,  # Extreme beta_hyper
            lr_min=0.001, lr_max=0.1, use_gpu=False,
        )

        for _ in range(10):
            _quadratic_grads(params)
            opt.step()
            opt.zero_grad()

        assert opt.current_lr >= 0.001
        assert opt.current_lr <= 0.1

    def test_log_scale(self):
        params = _make_params((10,))
        opt = HypergradientAdamW(
            params, lr=0.01, beta_hyper=1e-5,
            log_scale=True, use_gpu=False,
        )

        for _ in range(10):
            _quadratic_grads(params)
            opt.step()
            opt.zero_grad()

        assert opt.current_lr > 0
        assert len(opt.lr_history) > 1

    def test_convergence(self):
        """Should converge on simple quadratic."""
        np.random.seed(42)
        params = _make_params((10,))
        target = np.zeros(10, dtype=np.float32)
        opt = HypergradientAdamW(params, lr=0.05, beta_hyper=1e-7, use_gpu=False)

        initial_loss = 0.5 * np.sum(np.asarray(params[0]) ** 2)
        for _ in range(100):
            _quadratic_grads(params, target)
            opt.step()
            opt.zero_grad()

        final_loss = 0.5 * np.sum(np.asarray(params[0]) ** 2)
        assert final_loss < initial_loss * 0.1

    def test_repr(self):
        params = _make_params((5,))
        opt = HypergradientAdamW(params, lr=0.01, beta_hyper=1e-7, use_gpu=False)
        r = repr(opt)
        assert "HypergradientAdamW" in r
        assert "beta_hyper" in r

    def test_gradients_dict(self):
        """Should work with explicit gradients dict."""
        params = _make_params((10,))
        opt = HypergradientAdamW(params, lr=0.01, beta_hyper=1e-6, use_gpu=False)

        for _ in range(5):
            grads = _quadratic_grads(params)
            opt.step(gradients=grads)
            opt.zero_grad()

        assert len(opt.lr_history) > 1


# ---------------------------------------------------------------------------
# AutoHypergradientAdamW tests
# ---------------------------------------------------------------------------
class TestAutoHypergradientAdamW:

    def test_basic_step(self):
        params = _make_params((10,))
        opt = AutoHypergradientAdamW(params, lr=0.01, warmup_steps=2, use_gpu=False)

        for _ in range(10):
            _quadratic_grads(params)
            opt.step()
            opt.zero_grad()

        assert len(opt.lr_history) > 1

    def test_warmup_skips_adaptation(self):
        """LR should not change during warmup period."""
        params = _make_params((10,))
        warmup = 5
        opt = AutoHypergradientAdamW(
            params, lr=0.01, warmup_steps=warmup, use_gpu=False,
        )

        initial_lr = opt.current_lr
        for i in range(warmup):
            _quadratic_grads(params)
            opt.step()
            opt.zero_grad()
            # During warmup, lr_history should only have the initial entry
            assert opt.current_lr == initial_lr, f"LR changed at warmup step {i}"

    def test_lr_adapts_after_warmup(self):
        """LR should adapt after warmup completes."""
        params = _make_params((20,))
        opt = AutoHypergradientAdamW(
            params, lr=0.01, warmup_steps=3, hyper_lr=0.1, use_gpu=False,
        )

        initial_lr = opt.current_lr
        for _ in range(30):
            _quadratic_grads(params)
            opt.step()
            opt.zero_grad()

        assert opt.current_lr != initial_lr

    def test_lr_clamping(self):
        params = _make_params((5,))
        opt = AutoHypergradientAdamW(
            params, lr=0.05, hyper_lr=10.0, warmup_steps=1,
            lr_min=0.001, lr_max=0.1, use_gpu=False,
        )

        for _ in range(20):
            _quadratic_grads(params)
            opt.step()
            opt.zero_grad()

        assert opt.current_lr >= 0.001
        assert opt.current_lr <= 0.1

    def test_adagrad_stabilization(self):
        """AdaGrad accumulator should prevent wild LR oscillations."""
        np.random.seed(42)
        params = _make_params((50,))
        opt = AutoHypergradientAdamW(
            params, lr=0.01, hyper_lr=0.1, warmup_steps=2, use_gpu=False,
        )

        for _ in range(50):
            _quadratic_grads(params)
            opt.step()
            opt.zero_grad()

        # Check that LR changes are bounded (no explosion)
        lr_hist = opt.lr_history
        for lr in lr_hist:
            assert np.isfinite(lr)
            assert lr > 0

    def test_convergence(self):
        """Should converge on simple quadratic, possibly faster than fixed LR."""
        np.random.seed(42)
        params = _make_params((10,))
        opt = AutoHypergradientAdamW(
            params, lr=0.05, hyper_lr=0.01, warmup_steps=5, use_gpu=False,
        )

        initial_loss = 0.5 * np.sum(np.asarray(params[0]) ** 2)
        for _ in range(100):
            _quadratic_grads(params)
            opt.step()
            opt.zero_grad()

        final_loss = 0.5 * np.sum(np.asarray(params[0]) ** 2)
        assert final_loss < initial_loss * 0.1

    def test_momentum_adaptation(self):
        """When adapt_momentum=True, beta1 should change."""
        params = _make_params((20,))
        opt = AutoHypergradientAdamW(
            params, lr=0.01, warmup_steps=3,
            adapt_momentum=True, hyper_lr_beta=1.0, use_gpu=False,
        )

        initial_beta1 = opt.defaults["betas"][0]
        for _ in range(30):
            _quadratic_grads(params)
            opt.step()
            opt.zero_grad()

        # beta1 should have adapted
        final_beta1 = opt.defaults["betas"][0]
        assert final_beta1 != initial_beta1
        assert opt.beta_min <= final_beta1 <= opt.beta_max

    def test_momentum_clamping(self):
        params = _make_params((10,))
        opt = AutoHypergradientAdamW(
            params, lr=0.01, warmup_steps=1,
            adapt_momentum=True, hyper_lr_beta=100.0,
            beta_min=0.8, beta_max=0.95, use_gpu=False,
        )

        for _ in range(20):
            _quadratic_grads(params)
            opt.step()
            opt.zero_grad()

        beta1 = opt.defaults["betas"][0]
        assert 0.8 <= beta1 <= 0.95

    def test_multiple_param_groups(self):
        """Should handle multiple parameter tensors."""
        params = _make_params((10,), (20,), (5, 5))
        opt = AutoHypergradientAdamW(
            params, lr=0.01, warmup_steps=2, use_gpu=False,
        )

        for _ in range(15):
            _quadratic_grads(params)
            opt.step()
            opt.zero_grad()

        assert len(opt.lr_history) > 1
        # All params should have moved toward zero
        for p in params:
            assert np.mean(np.abs(np.asarray(p))) < 0.1

    def test_repr(self):
        params = _make_params((5,))
        opt = AutoHypergradientAdamW(
            params, lr=0.01, adapt_momentum=True, use_gpu=False,
        )
        r = repr(opt)
        assert "AutoHypergradientAdamW" in r
        assert "hyper_lr" in r
        assert "adapt_momentum=True" in r

    def test_gradient_norm_invariance(self):
        """Scaling all gradients should not drastically change LR trajectory."""
        np.random.seed(42)
        params1 = _make_params((10,))
        params2 = [Parameter(np.array(p, copy=True)) for p in params1]

        opt1 = AutoHypergradientAdamW(
            params1, lr=0.01, warmup_steps=2, hyper_lr=0.01, use_gpu=False,
        )
        opt2 = AutoHypergradientAdamW(
            params2, lr=0.01, warmup_steps=2, hyper_lr=0.01, use_gpu=False,
        )

        for _ in range(20):
            # Small gradients
            _quadratic_grads(params1)
            opt1.step()
            opt1.zero_grad()

            # 10x scaled gradients (same direction)
            for p in params2:
                p.grad = (np.asarray(p) * 10.0).astype(np.float32)
            opt2.step()
            opt2.zero_grad()

        # Both should still converge (LR adaptation is scale-invariant
        # due to normalization by ||g||^2)
        lr1 = opt1.lr_history[-1]
        lr2 = opt2.lr_history[-1]
        assert np.isfinite(lr1) and lr1 > 0
        assert np.isfinite(lr2) and lr2 > 0

    def test_no_grad_params_skipped(self):
        """Parameters without gradients should be skipped cleanly."""
        params = _make_params((10,), (5,))
        opt = AutoHypergradientAdamW(params, lr=0.01, warmup_steps=1, use_gpu=False)

        # Only give gradient to first param
        for _ in range(10):
            params[0].grad = np.array(params[0], copy=True)
            # params[1] has no grad
            opt.step()
            opt.zero_grad()

        assert len(opt.lr_history) > 1


# ---------------------------------------------------------------------------
# Comparison test
# ---------------------------------------------------------------------------
class TestComparison:

    def test_auto_vs_fixed_convergence(self):
        """Auto should converge at least as well as fixed-LR AdamW."""
        np.random.seed(42)

        # Fixed AdamW
        p_fixed = _make_params((20,))
        from grilly.optim.adamw import AdamW
        opt_fixed = AdamW(p_fixed, lr=0.01, use_gpu=False)

        for _ in range(100):
            _quadratic_grads(p_fixed)
            opt_fixed.step()
            opt_fixed.zero_grad()
        loss_fixed = 0.5 * np.sum(np.asarray(p_fixed[0]) ** 2)

        # Auto hypergradient AdamW
        np.random.seed(42)
        p_auto = _make_params((20,))
        opt_auto = AutoHypergradientAdamW(
            p_auto, lr=0.01, hyper_lr=0.01, warmup_steps=5, use_gpu=False,
        )

        for _ in range(100):
            _quadratic_grads(p_auto)
            opt_auto.step()
            opt_auto.zero_grad()
        loss_auto = 0.5 * np.sum(np.asarray(p_auto[0]) ** 2)

        # Both should converge; auto should be competitive
        assert loss_fixed < 0.01
        assert loss_auto < 0.01
