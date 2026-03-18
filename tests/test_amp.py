"""
Tests for Automatic Mixed Precision (AMP) framework.

Covers:
- autocast context manager flag management
- Nested autocast
- autocast as decorator
- GradScaler scale / unscale
- GradScaler step (clean and inf/NaN paths)
- GradScaler update (growth / backoff)
- GradScaler disabled mode
- GradScaler state_dict / load_state_dict
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyParam:
    """Minimal parameter stub with a grad attribute."""

    def __init__(self, data, grad=None):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = np.asarray(grad, dtype=np.float32) if grad is not None else None

    @property
    def shape(self):
        return self.data.shape


class _DummyOptimizer:
    """Minimal optimizer stub that records whether step() was called."""

    def __init__(self, params):
        self.param_groups = [{"params": params}]
        self.step_called = False
        self.step_count = 0

    def step(self, *args, **kwargs):
        self.step_called = True
        self.step_count += 1


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _import_amp():
    from grilly.backend.amp import GradScaler, autocast, is_autocast_enabled

    return autocast, GradScaler, is_autocast_enabled


# ---------------------------------------------------------------------------
# autocast tests
# ---------------------------------------------------------------------------


class TestAutocast:
    def test_import(self):
        autocast, GradScaler, is_autocast_enabled = _import_amp()
        assert autocast is not None
        assert is_autocast_enabled is not None

    def test_disabled_by_default(self):
        _, _, is_autocast_enabled = _import_amp()
        assert not is_autocast_enabled()

    def test_enabled_inside_context(self):
        autocast, _, is_autocast_enabled = _import_amp()
        assert not is_autocast_enabled()
        with autocast():
            assert is_autocast_enabled()
        assert not is_autocast_enabled()

    def test_disabled_after_context(self):
        autocast, _, is_autocast_enabled = _import_amp()
        with autocast():
            pass
        assert not is_autocast_enabled()

    def test_context_enabled_false(self):
        """autocast(enabled=False) inside outer autocast should disable it."""
        autocast, _, is_autocast_enabled = _import_amp()
        with autocast():
            assert is_autocast_enabled()
            with autocast(enabled=False):
                assert not is_autocast_enabled()
            # Restored after inner context
            assert is_autocast_enabled()

    def test_nested_autocast_restores_state(self):
        """Outer context state is restored after nested context exits."""
        autocast, _, is_autocast_enabled = _import_amp()
        with autocast():
            with autocast():
                assert is_autocast_enabled()
            assert is_autocast_enabled()
        assert not is_autocast_enabled()

    def test_autocast_as_decorator(self):
        autocast, _, is_autocast_enabled = _import_amp()

        @autocast()
        def fn():
            return is_autocast_enabled()

        assert fn()
        assert not is_autocast_enabled()

    def test_exception_restores_state(self):
        """State should be restored even if body raises."""
        autocast, _, is_autocast_enabled = _import_amp()
        with pytest.raises(ValueError):
            with autocast():
                raise ValueError("boom")
        assert not is_autocast_enabled()


# ---------------------------------------------------------------------------
# GradScaler tests
# ---------------------------------------------------------------------------


class TestGradScalerInit:
    def test_default_scale(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler()
        assert scaler.get_scale() == 65536.0

    def test_custom_scale(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=1024.0)
        assert scaler.get_scale() == 1024.0

    def test_enabled_flag(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler()
        assert scaler.enabled
        scaler_off = GradScaler(enabled=False)
        assert not scaler_off.enabled

    def test_invalid_growth_factor(self):
        _, GradScaler, _ = _import_amp()
        with pytest.raises(ValueError):
            GradScaler(growth_factor=0.9)

    def test_invalid_backoff_factor(self):
        _, GradScaler, _ = _import_amp()
        with pytest.raises(ValueError):
            GradScaler(backoff_factor=1.5)

    def test_invalid_growth_interval(self):
        _, GradScaler, _ = _import_amp()
        with pytest.raises(ValueError):
            GradScaler(growth_interval=0)


class TestGradScalerScale:
    def test_scale_float(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=4.0)
        assert scaler.scale(1.0) == 4.0

    def test_scale_numpy_scalar(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=8.0)
        result = scaler.scale(np.float32(2.0))
        assert float(result) == pytest.approx(16.0)

    def test_scale_numpy_array(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=2.0)
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = scaler.scale(arr)
        np.testing.assert_allclose(result, [2.0, 4.0, 6.0])

    def test_scale_disabled_returns_unchanged(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=1024.0, enabled=False)
        loss = 3.14
        assert scaler.scale(loss) is loss


class TestGradScalerUnscale:
    def test_unscale_gradients(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=4.0)
        param = _DummyParam([1.0], grad=[8.0, 8.0])
        param.grad = np.array([8.0, 8.0], dtype=np.float32)
        opt = _DummyOptimizer([param])
        scaler.unscale_(opt)
        np.testing.assert_allclose(param.grad, [2.0, 2.0])

    def test_unscale_disabled_noop(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=4.0, enabled=False)
        param = _DummyParam([1.0])
        param.grad = np.array([8.0], dtype=np.float32)
        opt = _DummyOptimizer([param])
        scaler.unscale_(opt)
        # Disabled scaler must not touch gradients
        np.testing.assert_allclose(param.grad, [8.0])

    def test_unscale_none_grad_ignored(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=2.0)
        param = _DummyParam([1.0])  # grad is None
        opt = _DummyOptimizer([param])
        # Should not raise
        scaler.unscale_(opt)


class TestGradScalerStep:
    def test_step_called_on_clean_grads(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=2.0)
        param = _DummyParam([1.0], grad=[1.0])
        opt = _DummyOptimizer([param])
        scaler.step(opt)
        assert opt.step_called

    def test_step_skipped_on_inf_grad(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=2.0)
        param = _DummyParam([1.0])
        param.grad = np.array([np.inf], dtype=np.float32)
        opt = _DummyOptimizer([param])
        result = scaler.step(opt)
        assert not opt.step_called
        assert result is None

    def test_step_skipped_on_nan_grad(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=2.0)
        param = _DummyParam([1.0])
        param.grad = np.array([np.nan], dtype=np.float32)
        opt = _DummyOptimizer([param])
        scaler.step(opt)
        assert not opt.step_called

    def test_step_disabled_always_calls_optimizer(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(enabled=False)
        param = _DummyParam([1.0])
        param.grad = np.array([np.inf], dtype=np.float32)
        opt = _DummyOptimizer([param])
        scaler.step(opt)
        # Disabled scaler should pass through regardless of gradient values
        assert opt.step_called


class TestGradScalerUpdate:
    def test_update_grows_scale_after_growth_interval(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=2.0, growth_factor=2.0, growth_interval=3)
        # Simulate 3 clean steps
        param = _DummyParam([1.0], grad=[1.0])
        opt = _DummyOptimizer([param])
        for _ in range(3):
            param.grad = np.array([1.0], dtype=np.float32)
            scaler.step(opt)
            scaler.update()
        assert scaler.get_scale() == pytest.approx(4.0)

    def test_update_backs_off_after_inf(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=64.0, backoff_factor=0.5, growth_interval=100)
        param = _DummyParam([1.0])
        param.grad = np.array([np.inf], dtype=np.float32)
        opt = _DummyOptimizer([param])
        scaler.step(opt)
        scaler.update()
        assert scaler.get_scale() == pytest.approx(32.0)

    def test_update_manual_new_scale(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=64.0)
        scaler.update(new_scale=128.0)
        assert scaler.get_scale() == 128.0

    def test_update_disabled_noop(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=64.0, enabled=False)
        scaler.update()
        # get_scale() returns 1.0 when disabled; internal _scale unchanged
        assert scaler._scale == 64.0

    def test_scale_never_below_min(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=2.0, backoff_factor=0.5, growth_interval=9999)
        param = _DummyParam([1.0])
        opt = _DummyOptimizer([param])
        # Drive scale below 1.0 by repeated inf steps
        for _ in range(10):
            param.grad = np.array([np.inf], dtype=np.float32)
            scaler.step(opt)
            scaler.update()
        assert scaler.get_scale() >= 1.0


class TestGradScalerStateDict:
    def test_state_dict_roundtrip(self):
        _, GradScaler, _ = _import_amp()
        scaler = GradScaler(init_scale=512.0, growth_factor=3.0, backoff_factor=0.25,
                            growth_interval=500)
        sd = scaler.state_dict()
        scaler2 = GradScaler()
        scaler2.load_state_dict(sd)
        assert scaler2._scale == 512.0
        assert scaler2._growth_factor == 3.0
        assert scaler2._backoff_factor == 0.25
        assert scaler2._growth_interval == 500


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------


class TestAMPIntegration:
    def test_autocast_and_scaler_together(self):
        """End-to-end: autocast active during forward, scaler handles backward."""
        autocast, GradScaler, is_autocast_enabled = _import_amp()

        scaler = GradScaler(init_scale=8.0)

        param = _DummyParam(np.random.randn(4, 4).astype(np.float32))
        opt = _DummyOptimizer([param])

        with autocast():
            assert is_autocast_enabled()
            # Simulate forward + loss (just a scalar)
            loss = np.float32(2.0)

        assert not is_autocast_enabled()

        scaled = scaler.scale(loss)
        assert float(scaled) == pytest.approx(16.0)

        # Set a clean gradient
        param.grad = np.ones((4, 4), dtype=np.float32)
        scaler.step(opt)
        scaler.update()

        assert opt.step_called
        # After 1 clean step, growth tracker == 1 (< 2000 default)
        assert scaler._growth_tracker == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
