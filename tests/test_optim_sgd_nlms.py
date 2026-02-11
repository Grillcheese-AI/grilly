"""Tests for SGD and NLMS optimizers."""

import numpy as np
import pytest

try:
    from grilly.nn import Parameter
    from grilly.optim import NLMS, SGD
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


class TestSGD:
    """Tests for SGD optimizer."""

    def test_sgd_import(self):
        """SGD can be imported."""
        from grilly.optim import SGD

        assert SGD is not None

    def test_sgd_init(self):
        """SGD initialization."""
        param = Parameter(np.random.randn(5, 5).astype(np.float32))
        opt = SGD([param], lr=0.01)
        assert opt.defaults["lr"] == 0.01
        assert opt.defaults["momentum"] == 0.0

    def test_sgd_step_basic(self):
        """SGD performs update step (CPU path)."""
        param = Parameter(np.ones((5, 5), dtype=np.float32))
        param.grad = np.ones_like(param.data) * 0.1

        opt = SGD([param], lr=0.01, use_gpu=False)
        before = np.asarray(param.data).copy()
        opt.step()
        after = np.asarray(param.data)

        assert not np.allclose(before, after)
        assert np.all(after < before)

    def test_sgd_with_momentum(self):
        """SGD with momentum."""
        param = Parameter(np.random.randn(5, 5).astype(np.float32))
        param.grad = np.random.randn(5, 5).astype(np.float32)

        opt = SGD([param], lr=0.01, momentum=0.9)
        opt.step()
        state = opt.state[id(param)]
        assert "momentum_buffer" in state

    def test_sgd_with_weight_decay(self):
        """SGD with weight decay."""
        param = Parameter(np.ones((5, 5), dtype=np.float32))
        param.grad = np.zeros_like(param.data)

        opt = SGD([param], lr=0.01, weight_decay=0.01)
        before = np.asarray(param.data).copy()
        opt.step()
        after = np.asarray(param.data)
        # With zero grad but weight decay, params should shrink
        assert np.all(np.abs(after) < np.abs(before))


class TestNLMS:
    """Tests for NLMS optimizer."""

    def test_nlms_import(self):
        """NLMS can be imported."""
        from grilly.optim import NLMS

        assert NLMS is not None

    def test_nlms_init(self):
        """NLMS initialization."""
        param = Parameter(np.random.randn(10).astype(np.float32))
        opt = NLMS([param], lr=0.5, use_gpu=False)
        assert opt.defaults["lr"] == 0.5
        assert opt.defaults["lr_min"] == 0.1

    def test_nlms_step_cpu(self):
        """NLMS performs update step (CPU path)."""
        param = Parameter(np.zeros(10, dtype=np.float32))
        param.grad = np.ones(10, dtype=np.float32) * 0.1

        opt = NLMS([param], lr=0.5, use_gpu=False)
        before = np.asarray(param.data).copy()
        opt.step()
        after = np.asarray(param.data)

        assert not np.allclose(before, after)
        state = opt.state[id(param)]
        assert "mu" in state
        assert "update_count" in state
