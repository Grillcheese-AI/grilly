"""
CPU-only tests for nn/normalization.py CPU fallback paths.

Tests BatchNorm2d._batchnorm2d_cpu() and _batchnorm2d_backward_cpu() directly,
bypassing Vulkan GPU dispatch entirely.
"""

import numpy as np
import pytest

try:
    from grilly.nn.normalization import BatchNorm1d, BatchNorm2d
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


@pytest.fixture(autouse=True)
def seed_rng():
    """Seed numpy RNG for reproducibility."""
    np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. Initialization tests
# ---------------------------------------------------------------------------
class TestBatchNorm2dInit:
    """Verify BatchNorm2d constructor defaults and parameter initialization."""

    def test_default_params(self):
        """num_features, eps, momentum, affine, track_running_stats defaults."""
        bn = BatchNorm2d(16)
        assert bn.num_features == 16
        assert bn.eps == 1e-5
        assert bn.momentum == 0.1
        assert bn.affine is True
        assert bn.track_running_stats is True

    def test_weight_ones(self):
        """Weight (gamma) should be initialized to ones."""
        bn = BatchNorm2d(8)
        np.testing.assert_array_equal(np.asarray(bn.weight), np.ones(8, dtype=np.float32))

    def test_bias_zeros(self):
        """Bias (beta) should be initialized to zeros."""
        bn = BatchNorm2d(8)
        np.testing.assert_array_equal(np.asarray(bn.bias), np.zeros(8, dtype=np.float32))

    def test_running_mean_zeros(self):
        """Running mean should be initialized to zeros."""
        bn = BatchNorm2d(8)
        np.testing.assert_array_equal(bn.running_mean, np.zeros(8, dtype=np.float32))

    def test_running_var_ones(self):
        """Running var should be initialized to ones."""
        bn = BatchNorm2d(8)
        np.testing.assert_array_equal(bn.running_var, np.ones(8, dtype=np.float32))

    def test_affine_false(self):
        """When affine=False, weight and bias should be None."""
        bn = BatchNorm2d(8, affine=False)
        assert bn.weight is None
        assert bn.bias is None

    def test_no_running_stats(self):
        """When track_running_stats=False, running_mean/var should be None."""
        bn = BatchNorm2d(8, track_running_stats=False)
        assert bn.running_mean is None
        assert bn.running_var is None

    def test_repr(self):
        """__repr__ should include num_features and key params."""
        bn = BatchNorm2d(32, eps=1e-3, momentum=0.2)
        r = repr(bn)
        assert "32" in r
        assert "BatchNorm2d" in r
        assert "eps" in r
        assert "momentum" in r

    def test_extra_repr(self):
        """extra_repr should include all config params."""
        bn = BatchNorm2d(16, eps=1e-4, momentum=0.05, affine=False, track_running_stats=False)
        er = bn.extra_repr()
        assert "16" in er
        assert "1e-04" in er or "0.0001" in er
        assert "0.05" in er
        assert "affine=False" in er
        assert "track_running_stats=False" in er


# ---------------------------------------------------------------------------
# 2. CPU forward pass tests
# ---------------------------------------------------------------------------
class TestBatchNorm2dCPUForward:
    """Test _batchnorm2d_cpu() forward path directly."""

    def _make_bn(self, num_features=8, **kwargs):
        """Create a BatchNorm2d in training mode with grad enabled."""
        bn = BatchNorm2d(num_features, **kwargs)
        bn.training = True
        bn._grad_enabled = True
        return bn

    def test_output_shape(self):
        """Output shape must match input (N, C, H, W)."""
        bn = self._make_bn(8)
        x = np.random.randn(4, 8, 6, 6).astype(np.float32)
        output = bn._batchnorm2d_cpu(x)
        assert output.shape == (4, 8, 6, 6)
        assert output.dtype == np.float32

    def test_training_normalization(self):
        """In training mode, per-channel output should have mean ~0 and var ~1."""
        bn = self._make_bn(8)
        x = np.random.randn(32, 8, 6, 6).astype(np.float32) * 5 + 3  # shifted/scaled
        output = bn._batchnorm2d_cpu(x)

        # With affine=True and default weight=1, bias=0 the output should
        # have per-channel mean ~0 and variance ~1.
        for c in range(8):
            ch = output[:, c, :, :]
            np.testing.assert_allclose(ch.mean(), 0.0, atol=1e-5)
            np.testing.assert_allclose(ch.var(), 1.0, atol=1e-2)

    def test_training_running_stats_update(self):
        """Running mean/var should be updated with momentum after forward."""
        bn = self._make_bn(4)
        x = np.random.randn(8, 4, 4, 4).astype(np.float32) * 2 + 1

        old_mean = bn.running_mean.copy()
        old_var = bn.running_var.copy()

        bn._batchnorm2d_cpu(x)

        # Running stats should have changed
        assert not np.allclose(bn.running_mean, old_mean)
        assert not np.allclose(bn.running_var, old_var)

        # Verify momentum blending: new = momentum * batch + (1-momentum) * old
        batch_mean = np.mean(x, axis=(0, 2, 3))
        batch_var = np.var(x, axis=(0, 2, 3))
        expected_mean = 0.1 * batch_mean + 0.9 * old_mean
        expected_var = 0.1 * batch_var + 0.9 * old_var
        np.testing.assert_allclose(bn.running_mean, expected_mean, rtol=1e-5)
        np.testing.assert_allclose(bn.running_var, expected_var, rtol=1e-5)

    def test_eval_uses_running_stats(self):
        """In eval mode with tracked stats, running_mean/var should be used."""
        bn = self._make_bn(4)
        bn.training = True

        # Populate running stats with a training pass
        x_train = np.random.randn(16, 4, 4, 4).astype(np.float32)
        bn._batchnorm2d_cpu(x_train)
        saved_mean = bn.running_mean.copy()
        saved_var = bn.running_var.copy()

        # Switch to eval
        bn.training = False
        x_eval = np.random.randn(4, 4, 4, 4).astype(np.float32) * 10 + 5
        output = bn._batchnorm2d_cpu(x_eval)

        # Running stats should NOT change during eval
        np.testing.assert_array_equal(bn.running_mean, saved_mean)
        np.testing.assert_array_equal(bn.running_var, saved_var)

        # Manually compute expected output using running stats
        mean_exp = saved_mean[np.newaxis, :, np.newaxis, np.newaxis]
        var_exp = saved_var[np.newaxis, :, np.newaxis, np.newaxis]
        x_norm = (x_eval - mean_exp) / np.sqrt(var_exp + bn.eps)
        gamma = np.asarray(bn.weight)[np.newaxis, :, np.newaxis, np.newaxis]
        beta = np.asarray(bn.bias)[np.newaxis, :, np.newaxis, np.newaxis]
        expected = gamma * x_norm + beta
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)

    def test_affine_transform(self):
        """Set weight/bias to known values, verify output = gamma*normalized + beta."""
        bn = self._make_bn(4)
        # Override weight and bias with known values
        bn.weight[:] = np.array([2.0, 0.5, 3.0, 1.0], dtype=np.float32)
        bn.bias[:] = np.array([1.0, -1.0, 0.0, 0.5], dtype=np.float32)

        x = np.random.randn(8, 4, 4, 4).astype(np.float32)
        output = bn._batchnorm2d_cpu(x)

        # Manually compute normalized values (using batch stats)
        mean = np.mean(x, axis=(0, 2, 3))
        var = np.var(x, axis=(0, 2, 3))
        mean_exp = mean[np.newaxis, :, np.newaxis, np.newaxis]
        var_exp = var[np.newaxis, :, np.newaxis, np.newaxis]
        x_norm = (x - mean_exp) / np.sqrt(var_exp + bn.eps)

        gamma = np.asarray(bn.weight)[np.newaxis, :, np.newaxis, np.newaxis]
        beta = np.asarray(bn.bias)[np.newaxis, :, np.newaxis, np.newaxis]
        expected = gamma * x_norm + beta
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)

    def test_affine_false_no_transform(self):
        """With affine=False, output should be just the normalized values."""
        bn = self._make_bn(4, affine=False)
        x = np.random.randn(8, 4, 4, 4).astype(np.float32) * 3 + 2
        output = bn._batchnorm2d_cpu(x)

        # Manually compute normalized values
        mean = np.mean(x, axis=(0, 2, 3))
        var = np.var(x, axis=(0, 2, 3))
        mean_exp = mean[np.newaxis, :, np.newaxis, np.newaxis]
        var_exp = var[np.newaxis, :, np.newaxis, np.newaxis]
        expected = (x - mean_exp) / np.sqrt(var_exp + bn.eps)
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)

    def test_track_running_stats_false(self):
        """When track_running_stats=False, batch stats are used even in eval."""
        bn = self._make_bn(4, track_running_stats=False)

        # Even in eval mode, with no running stats, batch stats must be used
        bn.training = False
        x = np.random.randn(16, 4, 4, 4).astype(np.float32) * 5 + 3
        output = bn._batchnorm2d_cpu(x)

        # Manually compute using batch stats (not running stats, which are None)
        mean = np.mean(x, axis=(0, 2, 3))
        var = np.var(x, axis=(0, 2, 3))
        mean_exp = mean[np.newaxis, :, np.newaxis, np.newaxis]
        var_exp = var[np.newaxis, :, np.newaxis, np.newaxis]
        x_norm = (x - mean_exp) / np.sqrt(var_exp + bn.eps)
        gamma = np.asarray(bn.weight)[np.newaxis, :, np.newaxis, np.newaxis]
        beta = np.asarray(bn.bias)[np.newaxis, :, np.newaxis, np.newaxis]
        expected = gamma * x_norm + beta
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. CPU backward pass tests
# ---------------------------------------------------------------------------
class TestBatchNorm2dCPUBackward:
    """Test _batchnorm2d_backward_cpu() directly."""

    def _forward_and_setup(self, num_features=8, batch=4, h=6, w=6):
        """Run CPU forward in training mode and return (bn, x, output)."""
        bn = BatchNorm2d(num_features)
        bn.training = True
        bn._grad_enabled = True

        x = np.random.randn(batch, num_features, h, w).astype(np.float32)
        output = bn._batchnorm2d_cpu(x)
        # _batchnorm2d_cpu caches batch_mean/batch_var but NOT input.
        # forward() does that, so replicate it here.
        bn._cache_input = x.copy()
        return bn, x, output

    def test_grad_input_shape(self):
        """grad_input should have the same shape as the input."""
        bn, x, _ = self._forward_and_setup()
        grad_out = np.random.randn(*x.shape).astype(np.float32)
        grad_input = bn._batchnorm2d_backward_cpu(grad_out)
        assert grad_input.shape == x.shape
        assert grad_input.dtype == np.float32

    def test_weight_grad_shape(self):
        """weight.grad should have shape (num_features,)."""
        bn, x, _ = self._forward_and_setup(num_features=8)
        grad_out = np.random.randn(*x.shape).astype(np.float32)
        bn._batchnorm2d_backward_cpu(grad_out)
        assert bn.weight.grad is not None
        assert bn.weight.grad.shape == (8,)

    def test_bias_grad_shape(self):
        """bias.grad should have shape (num_features,)."""
        bn, x, _ = self._forward_and_setup(num_features=8)
        grad_out = np.random.randn(*x.shape).astype(np.float32)
        bn._batchnorm2d_backward_cpu(grad_out)
        assert bn.bias.grad is not None
        assert bn.bias.grad.shape == (8,)

    def test_numerical_gradient(self):
        """Verify grad_input against finite-difference numerical gradient.

        Uses float64 for the numerical computation to reduce floating-point
        noise in the finite-difference estimate. Batch normalization gradients
        are coupled across the whole batch, so float32 finite differences are
        inherently noisier than element-wise ops.
        """
        num_features = 2
        bn, x, _ = self._forward_and_setup(num_features=num_features, batch=2, h=3, w=3)

        grad_out = np.random.randn(*x.shape).astype(np.float32)
        grad_input = bn._batchnorm2d_backward_cpu(grad_out)

        # Use float64 for finite-difference to reduce truncation error
        x_64 = x.astype(np.float64)
        grad_out_64 = grad_out.astype(np.float64)
        weight_64 = np.asarray(bn.weight).astype(np.float64)
        bias_64 = np.asarray(bn.bias).astype(np.float64)

        def _forward_64(inp):
            """Pure-numpy BN forward in float64 for finite-difference."""
            mean = np.mean(inp, axis=(0, 2, 3), keepdims=True)
            var = np.var(inp, axis=(0, 2, 3), keepdims=True)
            x_norm = (inp - mean) / np.sqrt(var + bn.eps)
            gamma = weight_64[np.newaxis, :, np.newaxis, np.newaxis]
            beta = bias_64[np.newaxis, :, np.newaxis, np.newaxis]
            return gamma * x_norm + beta

        eps_fd = 1e-5
        numerical_grad = np.zeros_like(x_64)
        for idx in np.ndindex(x_64.shape):
            x_plus = x_64.copy()
            x_plus[idx] += eps_fd
            x_minus = x_64.copy()
            x_minus[idx] -= eps_fd

            out_plus = _forward_64(x_plus)
            out_minus = _forward_64(x_minus)
            numerical_grad[idx] = np.sum((out_plus - out_minus) / (2 * eps_fd) * grad_out_64)

        np.testing.assert_allclose(
            grad_input, numerical_grad.astype(np.float32), rtol=1e-3, atol=1e-4
        )


# ---------------------------------------------------------------------------
# 4. BatchNorm1d tests
# ---------------------------------------------------------------------------
class TestBatchNorm1d:
    """Test BatchNorm1d wrapper using CPU fallback path."""

    def test_2d_input(self):
        """(N, C) input should produce output of the same shape."""
        bn = BatchNorm1d(16)
        bn.bn2d.training = True
        x = np.random.randn(8, 16).astype(np.float32)
        # Manually reshape and call CPU path like forward() does
        x_4d = x[:, :, np.newaxis, np.newaxis]
        out_4d = bn.bn2d._batchnorm2d_cpu(x_4d)
        output = out_4d.squeeze((2, 3))
        assert output.shape == (8, 16)
        assert output.dtype == np.float32

    def test_3d_input(self):
        """(N, C, L) input should produce output of the same shape."""
        bn = BatchNorm1d(16)
        bn.bn2d.training = True
        x = np.random.randn(8, 16, 20).astype(np.float32)
        x_4d = x[:, :, np.newaxis, :]
        out_4d = bn.bn2d._batchnorm2d_cpu(x_4d)
        output = out_4d.squeeze(2)
        assert output.shape == (8, 16, 20)
        assert output.dtype == np.float32

    def test_invalid_ndim_raises(self):
        """4D input should raise ValueError."""
        bn = BatchNorm1d(16)
        x = np.random.randn(4, 16, 8, 8).astype(np.float32)
        with pytest.raises(ValueError, match="2D or 3D"):
            bn.forward(x)

    def test_shares_parameters_with_bn2d(self):
        """BatchNorm1d.weight/bias should be the same objects as bn2d's."""
        bn = BatchNorm1d(8)
        assert bn.weight is bn.bn2d.weight
        assert bn.bias is bn.bn2d.bias

    def test_repr(self):
        """__repr__ should include 'BatchNorm1d'."""
        bn = BatchNorm1d(32)
        r = repr(bn)
        assert "BatchNorm1d" in r
        assert "32" in r
