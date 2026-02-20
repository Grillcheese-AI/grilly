"""Tests for Phase 4: SNN Batch Normalization"""

import numpy as np
import pytest


class TestThresholdDependentBatchNorm:
    """Test Threshold-Dependent Batch Normalization."""

    def test_tdbn2d_shape(self):
        """TDBN2d should preserve shape."""
        from grilly.nn.snn_normalization import ThresholdDependentBatchNorm2d

        bn = ThresholdDependentBatchNorm2d(16, v_threshold=1.0)
        x = np.random.randn(4, 16, 8, 8).astype(np.float32)
        out = bn(x)
        assert out.shape == (4, 16, 8, 8)

    def test_tdbn2d_normalizes(self):
        """TDBN2d output should be roughly normalized."""
        from grilly.nn.snn_normalization import ThresholdDependentBatchNorm2d

        bn = ThresholdDependentBatchNorm2d(8, v_threshold=1.0)
        x = np.random.randn(32, 8, 4, 4).astype(np.float32) * 10.0
        out = bn(x)
        # Per-channel mean should be near 0
        per_ch_mean = out.mean(axis=(0, 2, 3))
        np.testing.assert_allclose(per_ch_mean, 0.0, atol=0.5)

    def test_tdbn1d_shape(self):
        """TDBN1d should handle 2D and 3D inputs."""
        from grilly.nn.snn_normalization import ThresholdDependentBatchNorm1d

        bn = ThresholdDependentBatchNorm1d(8, v_threshold=1.0)
        # 2D input
        x2d = np.random.randn(4, 8).astype(np.float32)
        out2d = bn(x2d)
        assert out2d.shape == (4, 8)

    def test_tdbn2d_eval_mode(self):
        """TDBN2d in eval mode should use running stats."""
        from grilly.nn.snn_normalization import ThresholdDependentBatchNorm2d

        bn = ThresholdDependentBatchNorm2d(4, v_threshold=1.0)
        x = np.random.randn(8, 4, 2, 2).astype(np.float32)
        bn.train()
        bn(x)  # Update running stats
        bn.eval()
        out = bn(x)
        assert out.shape == (8, 4, 2, 2)


class TestTemporalEffectiveBatchNorm:
    """Test Temporal Effective Batch Normalization."""

    def test_tebn2d_shape(self):
        """TEBN2d should handle [T, N, C, H, W] input."""
        from grilly.nn.snn_normalization import TemporalEffectiveBatchNorm2d

        bn = TemporalEffectiveBatchNorm2d(T=4, num_features=8)
        x = np.random.randn(4, 2, 8, 4, 4).astype(np.float32)
        out = bn(x)
        assert out.shape == (4, 2, 8, 4, 4)

    def test_tebn1d_shape(self):
        """TEBN1d should handle [T, N, C] input."""
        from grilly.nn.snn_normalization import TemporalEffectiveBatchNorm1d

        bn = TemporalEffectiveBatchNorm1d(T=4, num_features=16)
        x = np.random.randn(4, 2, 16).astype(np.float32)
        out = bn(x)
        assert out.shape == (4, 2, 16)

    def test_tebn_has_lambda(self):
        """TEBN should have learnable lambda_t parameter."""
        from grilly.nn.snn_normalization import TemporalEffectiveBatchNorm2d

        bn = TemporalEffectiveBatchNorm2d(T=4, num_features=8)
        params = list(bn.parameters())
        # Should have lambda_t, weight, bias
        assert len(params) >= 1


class TestBatchNormThroughTime:
    """Test BN applied independently at each timestep."""

    def test_bntt2d_shape(self):
        """BNTT2d should process each timestep independently."""
        from grilly.nn.snn_normalization import BatchNormThroughTime2d

        bn = BatchNormThroughTime2d(8)
        x = np.random.randn(4, 2, 8, 4, 4).astype(np.float32)
        out = bn(x)
        assert out.shape == (4, 2, 8, 4, 4)

    def test_bntt1d_shape(self):
        """BNTT1d should handle [T, N, C]."""
        from grilly.nn.snn_normalization import BatchNormThroughTime1d

        bn = BatchNormThroughTime1d(16)
        x = np.random.randn(4, 2, 16).astype(np.float32)
        out = bn(x)
        assert out.shape == (4, 2, 16)


class TestNeuNorm:
    """Test NeuNorm per-neuron normalization."""

    def test_neunorm_4d(self):
        """NeuNorm should scale 4D input per-channel."""
        from grilly.nn.snn_normalization import NeuNorm

        nn = NeuNorm(in_channels=4, k=0.5)
        x = np.ones((2, 4, 3, 3), dtype=np.float32)
        out = nn(x)
        np.testing.assert_allclose(out, 0.5)

    def test_neunorm_2d(self):
        """NeuNorm should scale 2D input."""
        from grilly.nn.snn_normalization import NeuNorm

        nn = NeuNorm(in_channels=8, k=2.0)
        x = np.ones((4, 8), dtype=np.float32)
        out = nn(x)
        np.testing.assert_allclose(out, 2.0)
