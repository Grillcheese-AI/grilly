"""Tests for utils.initialization module."""

import numpy as np
import pytest

try:
    from grilly.utils.initialization import (
        kaiming_normal_,
        kaiming_uniform_,
        xavier_normal_,
        xavier_uniform_,
    )
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


class TestXavierUniform:
    """Tests for xavier_uniform_."""

    def test_xavier_uniform_2d(self):
        """xavier_uniform_ initializes 2D tensor."""
        t = np.zeros((64, 128), dtype=np.float32)
        out = xavier_uniform_(t)
        assert out is t
        assert not np.allclose(t, 0)
        assert np.abs(t).max() < 1.0

    def test_xavier_uniform_3d(self):
        """xavier_uniform_ works with 3D tensor."""
        t = np.zeros((4, 64, 128), dtype=np.float32)
        xavier_uniform_(t)
        assert not np.allclose(t, 0)

    def test_xavier_uniform_gain(self):
        """xavier_uniform_ respects gain parameter."""
        t = np.zeros((32, 32), dtype=np.float32)
        xavier_uniform_(t, gain=2.0)
        assert np.abs(t).max() > 0.1

    def test_xavier_uniform_1d_raises(self):
        """xavier_uniform_ raises for 1D tensor."""
        t = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            xavier_uniform_(t)


class TestXavierNormal:
    """Tests for xavier_normal_."""

    def test_xavier_normal_2d(self):
        """xavier_normal_ initializes 2D tensor."""
        t = np.zeros((64, 128), dtype=np.float32)
        out = xavier_normal_(t)
        assert out is t
        assert not np.allclose(t, 0)

    def test_xavier_normal_1d_raises(self):
        """xavier_normal_ raises for 1D tensor."""
        t = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            xavier_normal_(t)


class TestKaimingUniform:
    """Tests for kaiming_uniform_."""

    def test_kaiming_uniform_fan_in(self):
        """kaiming_uniform_ with fan_in mode."""
        t = np.zeros((64, 128), dtype=np.float32)
        kaiming_uniform_(t, mode="fan_in")
        assert not np.allclose(t, 0)

    def test_kaiming_uniform_fan_out(self):
        """kaiming_uniform_ with fan_out mode."""
        t = np.zeros((64, 128), dtype=np.float32)
        kaiming_uniform_(t, mode="fan_out")
        assert not np.allclose(t, 0)

    def test_kaiming_uniform_relu(self):
        """kaiming_uniform_ with relu nonlinearity."""
        t = np.zeros((32, 32), dtype=np.float32)
        kaiming_uniform_(t, nonlinearity="relu")
        assert not np.allclose(t, 0)

    def test_kaiming_uniform_leaky_relu(self):
        """kaiming_uniform_ with leaky_relu and slope."""
        t = np.zeros((32, 32), dtype=np.float32)
        kaiming_uniform_(t, a=0.01, nonlinearity="leaky_relu")
        assert not np.allclose(t, 0)

    def test_kaiming_uniform_1d_raises(self):
        """kaiming_uniform_ raises for 1D tensor."""
        t = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            kaiming_uniform_(t)


class TestKaimingNormal:
    """Tests for kaiming_normal_."""

    def test_kaiming_normal_fan_in(self):
        """kaiming_normal_ with fan_in mode."""
        t = np.zeros((64, 128), dtype=np.float32)
        kaiming_normal_(t, mode="fan_in")
        assert not np.allclose(t, 0)

    def test_kaiming_normal_fan_out(self):
        """kaiming_normal_ with fan_out mode."""
        t = np.zeros((64, 128), dtype=np.float32)
        kaiming_normal_(t, mode="fan_out")
        assert not np.allclose(t, 0)

    def test_kaiming_normal_1d_raises(self):
        """kaiming_normal_ raises for 1D tensor."""
        t = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            kaiming_normal_(t)
