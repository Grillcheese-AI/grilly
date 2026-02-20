"""Tests for Phase 10: SNN Monitoring"""

import numpy as np
import pytest


class TestMonitor:
    """Test Monitor recording and retrieval."""

    def test_record_membrane_potential(self):
        """Monitor should record v over time."""
        from grilly.nn.snn_monitor import Monitor
        from grilly.nn.snn_neurons import LIFNode

        node = LIFNode(tau=2.0, v_threshold=1.0, step_mode="s")
        mon = Monitor(node, var_names=["v"])

        T = 10
        for t in range(T):
            x = np.random.rand(1, 4).astype(np.float32) * 0.3
            node(x)
            mon.record()

        data = mon.get("v")
        assert data.shape == (T, 1, 4)

    def test_num_recordings(self):
        """num_recordings should track count."""
        from grilly.nn.snn_monitor import Monitor
        from grilly.nn.snn_neurons import IFNode

        node = IFNode(step_mode="s")
        mon = Monitor(node, var_names=["v"])
        assert mon.num_recordings == 0

        node(np.ones((1, 2), dtype=np.float32))
        mon.record()
        assert mon.num_recordings == 1

        node(np.ones((1, 2), dtype=np.float32))
        mon.record()
        assert mon.num_recordings == 2

    def test_reset_clears(self):
        """Reset should clear all recordings."""
        from grilly.nn.snn_monitor import Monitor
        from grilly.nn.snn_neurons import IFNode

        node = IFNode(step_mode="s")
        mon = Monitor(node, var_names=["v"])
        node(np.ones((1, 2), dtype=np.float32))
        mon.record()
        assert mon.num_recordings == 1

        mon.reset()
        assert mon.num_recordings == 0

    def test_disable_enable(self):
        """Disable should skip recordings."""
        from grilly.nn.snn_monitor import Monitor
        from grilly.nn.snn_neurons import IFNode

        node = IFNode(step_mode="s")
        mon = Monitor(node, var_names=["v"])

        node(np.ones((1, 2), dtype=np.float32))
        mon.record()
        assert mon.num_recordings == 1

        mon.disable()
        node(np.ones((1, 2), dtype=np.float32))
        mon.record()
        assert mon.num_recordings == 1  # Still 1

        mon.enable()
        node(np.ones((1, 2), dtype=np.float32))
        mon.record()
        assert mon.num_recordings == 2

    def test_invalid_var_name(self):
        """Getting non-monitored variable should raise KeyError."""
        from grilly.nn.snn_monitor import Monitor
        from grilly.nn.snn_neurons import IFNode

        node = IFNode(step_mode="s")
        mon = Monitor(node, var_names=["v"])
        with pytest.raises(KeyError, match="not monitored"):
            mon.get("nonexistent")

    def test_empty_get(self):
        """Getting variable with no recordings returns empty array."""
        from grilly.nn.snn_monitor import Monitor
        from grilly.nn.snn_neurons import IFNode

        node = IFNode(step_mode="s")
        mon = Monitor(node, var_names=["v"])
        data = mon.get("v")
        assert data.size == 0

    def test_repr(self):
        from grilly.nn.snn_monitor import Monitor
        from grilly.nn.snn_neurons import IFNode

        node = IFNode(step_mode="s")
        mon = Monitor(node, var_names=["v"])
        r = repr(mon)
        assert "Monitor" in r
        assert "IFNode" in r
