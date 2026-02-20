"""Tests for Phase 2: Neuron Nodes (IFNode, LIFNode, ParametricLIFNode)"""

import numpy as np
import pytest


class TestIFNode:
    """Test Integrate-and-Fire neuron."""

    def test_accumulates_and_fires(self):
        """IF should accumulate input and fire at threshold."""
        from grilly.nn.snn_neurons import IFNode

        node = IFNode(v_threshold=1.0, v_reset=0.0, step_mode="s")
        x = np.array([[0.6]], dtype=np.float32)

        # First step: 0.6 < 1.0, no spike
        spike1 = node(x)
        assert spike1[0, 0] == 0.0
        assert node.v[0, 0] == pytest.approx(0.6)

        # Second step: 0.6 + 0.6 = 1.2 >= 1.0, spike!
        spike2 = node(x)
        assert spike2[0, 0] == 1.0

    def test_hard_reset(self):
        """IF with hard reset should reset to v_reset after spike."""
        from grilly.nn.snn_neurons import IFNode

        node = IFNode(v_threshold=1.0, v_reset=0.0, step_mode="s")
        x = np.array([[1.5]], dtype=np.float32)
        node(x)  # h=1.5, spike, hard reset -> v=0.0
        assert node.v[0, 0] == pytest.approx(0.0)

    def test_soft_reset(self):
        """IF with soft reset subtracts threshold."""
        from grilly.nn.snn_neurons import IFNode

        node = IFNode(v_threshold=1.0, v_reset=None, step_mode="s")
        x = np.array([[1.5]], dtype=np.float32)
        node(x)  # h=1.5, spike, soft reset -> v = 1.5 - 1.0 = 0.5
        assert node.v[0, 0] == pytest.approx(0.5)

    def test_single_step_shape(self):
        """Single step: [N, D] -> [N, D]."""
        from grilly.nn.snn_neurons import IFNode

        node = IFNode(step_mode="s")
        x = np.random.rand(8, 128).astype(np.float32)
        out = node(x)
        assert out.shape == (8, 128)

    def test_multi_step_shape(self):
        """Multi step: [T, N, D] -> [T, N, D]."""
        from grilly.nn.snn_neurons import IFNode

        node = IFNode(step_mode="m")
        x = np.random.rand(4, 8, 128).astype(np.float32) * 0.5
        out = node(x)
        assert out.shape == (4, 8, 128)
        assert set(np.unique(out)).issubset({0.0, 1.0})


class TestLIFNode:
    """Test Leaky Integrate-and-Fire neuron."""

    def test_leaks_toward_reset(self):
        """LIF should leak toward v_reset when no input."""
        from grilly.nn.snn_neurons import LIFNode

        node = LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0, step_mode="s")
        # First: inject enough to get some v
        x = np.array([[0.8]], dtype=np.float32)
        node(x)
        v_after_inject = node.v[0, 0].copy()

        # Then give zero input — v should decay
        x_zero = np.array([[0.0]], dtype=np.float32)
        node(x_zero)
        v_after_decay = node.v[0, 0]
        assert v_after_decay < v_after_inject

    def test_tau_controls_leak(self):
        """Higher tau = slower leak. With decay_input=False (default),
        the leak only affects stored voltage, not input."""
        from grilly.nn.snn_neurons import LIFNode

        # With decay_input=False: H = V*(1-1/tau) + X
        # First step (V=0): H = X = 0.5 for both — no spike (thresh=10), so V = 0.5
        # Second step: H = 0.5*(1-1/tau) + 0.5
        node_fast = LIFNode(tau=1.5, v_threshold=10.0, v_reset=0.0, step_mode="s")
        x = np.array([[0.5]], dtype=np.float32)
        node_fast(x)  # V = 0.5
        node_fast(x)  # H = 0.5*(1-1/1.5) + 0.5 = 0.5*0.333 + 0.5 = 0.6667
        v_fast = node_fast.v[0, 0].copy()

        node_slow = LIFNode(tau=10.0, v_threshold=10.0, v_reset=0.0, step_mode="s")
        node_slow(x)  # V = 0.5
        node_slow(x)  # H = 0.5*(1-1/10) + 0.5 = 0.5*0.9 + 0.5 = 0.95
        v_slow = node_slow.v[0, 0].copy()

        # Higher tau = less leak = more voltage retained
        assert v_fast == pytest.approx(0.5 * (1 - 1 / 1.5) + 0.5, abs=1e-5)
        assert v_slow == pytest.approx(0.5 * (1 - 1 / 10.0) + 0.5, abs=1e-5)
        assert v_slow > v_fast  # slower leak retains more voltage

    def test_decay_input_mode(self):
        """decay_input=True divides input by tau (physics-accurate)."""
        from grilly.nn.snn_neurons import LIFNode

        node = LIFNode(tau=2.0, decay_input=True, v_threshold=10.0, v_reset=0.0, step_mode="s")
        x = np.array([[1.0]], dtype=np.float32)
        node(x)
        # H = V*(1-1/tau) + X/tau = 0 + 1.0/2.0 = 0.5
        assert node.v[0, 0] == pytest.approx(0.5, abs=1e-5)

        node2 = LIFNode(tau=2.0, decay_input=False, v_threshold=10.0, v_reset=0.0, step_mode="s")
        node2(x)
        # H = V*(1-1/tau) + X = 0 + 1.0 = 1.0
        assert node2.v[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_tau_validation(self):
        """tau < 1.0 should raise ValueError."""
        from grilly.nn.snn_neurons import LIFNode

        with pytest.raises(ValueError, match="tau must be >= 1.0"):
            LIFNode(tau=0.5)

    def test_multi_step(self):
        """LIF multi-step should process temporal sequence."""
        from grilly.nn.snn_neurons import LIFNode

        node = LIFNode(tau=2.0, step_mode="m")
        x = np.random.rand(10, 4, 64).astype(np.float32) * 0.3
        out = node(x)
        assert out.shape == (10, 4, 64)


class TestParametricLIFNode:
    """Test Parametric LIF with learnable tau."""

    def test_tau_is_parameter(self):
        """ParametricLIF tau should be a registered Parameter."""
        from grilly.nn.snn_neurons import ParametricLIFNode

        node = ParametricLIFNode(init_tau=2.0)
        params = list(node.parameters())
        assert len(params) == 1  # tau
        assert params[0].shape == (1,)

    def test_tau_gradient(self):
        """ParametricLIF tau should be a Parameter with requires_grad."""
        from grilly.nn.snn_neurons import ParametricLIFNode

        node = ParametricLIFNode(init_tau=2.0)
        assert node.tau.requires_grad is True

    def test_forward_produces_spikes(self):
        """ParametricLIF should produce binary spikes."""
        from grilly.nn.snn_neurons import ParametricLIFNode

        node = ParametricLIFNode(init_tau=2.0, step_mode="s")
        x = np.random.rand(4, 32).astype(np.float32) * 2.0
        out = node(x)
        assert out.shape == (4, 32)
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_multi_step(self):
        """ParametricLIF multi-step."""
        from grilly.nn.snn_neurons import ParametricLIFNode

        node = ParametricLIFNode(init_tau=2.0, step_mode="m")
        x = np.random.rand(5, 4, 32).astype(np.float32) * 0.5
        out = node(x)
        assert out.shape == (5, 4, 32)

    def test_repr(self):
        from grilly.nn.snn_neurons import ParametricLIFNode

        node = ParametricLIFNode(init_tau=3.0)
        r = repr(node)
        assert "ParametricLIFNode" in r
        assert "3.00" in r


class TestBackwardSurrogate:
    """Test that backward through surrogate produces non-zero gradients."""

    def test_surrogate_gradient_nonzero(self):
        """Surrogate gradient at the firing threshold should be nonzero."""
        from grilly.nn.snn_surrogate import ATan

        fn = ATan(alpha=2.0)
        # At x=0 (threshold), gradient should be maximal
        x = np.array([0.0], dtype=np.float32)
        grad = fn.gradient(x)
        assert grad[0] > 0

    def test_gradient_decays_away_from_threshold(self):
        """Gradient should decrease away from threshold."""
        from grilly.nn.snn_surrogate import ATan

        fn = ATan(alpha=2.0)
        x_near = np.array([0.01], dtype=np.float32)
        x_far = np.array([5.0], dtype=np.float32)
        assert fn.gradient(x_near)[0] > fn.gradient(x_far)[0]
