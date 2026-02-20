"""Tests for Phase 1: Surrogate Gradients + MemoryModule + BaseNode"""

import numpy as np
import pytest


class TestSurrogateGradients:
    """Test surrogate gradient functions."""

    def test_atan_forward_binary(self):
        """ATan forward should produce binary output (Heaviside)."""
        from grilly.nn.snn_surrogate import ATan

        fn = ATan(alpha=2.0)
        x = np.array([-1.0, -0.1, 0.0, 0.1, 1.0], dtype=np.float32)
        out = fn(x)
        # Heaviside: >= 0 -> 1, < 0 -> 0
        expected = np.array([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_atan_gradient_smooth(self):
        """ATan backward should give smooth, non-zero gradient."""
        from grilly.nn.snn_surrogate import ATan

        fn = ATan(alpha=2.0)
        x = np.array([-1.0, -0.1, 0.0, 0.1, 1.0], dtype=np.float32)
        grad = fn.gradient(x)
        # All gradients should be positive
        assert np.all(grad > 0)
        # Peak at x=0
        assert grad[2] >= grad[0]
        assert grad[2] >= grad[4]

    def test_sigmoid_forward_binary(self):
        """Sigmoid forward should produce binary output."""
        from grilly.nn.snn_surrogate import Sigmoid

        fn = Sigmoid(alpha=4.0)
        x = np.array([-2.0, 0.5, 1.0], dtype=np.float32)
        out = fn(x)
        expected = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_sigmoid_gradient_smooth(self):
        """Sigmoid backward gradient is smooth and non-zero."""
        from grilly.nn.snn_surrogate import Sigmoid

        fn = Sigmoid(alpha=4.0)
        x = np.linspace(-2, 2, 100).astype(np.float32)
        grad = fn.gradient(x)
        assert np.all(grad >= 0)
        assert np.max(grad) > 0

    def test_fast_sigmoid_forward_binary(self):
        """FastSigmoid forward produces binary output."""
        from grilly.nn.snn_surrogate import FastSigmoid

        fn = FastSigmoid(alpha=2.0)
        x = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
        out = fn(x)
        expected = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_fast_sigmoid_gradient_smooth(self):
        """FastSigmoid backward gradient is smooth."""
        from grilly.nn.snn_surrogate import FastSigmoid

        fn = FastSigmoid(alpha=2.0)
        x = np.linspace(-2, 2, 100).astype(np.float32)
        grad = fn.gradient(x)
        assert np.all(grad >= 0)
        assert np.max(grad) > 0

    def test_surrogate_repr(self):
        from grilly.nn.snn_surrogate import ATan

        fn = ATan(alpha=3.0)
        assert "ATan" in repr(fn)
        assert "3.0" in repr(fn)


class TestMemoryModule:
    """Test MemoryModule state management."""

    def test_register_and_reset(self):
        """MemoryModule should reset registered memory to defaults."""
        from grilly.nn.snn_base import MemoryModule

        mod = MemoryModule()
        mod.register_memory("state", np.zeros(5, dtype=np.float32))
        mod.state = np.ones(5, dtype=np.float32)
        assert np.all(mod.state == 1.0)

        mod.reset()
        assert np.all(mod.state == 0.0)

    def test_reset_none_memory(self):
        """MemoryModule resets None memory to None."""
        from grilly.nn.snn_base import MemoryModule

        mod = MemoryModule()
        mod.register_memory("v", None)
        mod.v = np.array([1.0])
        mod.reset()
        assert mod.v is None

    def test_detach(self):
        """Detach should copy memory arrays."""
        from grilly.nn.snn_base import MemoryModule

        mod = MemoryModule()
        mod.register_memory("state", np.zeros(5, dtype=np.float32))
        mod.state = np.ones(5, dtype=np.float32)
        original_id = id(mod.state)
        mod.detach()
        # After detach, should be a new array (copied)
        assert id(mod.state) != original_id
        assert np.all(mod.state == 1.0)


class TestBaseNode:
    """Test BaseNode spiking neuron base class."""

    def test_single_step_produces_binary(self):
        """Single step should produce binary spikes."""
        from grilly.nn.snn_base import BaseNode

        class SimpleNode(BaseNode):
            def neuronal_charge(self, x):
                self.h = self.v + x

        node = SimpleNode(v_threshold=1.0, v_reset=0.0, step_mode="s")
        x = np.array([[0.5, 1.5, 0.3]], dtype=np.float32)  # (1, 3)
        spike = node(x)
        assert spike.shape == (1, 3)
        # Only neuron with cumulative input >= threshold should spike
        assert spike[0, 0] == 0.0  # 0.5 < 1.0
        assert spike[0, 1] == 1.0  # 1.5 >= 1.0
        assert spike[0, 2] == 0.0  # 0.3 < 1.0

    def test_multi_step_iterates_T(self):
        """Multi-step should iterate T and stack outputs."""
        from grilly.nn.snn_base import BaseNode

        class SimpleNode(BaseNode):
            def neuronal_charge(self, x):
                self.h = self.v + x

        node = SimpleNode(v_threshold=1.0, v_reset=0.0, step_mode="m")
        T, N, D = 4, 2, 3
        x_seq = np.random.rand(T, N, D).astype(np.float32) * 0.4
        spikes = node(x_seq)
        assert spikes.shape == (T, N, D)
        # All values should be 0 or 1
        assert set(np.unique(spikes)).issubset({0.0, 1.0})

    def test_hard_reset(self):
        """Hard reset should set V to v_reset after spike."""
        from grilly.nn.snn_base import BaseNode

        class SimpleNode(BaseNode):
            def neuronal_charge(self, x):
                self.h = self.v + x

        node = SimpleNode(v_threshold=1.0, v_reset=0.0, step_mode="s")
        x = np.array([[2.0]], dtype=np.float32)  # Will spike
        node(x)
        # After hard reset, v should be v_reset (0.0)
        assert node.v[0, 0] == pytest.approx(0.0)

    def test_soft_reset(self):
        """Soft reset should subtract threshold from V after spike."""
        from grilly.nn.snn_base import BaseNode

        class SimpleNode(BaseNode):
            def neuronal_charge(self, x):
                self.h = self.v + x

        node = SimpleNode(v_threshold=1.0, v_reset=None, step_mode="s")
        x = np.array([[2.0]], dtype=np.float32)  # h = 2.0, spike, v = 2.0 - 1.0 = 1.0
        node(x)
        assert node.v[0, 0] == pytest.approx(1.0)

    def test_reset_clears_v(self):
        """Reset should clear membrane potential."""
        from grilly.nn.snn_base import BaseNode

        class SimpleNode(BaseNode):
            def neuronal_charge(self, x):
                self.h = self.v + x

        node = SimpleNode(v_threshold=1.0, v_reset=0.0, step_mode="s")
        x = np.array([[0.5]], dtype=np.float32)
        node(x)  # v accumulates
        assert node.v is not None
        node.reset()
        assert node.v is None  # Back to default (None = lazy init)

    def test_invalid_step_mode(self):
        """Invalid step_mode should raise ValueError."""
        from grilly.nn.snn_base import BaseNode

        class SimpleNode(BaseNode):
            def neuronal_charge(self, x):
                self.h = self.v + x

        node = SimpleNode(step_mode="x")
        with pytest.raises(ValueError, match="Invalid step_mode"):
            node(np.array([[1.0]], dtype=np.float32))
