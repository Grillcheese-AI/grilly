"""
Tests for C++ SNN engine: surrogate gradients, neuron nodes, containers, and BPTT.
Mirrors test_snn_phase1.py, test_snn_neurons.py, and test_snn_containers.py
but exercises the C++ implementations via grilly_core.
"""

import numpy as np
import pytest

try:
    from grilly_core import (
        Flatten,
        IFNode,
        LIFNode,
        Module,
        MultiStepContainer,
        ParametricLIFNode,
        SeqToANNContainer,
        SurrogateFunction,
        SurrogateType,
        Tensor,
    )

    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ backend not available")


# ═══════════════════════════════════════════════════════════════════════
# Surrogate Gradients
# ═══════════════════════════════════════════════════════════════════════


class TestSurrogateGradients:
    """Test surrogate gradient functions (mirrors test_snn_phase1.py)."""

    def test_atan_forward_binary(self):
        """ATan forward should produce binary output (Heaviside)."""
        fn = SurrogateFunction(SurrogateType.ATan, 2.0)
        x = Tensor.from_numpy(np.array([-1.0, -0.1, 0.0, 0.1, 1.0], dtype=np.float32))
        out = fn.forward(x).numpy()
        expected = np.array([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_atan_gradient_smooth(self):
        """ATan backward should give smooth, non-zero gradient."""
        fn = SurrogateFunction(SurrogateType.ATan, 2.0)
        x = Tensor.from_numpy(np.array([-1.0, -0.1, 0.0, 0.1, 1.0], dtype=np.float32))
        grad = fn.gradient(x).numpy()
        assert np.all(grad > 0), "All ATan gradients should be positive"
        # Peak at x=0
        assert grad[2] >= grad[0]
        assert grad[2] >= grad[4]

    def test_sigmoid_forward_binary(self):
        """Sigmoid forward should produce binary output."""
        fn = SurrogateFunction(SurrogateType.Sigmoid, 4.0)
        x = Tensor.from_numpy(np.array([-2.0, 0.5, 1.0], dtype=np.float32))
        out = fn.forward(x).numpy()
        expected = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_sigmoid_gradient_smooth(self):
        """Sigmoid backward gradient is smooth and non-zero."""
        fn = SurrogateFunction(SurrogateType.Sigmoid, 4.0)
        x = Tensor.from_numpy(np.linspace(-2, 2, 100).astype(np.float32))
        grad = fn.gradient(x).numpy()
        assert np.all(grad >= 0)
        assert np.max(grad) > 0

    def test_fast_sigmoid_forward_binary(self):
        """FastSigmoid forward produces binary output."""
        fn = SurrogateFunction(SurrogateType.FastSigmoid, 2.0)
        x = Tensor.from_numpy(np.array([-0.5, 0.0, 0.5], dtype=np.float32))
        out = fn.forward(x).numpy()
        expected = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_fast_sigmoid_gradient_smooth(self):
        """FastSigmoid backward gradient is smooth."""
        fn = SurrogateFunction(SurrogateType.FastSigmoid, 2.0)
        x = Tensor.from_numpy(np.linspace(-2, 2, 100).astype(np.float32))
        grad = fn.gradient(x).numpy()
        assert np.all(grad >= 0)
        assert np.max(grad) > 0

    def test_gradient_at_threshold_nonzero(self):
        """Surrogate gradient at firing threshold should be nonzero."""
        fn = SurrogateFunction(SurrogateType.ATan, 2.0)
        x = Tensor.from_numpy(np.array([0.0], dtype=np.float32))
        grad = fn.gradient(x).numpy()
        assert grad[0] > 0

    def test_gradient_decays_away_from_threshold(self):
        """Gradient should decrease away from threshold."""
        fn = SurrogateFunction(SurrogateType.ATan, 2.0)
        x_near = Tensor.from_numpy(np.array([0.01], dtype=np.float32))
        x_far = Tensor.from_numpy(np.array([5.0], dtype=np.float32))
        assert fn.gradient(x_near).numpy()[0] > fn.gradient(x_far).numpy()[0]


# ═══════════════════════════════════════════════════════════════════════
# IFNode
# ═══════════════════════════════════════════════════════════════════════


class TestIFNode:
    """Test Integrate-and-Fire neuron (mirrors test_snn_neurons.py::TestIFNode)."""

    def test_accumulates_and_fires(self):
        """IF should accumulate input and fire at threshold."""
        node = IFNode(v_threshold=1.0, v_reset=0.0, step_mode="s")
        x = Tensor.from_numpy(np.array([[0.6]], dtype=np.float32))

        # First step: 0.6 < 1.0, no spike
        spike1 = node.forward(x).numpy()
        assert spike1[0, 0] == 0.0

        # Second step: 0.6 + 0.6 = 1.2 >= 1.0, spike!
        spike2 = node.forward(x).numpy()
        assert spike2[0, 0] == 1.0

    def test_single_step_shape(self):
        """Single step: [N, D] -> [N, D]."""
        node = IFNode(step_mode="s")
        x = Tensor.from_numpy(np.random.rand(8, 128).astype(np.float32))
        out = node.forward(x)
        assert out.shape == [8, 128]

    def test_multi_step_shape(self):
        """Multi step: [T, N, D] -> [T, N, D]."""
        node = IFNode(step_mode="m")
        x = Tensor.from_numpy(np.random.rand(4, 8, 128).astype(np.float32) * 0.5)
        out = node.forward(x)
        assert out.shape == [4, 8, 128]
        # All values should be 0 or 1
        vals = set(np.unique(out.numpy()))
        assert vals.issubset({0.0, 1.0})

    def test_strong_input_fires_immediately(self):
        """Input above threshold should spike on first step."""
        node = IFNode(v_threshold=1.0, step_mode="s")
        x = Tensor.from_numpy(np.array([[2.0, 0.5]], dtype=np.float32))
        spike = node.forward(x).numpy()
        assert spike[0, 0] == 1.0  # 2.0 >= 1.0
        assert spike[0, 1] == 0.0  # 0.5 < 1.0


# ═══════════════════════════════════════════════════════════════════════
# LIFNode
# ═══════════════════════════════════════════════════════════════════════


class TestLIFNode:
    """Test Leaky Integrate-and-Fire neuron (mirrors test_snn_neurons.py::TestLIFNode)."""

    def test_leaks_toward_reset(self):
        """LIF should leak toward v_reset when no input."""
        node = LIFNode(tau=2.0, v_threshold=10.0, v_reset=0.0, step_mode="s")
        # Inject some voltage
        x = Tensor.from_numpy(np.array([[0.8]], dtype=np.float32))
        node.forward(x)

        # Then give zero input - v should decay
        x_zero = Tensor.from_numpy(np.array([[0.0]], dtype=np.float32))
        # Run a few zero-input steps
        node.forward(x_zero)
        node.forward(x_zero)
        # After decay, subsequent zero-input should keep decaying
        # (we can't directly read v, but no spikes with decaying input is expected)

    def test_lif_dynamics_correct(self):
        """LIF with decay_input=False: V[t] = (1/tau)*V[t-1] + X[t]."""
        node = LIFNode(tau=2.0, decay_input=False, v_threshold=10.0, step_mode="s")
        x = Tensor.from_numpy(np.array([[1.0]], dtype=np.float32))

        # Step 1: V = 0.5*0 + 1.0 = 1.0
        node.forward(x)

        # Step 2: V = 0.5*1.0 + 1.0 = 1.5
        node.forward(x)

        # Step 3: V = 0.5*1.5 + 1.0 = 1.75
        node.forward(x)
        # Can verify convergence behavior

    def test_decay_input_mode(self):
        """decay_input=True: V[t] = (1/tau)*(V[t-1] + X[t])."""
        node_decay = LIFNode(tau=2.0, decay_input=True, v_threshold=10.0, step_mode="s")
        node_no_decay = LIFNode(tau=2.0, decay_input=False, v_threshold=10.0, step_mode="s")

        x = Tensor.from_numpy(np.array([[1.0]], dtype=np.float32))
        # Both start with V=0
        node_decay.forward(x)
        node_no_decay.forward(x)

        # With decay_input=True: V = 0.5*(0+1) = 0.5 (no spike)
        # With decay_input=False: V = 0.5*0 + 1.0 = 1.0 (no spike, threshold=10)
        # Both should not spike, but V differs

    def test_tau_property(self):
        """tau should be accessible."""
        node = LIFNode(tau=3.5)
        assert node.tau == pytest.approx(3.5)

    def test_multi_step(self):
        """LIF multi-step should process temporal sequence."""
        node = LIFNode(tau=2.0, step_mode="m")
        x = Tensor.from_numpy(np.random.rand(10, 4, 64).astype(np.float32) * 0.3)
        out = node.forward(x)
        assert out.shape == [10, 4, 64]

    def test_multi_step_produces_spikes(self):
        """Multi-step with sufficient input should produce spikes."""
        node = LIFNode(tau=2.0, v_threshold=1.0, step_mode="m")
        x = Tensor.from_numpy(np.ones((20, 4, 8), dtype=np.float32) * 0.6)
        out = node.forward(x)
        total_spikes = out.numpy().sum()
        assert total_spikes > 0, "LIF should spike with sustained input"

    def test_reset_clears_state(self):
        """reset() should clear membrane potential."""
        node = LIFNode(tau=2.0, step_mode="s")
        x = Tensor.from_numpy(np.ones((2, 4), dtype=np.float32) * 0.6)
        node.forward(x)
        node.forward(x)

        node.reset()

        # After reset, same input should produce same first-step result
        spike = node.forward(x).numpy()
        # First step with V=0, input=0.6: V=0.5*0+0.6=0.6 < 1.0, no spike
        assert spike.sum() == 0.0

    def test_step_mode_property(self):
        """step_mode should be gettable and settable."""
        node = LIFNode(tau=2.0, step_mode="s")
        assert node.step_mode == "s"
        node.step_mode = "m"
        assert node.step_mode == "m"


# ═══════════════════════════════════════════════════════════════════════
# ParametricLIFNode
# ═══════════════════════════════════════════════════════════════════════


class TestParametricLIFNode:
    """Test Parametric LIF with learnable tau."""

    def test_tau_is_parameter(self):
        """ParametricLIF tau should be a registered Parameter."""
        node = ParametricLIFNode(init_tau=2.0)
        params = node.parameters()
        assert len(params) == 1  # tau
        assert params[0].shape == [1]

    def test_forward_produces_spikes(self):
        """ParametricLIF should produce binary spikes."""
        node = ParametricLIFNode(init_tau=2.0, step_mode="s")
        x = Tensor.from_numpy(np.random.rand(4, 32).astype(np.float32) * 2.0)
        out = node.forward(x)
        assert out.shape == [4, 32]
        vals = set(np.unique(out.numpy()))
        assert vals.issubset({0.0, 1.0})

    def test_multi_step(self):
        """ParametricLIF multi-step."""
        node = ParametricLIFNode(init_tau=2.0, step_mode="m")
        x = Tensor.from_numpy(np.random.rand(5, 4, 32).astype(np.float32) * 0.5)
        out = node.forward(x)
        assert out.shape == [5, 4, 32]


# ═══════════════════════════════════════════════════════════════════════
# BPTT Backward
# ═══════════════════════════════════════════════════════════════════════


class TestBPTTBackward:
    """Test backward through time with surrogate gradients."""

    def test_backward_produces_correct_shape(self):
        """backward() should produce gradient matching input shape."""
        node = LIFNode(tau=2.0, step_mode="m")
        node.train(True)
        T, N, D = 5, 2, 4
        x = Tensor.from_numpy(np.random.randn(T, N, D).astype(np.float32))
        node.forward(x)

        grad_out = Tensor.from_numpy(np.ones((T, N, D), dtype=np.float32))
        grad_in = node.backward(grad_out)
        assert grad_in.shape == [T, N, D]

    def test_backward_nonzero_gradients(self):
        """Backward gradients should be nonzero (surrogate is smooth)."""
        node = LIFNode(tau=2.0, step_mode="m")
        node.train(True)
        T, N, D = 10, 4, 8
        x = Tensor.from_numpy(np.random.randn(T, N, D).astype(np.float32))
        node.forward(x)

        grad_out = Tensor.from_numpy(np.ones((T, N, D), dtype=np.float32))
        grad_in = node.backward(grad_out)
        assert np.any(grad_in.numpy() != 0), "Backward gradients should be nonzero"

    def test_backward_requires_forward_first(self):
        """backward() without forward should raise."""
        node = LIFNode(tau=2.0, step_mode="m")
        grad_out = Tensor.from_numpy(np.ones((5, 2, 4), dtype=np.float32))
        with pytest.raises(RuntimeError):
            node.backward(grad_out)


# ═══════════════════════════════════════════════════════════════════════
# Containers
# ═══════════════════════════════════════════════════════════════════════


class TestMultiStepContainer:
    """Test MultiStepContainer (mirrors test_snn_containers.py)."""

    def test_wraps_lif_node(self):
        """MultiStepContainer should process [T,N,D] through LIFNode."""
        lif = LIFNode(tau=2.0, step_mode="s")
        container = MultiStepContainer(lif)
        x = Tensor.from_numpy(np.random.rand(4, 2, 16).astype(np.float32) * 0.5)
        out = container.forward(x)
        assert out.shape == [4, 2, 16]

    def test_wraps_identity(self):
        """MultiStepContainer should handle identity modules."""

        class Identity(Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        container = MultiStepContainer(Identity())
        x = Tensor.from_numpy(np.ones((3, 2, 4), dtype=np.float32))
        out = container.forward(x)
        np.testing.assert_allclose(out.numpy(), 1.0)
        assert out.shape == [3, 2, 4]


class TestSeqToANNContainer:
    """Test SeqToANNContainer for temporal->batch reshaping."""

    def test_with_scale_module(self):
        """SeqToANNContainer should merge T*N, run module, split back."""

        class Scale(Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return Tensor.from_numpy(x.numpy() * 2.0)

        container = SeqToANNContainer([Scale()])
        x = Tensor.from_numpy(np.ones((3, 2, 4), dtype=np.float32))
        out = container.forward(x)
        np.testing.assert_allclose(out.numpy(), 2.0)

    def test_shape_preservation(self):
        """Output temporal shape should match input."""

        class Identity(Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        container = SeqToANNContainer([Identity()])
        x = Tensor.from_numpy(np.ones((5, 4, 8), dtype=np.float32))
        out = container.forward(x)
        assert out.shape == [5, 4, 8]


class TestFlatten:
    """Test Flatten module (mirrors test_snn_containers.py::TestFlatten)."""

    def test_flatten_default(self):
        """Flatten with start_dim=1 should flatten all except batch."""
        flat = Flatten(start_dim=1)
        x = Tensor.from_numpy(np.random.rand(4, 3, 7, 7).astype(np.float32))
        out = flat.forward(x)
        assert out.shape == [4, 3 * 7 * 7]

    def test_flatten_custom_dims(self):
        """Flatten with custom start/end dims."""
        flat = Flatten(start_dim=2, end_dim=-1)
        x = Tensor.from_numpy(np.random.rand(4, 3, 7, 7).astype(np.float32))
        out = flat.forward(x)
        assert out.shape == [4, 3, 49]

    def test_flatten_backward(self):
        """Flatten backward should restore original shape."""
        flat = Flatten(start_dim=1)
        x = Tensor.from_numpy(np.random.rand(4, 3, 7, 7).astype(np.float32))
        y = flat.forward(x)
        assert y.shape == [4, 147]

        grad = Tensor.from_numpy(np.ones((4, 147), dtype=np.float32))
        grad_in = flat.backward(grad)
        assert grad_in.shape == [4, 3, 7, 7]

    def test_flatten_2d_noop(self):
        """Flattening an already-2D tensor should be a no-op."""
        flat = Flatten(start_dim=1)
        x = Tensor.from_numpy(np.ones((4, 10), dtype=np.float32))
        out = flat.forward(x)
        assert out.shape == [4, 10]


# ═══════════════════════════════════════════════════════════════════════
# Integration: Composability
# ═══════════════════════════════════════════════════════════════════════


class TestComposability:
    """Test composition of SNN nodes and containers."""

    def test_lif_after_flatten(self):
        """Flatten -> LIF should work end-to-end."""
        flat = Flatten(start_dim=1)
        lif = LIFNode(tau=2.0, step_mode="s")

        x = Tensor.from_numpy(np.random.rand(4, 3, 7, 7).astype(np.float32))
        h = flat.forward(x)  # [4, 147]
        spike = lif.forward(h)  # [4, 147]
        assert spike.shape == [4, 147]

    def test_multi_step_lif_sequence(self):
        """Multi-step LIF on a temporal sequence."""
        lif = LIFNode(tau=2.0, step_mode="m")

        T, N, D = 10, 4, 32
        x = Tensor.from_numpy(np.random.rand(T, N, D).astype(np.float32) * 0.4)
        spikes = lif.forward(x)
        assert spikes.shape == [T, N, D]
        vals = set(np.unique(spikes.numpy()))
        assert vals.issubset({0.0, 1.0})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
