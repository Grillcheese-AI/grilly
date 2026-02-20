"""Tests for SNN Synapses — temporal filtering, dual-timescale, STP, GPU dispatch."""

import numpy as np
import pytest


class TestElementWiseRecurrentContainer:
    """Test recurrent container for spiking neurons."""

    def test_feeds_back_spikes(self):
        """Recurrent container should feed output back as input."""
        from grilly.nn.snn_neurons import IFNode
        from grilly.nn.snn_synapses import ElementWiseRecurrentContainer

        node = IFNode(v_threshold=1.0, v_reset=0.0, step_mode="s")
        rec = ElementWiseRecurrentContainer(node)

        x = np.array([[0.6]], dtype=np.float32)
        # Step 1: 0.6 + 0 (no feedback) = 0.6, no spike
        out1 = rec(x)
        assert out1[0, 0] == 0.0

        # Step 2: 0.6 + 0 (feedback=0) = 0.6 + accumulated v
        rec(x)  # v should have accumulated from step 1

    def test_custom_function(self):
        """Recurrent container with custom combining function."""
        from grilly.nn.snn_neurons import IFNode
        from grilly.nn.snn_synapses import ElementWiseRecurrentContainer

        node = IFNode(v_threshold=1.0, v_reset=0.0, step_mode="s")
        # Multiply instead of add
        rec = ElementWiseRecurrentContainer(
            node, element_wise_function=lambda x, y: x * 0.5 + y * 0.5
        )

        x = np.array([[2.0]], dtype=np.float32)
        out = rec(x)
        assert out.shape == (1, 1)

    def test_reset(self):
        """Reset should clear feedback state."""
        from grilly.nn.snn_neurons import IFNode
        from grilly.nn.snn_synapses import ElementWiseRecurrentContainer

        node = IFNode(v_threshold=1.0, v_reset=0.0, step_mode="s")
        rec = ElementWiseRecurrentContainer(node)
        x = np.array([[0.6]], dtype=np.float32)
        rec(x)
        assert rec.y_prev is not None
        rec.reset()
        assert rec.y_prev is None


class TestSynapseFilter:
    """Test exponential decay synaptic filter."""

    def test_exponential_decay(self):
        """Filter should decay toward zero with no input."""
        from grilly.nn.snn_synapses import SynapseFilter

        sf = SynapseFilter(tau=2.0, use_gpu=False)
        # Inject pulse
        x = np.array([[1.0]], dtype=np.float32)
        out1 = sf(x)
        assert out1[0, 0] == pytest.approx(1.0)

        # No input - should decay
        x_zero = np.array([[0.0]], dtype=np.float32)
        out2 = sf(x_zero)
        assert out2[0, 0] < out1[0, 0]
        assert out2[0, 0] > 0

        # Continued decay
        out3 = sf(x_zero)
        assert out3[0, 0] < out2[0, 0]

    def test_exact_decay_value(self):
        """Verify exact exponential decay computation."""
        from grilly.nn.snn_synapses import SynapseFilter

        sf = SynapseFilter(tau=5.0, use_gpu=False)
        x = np.array([[1.0]], dtype=np.float32)
        out1 = sf(x)
        assert out1[0, 0] == pytest.approx(1.0)

        x_zero = np.array([[0.0]], dtype=np.float32)
        out2 = sf(x_zero)
        decay = np.exp(-1.0 / 5.0)
        assert out2[0, 0] == pytest.approx(decay, abs=1e-6)

    def test_learnable_tau(self):
        """SynapseFilter with learnable tau should have parameter."""
        from grilly.nn.snn_synapses import SynapseFilter

        sf = SynapseFilter(tau=5.0, learnable=True, use_gpu=False)
        params = list(sf.parameters())
        assert len(params) == 1
        assert params[0].requires_grad is True

    def test_shape_preserved(self):
        """Filter should preserve input shape."""
        from grilly.nn.snn_synapses import SynapseFilter

        sf = SynapseFilter(tau=2.0, use_gpu=False)
        x = np.random.rand(4, 32).astype(np.float32)
        out = sf(x)
        assert out.shape == (4, 32)

    def test_reset(self):
        """Reset should clear filter state."""
        from grilly.nn.snn_synapses import SynapseFilter

        sf = SynapseFilter(tau=2.0, use_gpu=False)
        sf(np.ones((2, 4), dtype=np.float32))
        assert sf.y is not None
        sf.reset()
        assert sf.y is None

    def test_backward(self):
        """Backward pass should pass gradients through."""
        from grilly.nn.snn_synapses import SynapseFilter

        sf = SynapseFilter(tau=2.0, use_gpu=False)
        x = np.random.rand(4, 8).astype(np.float32)
        sf(x)

        grad = np.ones((4, 8), dtype=np.float32)
        grad_input = sf.backward(grad)
        assert grad_input.shape == (4, 8)
        assert np.allclose(grad_input, grad)

    def test_accumulation(self):
        """Repeated input should accumulate in filter state."""
        from grilly.nn.snn_synapses import SynapseFilter

        sf = SynapseFilter(tau=10.0, use_gpu=False)
        x = np.array([[1.0]], dtype=np.float32)
        out1 = sf(x)
        out2 = sf(x)
        out3 = sf(x)
        # Each step adds 1.0 and decays, so output grows
        assert out3[0, 0] > out2[0, 0] > out1[0, 0]


class TestSynapseFilterGPU:
    """Test GPU-accelerated synapse filter."""

    @pytest.fixture
    def gpu_available(self):
        """Set up shared GPU compute instance."""
        try:
            from grilly import Compute
            from grilly.nn.snn_synapses import set_gpu_compute

            compute = Compute()
            set_gpu_compute(compute)
            return compute.snn
        except Exception:
            pytest.skip("Vulkan GPU not available")

    def test_gpu_matches_cpu(self, gpu_available):
        """GPU and CPU synapse filter should produce identical results."""
        from grilly.nn.snn_synapses import SynapseFilter

        np.random.seed(42)
        x = np.random.rand(64, 128).astype(np.float32)

        sf_cpu = SynapseFilter(tau=3.0, use_gpu=False)
        sf_gpu = SynapseFilter(tau=3.0, use_gpu=True)

        # Step 1: initial input
        out_cpu = sf_cpu(x)
        out_gpu = sf_gpu(x)
        assert np.allclose(out_cpu, out_gpu, atol=1e-5)

        # Step 2: decay with new input
        x2 = np.random.rand(64, 128).astype(np.float32)
        out_cpu2 = sf_cpu(x2)
        out_gpu2 = sf_gpu(x2)
        assert np.allclose(out_cpu2, out_gpu2, atol=1e-5)

    def test_gpu_repr(self, gpu_available):
        """GPU filter repr should indicate gpu backend."""
        from grilly.nn.snn_synapses import SynapseFilter

        sf = SynapseFilter(tau=2.0, use_gpu=True)
        sf(np.ones((2, 4), dtype=np.float32))  # triggers lazy init
        assert "gpu" in repr(sf)


class TestDualTimescaleSynapse:
    """Test dual-timescale (fast+slow) synaptic filter."""

    def test_output_shape(self):
        """Output shape should match input."""
        from grilly.nn.snn_synapses import DualTimescaleSynapse

        syn = DualTimescaleSynapse(tau_fast=2.0, tau_slow=20.0, use_gpu=False)
        x = np.random.rand(4, 32).astype(np.float32)
        out = syn(x)
        assert out.shape == (4, 32)

    def test_fast_decays_faster(self):
        """Fast component should decay faster than slow."""
        from grilly.nn.snn_synapses import DualTimescaleSynapse

        syn = DualTimescaleSynapse(tau_fast=2.0, tau_slow=50.0, w_fast=1.0, w_slow=0.0,
                                   use_gpu=False)
        x = np.array([[1.0]], dtype=np.float32)
        syn(x)
        x_zero = np.array([[0.0]], dtype=np.float32)
        fast_decay = syn(x_zero)[0, 0]

        syn2 = DualTimescaleSynapse(tau_fast=2.0, tau_slow=50.0, w_fast=0.0, w_slow=1.0,
                                    use_gpu=False)
        syn2(x)
        slow_decay = syn2(x_zero)[0, 0]

        # Slow component retains more after one step
        assert slow_decay > fast_decay

    def test_both_components_contribute(self):
        """Output should be weighted sum of fast and slow."""
        from grilly.nn.snn_synapses import DualTimescaleSynapse

        syn = DualTimescaleSynapse(tau_fast=2.0, tau_slow=20.0, w_fast=0.7, w_slow=0.3,
                                   use_gpu=False)
        x = np.array([[1.0]], dtype=np.float32)
        out = syn(x)
        # First step: y_fast = 1.0, y_slow = 1.0
        # output = 0.7*1.0 + 0.3*1.0 = 1.0
        assert out[0, 0] == pytest.approx(1.0)

    def test_reset(self):
        """Reset should clear both fast and slow states."""
        from grilly.nn.snn_synapses import DualTimescaleSynapse

        syn = DualTimescaleSynapse(use_gpu=False)
        syn(np.ones((2, 4), dtype=np.float32))
        assert syn.y_fast is not None
        assert syn.y_slow is not None
        syn.reset()
        assert syn.y_fast is None
        assert syn.y_slow is None

    def test_learnable_parameters(self):
        """Learnable mode should have 4 parameters."""
        from grilly.nn.snn_synapses import DualTimescaleSynapse

        syn = DualTimescaleSynapse(learnable=True, use_gpu=False)
        params = list(syn.parameters())
        assert len(params) == 4  # tau_fast, tau_slow, w_fast, w_slow

    def test_backward(self):
        """Backward should pass gradients through."""
        from grilly.nn.snn_synapses import DualTimescaleSynapse

        syn = DualTimescaleSynapse(w_fast=0.7, w_slow=0.3, use_gpu=False)
        x = np.random.rand(4, 8).astype(np.float32)
        syn(x)

        grad = np.ones((4, 8), dtype=np.float32)
        grad_input = syn.backward(grad)
        assert grad_input.shape == (4, 8)
        # gradient scaled by (w_fast + w_slow) = 1.0
        assert np.allclose(grad_input, grad)

    def test_repr(self):
        """Repr should show both timescales."""
        from grilly.nn.snn_synapses import DualTimescaleSynapse

        syn = DualTimescaleSynapse(tau_fast=3.0, tau_slow=30.0)
        r = repr(syn)
        assert "tau_fast=3.0" in r
        assert "tau_slow=30.0" in r


class TestSTPSynapse:
    """Test Short-Term Plasticity synapse."""

    def test_output_shape(self):
        """Output shape should match input."""
        from grilly.nn.snn_synapses import STPSynapse

        stp = STPSynapse()
        x = np.random.rand(4, 32).astype(np.float32)
        out = stp(x)
        assert out.shape == (4, 32)

    def test_depression_reduces_output(self):
        """Repeated spiking should depress output (STD)."""
        from grilly.nn.snn_synapses import STPSynapse

        stp = STPSynapse(U=0.5, tau_f=5.0, tau_d=100.0)
        x = np.ones((1, 4), dtype=np.float32)

        out1 = stp(x)
        stp(x)  # step 2: deplete resources
        out3 = stp(x)

        # Depression: repeated spikes deplete resources
        # After many spikes, output should decrease
        mean1 = out1.mean()
        mean3 = out3.mean()
        assert mean3 < mean1  # Depressed after repeated activation

    def test_recovery_after_silence(self):
        """Resources should recover during silence."""
        from grilly.nn.snn_synapses import STPSynapse

        stp = STPSynapse(U=0.5, tau_f=5.0, tau_d=10.0)
        x = np.ones((1, 4), dtype=np.float32)
        x_zero = np.zeros((1, 4), dtype=np.float32)

        # Deplete with spikes
        stp(x)
        stp(x)
        stp(x)

        # Recover during silence (many steps)
        for _ in range(50):
            stp(x_zero)

        # After recovery, r should be closer to 1.0
        assert np.mean(stp.r) > 0.5

    def test_reset(self):
        """Reset should clear STP state."""
        from grilly.nn.snn_synapses import STPSynapse

        stp = STPSynapse()
        stp(np.ones((2, 4), dtype=np.float32))
        assert stp.u is not None
        assert stp.r is not None
        stp.reset()
        assert stp.u is None
        assert stp.r is None

    def test_no_spike_passthrough(self):
        """With zero input (no spikes), u and r should stay near initial."""
        from grilly.nn.snn_synapses import STPSynapse

        stp = STPSynapse(U=0.2)
        x = np.zeros((1, 4), dtype=np.float32)
        out = stp(x)
        # Zero input * u * r = zero
        assert np.allclose(out, 0.0)

    def test_backward(self):
        """Backward should scale gradients by u*r."""
        from grilly.nn.snn_synapses import STPSynapse

        stp = STPSynapse(U=0.3)
        x = np.ones((2, 4), dtype=np.float32)
        stp(x)

        grad = np.ones((2, 4), dtype=np.float32)
        grad_input = stp.backward(grad)
        assert grad_input.shape == (2, 4)
        # Should be scaled by u*r
        expected = stp.u * stp.r
        assert np.allclose(grad_input, expected, atol=1e-5)

    def test_repr(self):
        """Repr should show STP parameters."""
        from grilly.nn.snn_synapses import STPSynapse

        stp = STPSynapse(U=0.3, tau_f=15.0, tau_d=150.0)
        r = repr(stp)
        assert "U=0.3" in r
        assert "tau_f=15.0" in r
        assert "tau_d=150.0" in r
