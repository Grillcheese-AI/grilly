"""Tests for Phase 6: SNN Functional API"""

import numpy as np
import pytest


class TestLIFStep:
    """Test functional lif_step."""

    def test_basic_lif_step(self):
        """lif_step should return spike and new v."""
        from grilly.functional.snn import lif_step

        x = np.array([[1.5]], dtype=np.float32)
        v = np.array([[0.0]], dtype=np.float32)
        spike, v_new = lif_step(x, v, tau=2.0, v_threshold=1.0, v_reset=0.0)
        # h = 0 + (1.5 - 0) / 2 = 0.75 < 1.0 => no spike
        assert spike[0, 0] == 0.0
        assert v_new[0, 0] == pytest.approx(0.75)

    def test_lif_step_fires(self):
        """lif_step should fire when over threshold."""
        from grilly.functional.snn import lif_step

        x = np.array([[3.0]], dtype=np.float32)
        v = np.array([[0.0]], dtype=np.float32)
        spike, v_new = lif_step(x, v, tau=2.0, v_threshold=1.0, v_reset=0.0)
        # h = 0 + 3.0/2 = 1.5 >= 1.0 => spike
        assert spike[0, 0] == 1.0
        # Hard reset: v = h*(1-s) + v_reset*s = 0.0
        assert v_new[0, 0] == pytest.approx(0.0)

    def test_lif_step_soft_reset(self):
        """lif_step with v_reset=None uses soft reset."""
        from grilly.functional.snn import lif_step

        x = np.array([[3.0]], dtype=np.float32)
        v = np.array([[0.0]], dtype=np.float32)
        spike, v_new = lif_step(x, v, tau=2.0, v_threshold=1.0, v_reset=None)
        # h = 1.5, spike, soft reset: v = 1.5 - 1.0 = 0.5
        assert spike[0, 0] == 1.0
        assert v_new[0, 0] == pytest.approx(0.5)


class TestIFStep:
    """Test functional if_step."""

    def test_basic_if_step(self):
        """if_step should accumulate and fire."""
        from grilly.functional.snn import if_step

        x = np.array([[0.6]], dtype=np.float32)
        v = np.array([[0.5]], dtype=np.float32)
        spike, v_new = if_step(x, v, v_threshold=1.0, v_reset=0.0)
        # h = 0.5 + 0.6 = 1.1 >= 1.0 => spike
        assert spike[0, 0] == 1.0


class TestMultiStepForward:
    """Test functional multi_step_forward."""

    def test_processes_sequence(self):
        """multi_step_forward should loop over T dimension."""
        from grilly.functional.snn import multi_step_forward
        from grilly.nn.snn_neurons import IFNode

        node = IFNode(v_threshold=1.0, step_mode="s")
        x = np.random.rand(5, 2, 8).astype(np.float32) * 0.3
        out = multi_step_forward(x, node)
        assert out.shape == (5, 2, 8)


class TestSeqToANNForward:
    """Test functional seq_to_ann_forward."""

    def test_reshape_correct(self):
        """seq_to_ann_forward should merge/split T and N."""
        from grilly.functional.snn import seq_to_ann_forward
        from grilly.nn.module import Module

        class Double(Module):
            def forward(self, x):
                return x * 2.0

        x = np.ones((3, 4, 8), dtype=np.float32)
        out = seq_to_ann_forward(x, Double())
        assert out.shape == (3, 4, 8)
        np.testing.assert_allclose(out, 2.0)


class TestResetNet:
    """Test reset_net utility."""

    def test_resets_memory_modules(self):
        """reset_net should reset all MemoryModules."""
        from grilly.functional.snn import reset_net
        from grilly.nn.snn_neurons import LIFNode

        node = LIFNode(tau=2.0, step_mode="s")
        x = np.ones((2, 4), dtype=np.float32)
        node(x)
        assert node.v is not None

        reset_net(node)
        assert node.v is None

    def test_reset_nested(self):
        """reset_net should reset nested modules."""
        from grilly.functional.snn import reset_net
        from grilly.nn.modules import Sequential
        from grilly.nn.snn_neurons import IFNode

        seq = Sequential(IFNode(step_mode="s"), IFNode(step_mode="s"))
        x = np.ones((2, 4), dtype=np.float32)
        # Run through each module
        for mod in seq._modules.values():
            mod(x)

        reset_net(seq)
        for mod in seq._modules.values():
            assert mod.v is None


class TestSetStepMode:
    """Test set_step_mode utility."""

    def test_sets_mode(self):
        """set_step_mode should update all BaseNodes."""
        from grilly.functional.snn import set_step_mode
        from grilly.nn.snn_neurons import IFNode, LIFNode

        node1 = IFNode(step_mode="s")
        node2 = LIFNode(tau=2.0, step_mode="s")

        set_step_mode(node1, "m")
        set_step_mode(node2, "m")
        assert node1.step_mode == "m"
        assert node2.step_mode == "m"
