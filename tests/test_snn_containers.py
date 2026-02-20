"""Tests for Phase 3: SNN Containers"""

import numpy as np
import pytest


class TestMultiStepContainer:
    """Test MultiStepContainer wrapping single-step modules."""

    def test_wraps_lif_node(self):
        """MultiStepContainer should process [T,N,D] through LIFNode."""
        from grilly.nn.snn_containers import MultiStepContainer
        from grilly.nn.snn_neurons import LIFNode

        lif = LIFNode(tau=2.0, step_mode="s")
        container = MultiStepContainer(lif)
        x = np.random.rand(4, 2, 16).astype(np.float32) * 0.5
        out = container(x)
        assert out.shape == (4, 2, 16)

    def test_wraps_identity(self):
        """MultiStepContainer should handle simple modules."""
        from grilly.nn.module import Module
        from grilly.nn.snn_containers import MultiStepContainer

        class Scale(Module):
            def forward(self, x):
                return x * 2.0

        container = MultiStepContainer(Scale())
        x = np.ones((3, 2, 4), dtype=np.float32)
        out = container(x)
        np.testing.assert_allclose(out, 2.0)
        assert out.shape == (3, 2, 4)


class TestSeqToANNContainer:
    """Test SeqToANNContainer for temporal->batch reshaping."""

    def test_reshape_linear(self):
        """SeqToANNContainer with Linear should process [T,N,D]."""
        from grilly.nn.modules import Linear
        from grilly.nn.snn_containers import SeqToANNContainer

        linear = Linear(16, 8, bias=False)
        container = SeqToANNContainer(linear)
        x = np.random.rand(4, 2, 16).astype(np.float32)
        out = container(x)
        assert out.shape == (4, 2, 8)

    def test_multiple_modules(self):
        """SeqToANNContainer with multiple modules."""
        from grilly.nn.module import Module
        from grilly.nn.snn_containers import SeqToANNContainer

        class Double(Module):
            def forward(self, x):
                return x * 2.0

        class AddOne(Module):
            def forward(self, x):
                return x + 1.0

        container = SeqToANNContainer(Double(), AddOne())
        x = np.ones((3, 2, 4), dtype=np.float32)
        out = container(x)
        np.testing.assert_allclose(out, 3.0)  # (1 * 2) + 1 = 3

    def test_repr(self):
        from grilly.nn.module import Module
        from grilly.nn.snn_containers import SeqToANNContainer

        class Foo(Module):
            pass

        c = SeqToANNContainer(Foo())
        assert "SeqToANNContainer" in repr(c)


class TestFlatten:
    """Test Flatten module."""

    def test_flatten_default(self):
        """Flatten with start_dim=1 should flatten all except batch."""
        from grilly.nn.snn_containers import Flatten

        flat = Flatten(start_dim=1)
        x = np.random.rand(4, 3, 7, 7).astype(np.float32)
        out = flat(x)
        assert out.shape == (4, 3 * 7 * 7)

    def test_flatten_custom_dims(self):
        """Flatten with custom start/end dims."""
        from grilly.nn.snn_containers import Flatten

        flat = Flatten(start_dim=2, end_dim=-1)
        x = np.random.rand(4, 3, 7, 7).astype(np.float32)
        out = flat(x)
        assert out.shape == (4, 3, 49)


class TestComposability:
    """Test composability of containers with Sequential."""

    def test_seq_to_ann_then_neuron(self):
        """SeqToANNContainer(Linear) -> LIFNode(step_mode='m')."""
        from grilly.nn.modules import Linear
        from grilly.nn.snn_containers import SeqToANNContainer
        from grilly.nn.snn_neurons import LIFNode

        linear = Linear(16, 8, bias=False)
        container = SeqToANNContainer(linear)
        lif = LIFNode(tau=2.0, step_mode="m")

        T, N, D = 4, 2, 16
        x = np.random.rand(T, N, D).astype(np.float32)
        h = container(x)  # (4, 2, 8)
        out = lif(h)  # (4, 2, 8)
        assert out.shape == (4, 2, 8)
        assert set(np.unique(out)).issubset({0.0, 1.0})
