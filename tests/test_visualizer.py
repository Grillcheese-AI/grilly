"""
Tests for visualization modules:
  - backend.snn_visualizer (SNNState, SNNVisualizer, get_visualizer)
  - utils.visualization (matplotlib-based training plots)
  - utils.visualizer (ASCII/HTML/SVG architecture diagrams)
"""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# SNNVisualizer tests
# ---------------------------------------------------------------------------
from grilly.backend.snn_visualizer import SNNState, SNNVisualizer, get_visualizer


class TestSNNState:
    """Tests for SNNState dataclass."""

    def test_to_dict_basic(self):
        state = SNNState(
            timestamp=1.0,
            neuron_potentials=[0.1, 0.5, -0.3],
            spike_events=[1],
        )
        d = state.to_dict()
        assert d["timestamp"] == 1.0
        assert d["spike_count"] == 1
        assert d["spike_events"] == [1]
        assert isinstance(d["avg_potential"], float)
        assert isinstance(d["max_potential"], float)
        assert d["stdp_trace"] is None

    def test_to_dict_limits_potentials_to_100(self):
        potentials = list(range(200))
        state = SNNState(
            timestamp=0.0,
            neuron_potentials=potentials,
            spike_events=[],
        )
        d = state.to_dict()
        assert len(d["neuron_potentials"]) == 100

    def test_to_dict_with_stdp_trace(self):
        trace = list(range(50))
        state = SNNState(
            timestamp=0.0,
            neuron_potentials=[1.0],
            spike_events=[],
            stdp_trace=trace,
        )
        d = state.to_dict()
        assert len(d["stdp_trace"]) == 20

    def test_to_dict_avg_and_max(self):
        state = SNNState(
            timestamp=0.0,
            neuron_potentials=[1.0, 3.0, 5.0],
            spike_events=[],
        )
        d = state.to_dict()
        assert d["avg_potential"] == pytest.approx(3.0)
        assert d["max_potential"] == pytest.approx(5.0)


class TestSNNVisualizer:
    """Tests for SNNVisualizer."""

    def test_init_defaults(self):
        viz = SNNVisualizer()
        assert viz.max_history == 100
        assert viz.enabled is False
        assert len(viz.state_history) == 0
        assert len(viz.active_clients) == 0

    def test_init_custom_history(self):
        viz = SNNVisualizer(max_history=10)
        assert viz.max_history == 10

    def test_capture_state(self):
        viz = SNNVisualizer()
        potentials = np.array([0.1, 0.5, -0.3], dtype=np.float32)
        spikes = np.array([1], dtype=np.int32)

        state = viz.capture_state(potentials, spikes)

        assert isinstance(state, SNNState)
        assert state.neuron_potentials == potentials.tolist()
        assert state.spike_events == spikes.tolist()
        assert len(viz.state_history) == 1

    def test_capture_state_with_weights_and_stdp(self):
        viz = SNNVisualizer()
        potentials = np.zeros(5, dtype=np.float32)
        spikes = np.array([], dtype=np.int32)
        weights = np.eye(5, dtype=np.float32)
        stdp = np.ones(5, dtype=np.float32)

        state = viz.capture_state(potentials, spikes, weights, stdp)

        assert state.synaptic_weights is not None
        assert state.stdp_trace is not None
        assert len(state.synaptic_weights) == 5

    def test_capture_state_trims_history(self):
        viz = SNNVisualizer(max_history=3)
        for i in range(5):
            viz.capture_state(
                np.array([float(i)], dtype=np.float32),
                np.array([], dtype=np.int32),
            )
        assert len(viz.state_history) == 3
        # Should keep the most recent
        assert viz.state_history[0].neuron_potentials == [2.0]
        assert viz.state_history[-1].neuron_potentials == [4.0]

    def test_capture_state_accepts_list_spikes(self):
        viz = SNNVisualizer()
        state = viz.capture_state(
            np.array([1.0], dtype=np.float32),
            [0, 2, 3],  # list instead of ndarray
        )
        assert state.spike_events == [0, 2, 3]

    def test_enable_disable(self):
        viz = SNNVisualizer()
        assert viz.enabled is False
        viz.enable()
        assert viz.enabled is True
        viz.disable()
        assert viz.enabled is False

    def test_register_unregister_client(self):
        viz = SNNVisualizer()
        client = MagicMock()

        viz.register_client(client)
        assert client in viz.active_clients

        viz.unregister_client(client)
        assert client not in viz.active_clients

    def test_unregister_absent_client_no_error(self):
        viz = SNNVisualizer()
        viz.unregister_client(MagicMock())  # should not raise

    def test_get_history_all(self):
        viz = SNNVisualizer()
        for i in range(3):
            viz.capture_state(
                np.array([float(i)], dtype=np.float32),
                np.array([], dtype=np.int32),
            )
        history = viz.get_history()
        assert len(history) == 3
        assert all(isinstance(h, dict) for h in history)

    def test_get_history_n(self):
        viz = SNNVisualizer()
        for i in range(5):
            viz.capture_state(
                np.array([float(i)], dtype=np.float32),
                np.array([], dtype=np.int32),
            )
        history = viz.get_history(n=2)
        assert len(history) == 2

    def test_get_statistics_empty(self):
        viz = SNNVisualizer()
        stats = viz.get_statistics()
        assert "error" in stats

    def test_get_statistics(self):
        viz = SNNVisualizer()
        for i in range(5):
            viz.capture_state(
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
            )
        stats = viz.get_statistics()
        assert stats["total_snapshots"] == 5
        assert stats["avg_spike_rate"] == pytest.approx(2.0)
        assert stats["avg_membrane_potential"] == pytest.approx(2.0)
        assert stats["active_clients"] == 0
        assert stats["visualization_enabled"] is False

    def test_broadcast_noop_when_disabled(self):
        viz = SNNVisualizer()
        client = AsyncMock()
        viz.register_client(client)
        # disabled by default
        state = SNNState(timestamp=0, neuron_potentials=[1.0], spike_events=[])
        asyncio.get_event_loop().run_until_complete(viz.broadcast_state(state))
        client.send_text.assert_not_called()

    def test_broadcast_sends_to_clients(self):
        viz = SNNVisualizer()
        viz.enable()
        client = AsyncMock()
        viz.register_client(client)

        state = SNNState(timestamp=0, neuron_potentials=[1.0], spike_events=[0])
        asyncio.get_event_loop().run_until_complete(viz.broadcast_state(state))

        client.send_text.assert_called_once()
        sent = json.loads(client.send_text.call_args[0][0])
        assert sent["type"] == "snn_state"
        assert sent["data"]["spike_count"] == 1

    def test_broadcast_removes_disconnected_clients(self):
        viz = SNNVisualizer()
        viz.enable()

        good_client = AsyncMock()
        bad_client = AsyncMock()
        bad_client.send_text.side_effect = ConnectionError("gone")

        viz.register_client(good_client)
        viz.register_client(bad_client)
        assert len(viz.active_clients) == 2

        state = SNNState(timestamp=0, neuron_potentials=[1.0], spike_events=[])
        asyncio.get_event_loop().run_until_complete(viz.broadcast_state(state))

        assert bad_client not in viz.active_clients
        assert good_client in viz.active_clients


class TestGetVisualizer:
    """Tests for the global singleton."""

    def test_returns_singleton(self):
        import grilly.backend.snn_visualizer as mod
        mod._visualizer = None  # reset
        v1 = get_visualizer()
        v2 = get_visualizer()
        assert v1 is v2
        mod._visualizer = None  # cleanup


# ---------------------------------------------------------------------------
# utils.visualization tests (matplotlib-based)
# ---------------------------------------------------------------------------

from grilly.utils.visualization import (
    MATPLOTLIB_AVAILABLE,
    plot_training_history,
    print_model_summary,
    visualize_attention_weights,
)


class _FakeParam:
    """Minimal parameter mock for visualization functions."""
    def __init__(self, shape, requires_grad=True):
        self.data = np.random.randn(*shape).astype(np.float32)
        self.grad = np.random.randn(*shape).astype(np.float32)
        self.requires_grad = requires_grad
        self.shape = shape

    def flatten(self):
        return self.data.flatten()


class _FakeModule:
    """Minimal module mock for model visualization."""
    def __init__(self):
        self._parameters = {
            "weight": _FakeParam((10, 5)),
            "bias": _FakeParam((10,)),
        }
        self._modules = {}


class _FakeNestedModule:
    """Module with submodules."""
    def __init__(self):
        self._parameters = {}
        self._modules = {"layer1": _FakeModule(), "layer2": _FakeModule()}


class TestPlotTrainingHistory:
    """Tests for plot_training_history."""

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
    def test_loss_only(self, tmp_path):
        path = str(tmp_path / "loss.png")
        plot_training_history([1.0, 0.8, 0.6], save_path=path, show=False)
        assert os.path.exists(path)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
    def test_loss_and_accuracy(self, tmp_path):
        path = str(tmp_path / "loss_acc.png")
        plot_training_history(
            [1.0, 0.8, 0.6],
            accuracies=[0.5, 0.7, 0.9],
            save_path=path,
            show=False,
        )
        assert os.path.exists(path)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
    def test_with_validation(self, tmp_path):
        path = str(tmp_path / "val.png")
        plot_training_history(
            [1.0, 0.8, 0.6],
            accuracies=[0.5, 0.7, 0.9],
            val_losses=[1.1, 0.9, 0.7],
            val_accuracies=[0.4, 0.6, 0.85],
            save_path=path,
            show=False,
        )
        assert os.path.exists(path)

    def test_no_matplotlib_no_crash(self):
        with patch("grilly.utils.visualization.MATPLOTLIB_AVAILABLE", False):
            # Should print a message and return, not crash
            plot_training_history([1.0, 0.8])


class TestPrintModelSummary:
    """Tests for print_model_summary."""

    def test_flat_module(self, capsys):
        model = _FakeModule()
        print_model_summary(model)
        captured = capsys.readouterr()
        assert "Model Summary" in captured.out
        assert "weight" in captured.out
        assert "bias" in captured.out
        assert "Total parameters" in captured.out

    def test_nested_module(self, capsys):
        model = _FakeNestedModule()
        print_model_summary(model)
        captured = capsys.readouterr()
        assert "layer1" in captured.out
        assert "layer2" in captured.out


class TestVisualizeAttentionWeights:
    """Tests for visualize_attention_weights."""

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
    def test_2d_attention(self, tmp_path):
        path = str(tmp_path / "attn.png")
        weights = np.random.rand(8, 8).astype(np.float32)
        visualize_attention_weights(weights, save_path=path, show=False)
        assert os.path.exists(path)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
    def test_3d_multihead_attention(self, tmp_path):
        path = str(tmp_path / "mh_attn.png")
        weights = np.random.rand(4, 8, 8).astype(np.float32)
        visualize_attention_weights(weights, save_path=path, show=False)
        assert os.path.exists(path)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
    def test_with_tokens(self, tmp_path):
        path = str(tmp_path / "tok_attn.png")
        tokens = ["the", "cat", "sat", "on"]
        weights = np.random.rand(4, 4).astype(np.float32)
        visualize_attention_weights(weights, tokens=tokens, save_path=path, show=False)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# utils.visualizer tests (architecture diagrams)
# ---------------------------------------------------------------------------

from grilly.utils.visualizer import (
    LayerInfo,
    _classify_layer,
    _count_params,
    _format_memory,
    _format_param_count,
    _infer_shapes,
    discover_layers,
    render_ascii,
    render_html,
    summary,
    visualize,
)


class TestFormatParamCount:
    def test_small(self):
        assert _format_param_count(500) == "500"

    def test_thousands(self):
        assert _format_param_count(1500) == "1.5K"

    def test_millions(self):
        assert _format_param_count(1_500_000) == "1.5M"

    def test_billions(self):
        assert _format_param_count(1_500_000_000) == "1.5B"


class TestFormatMemory:
    def test_bytes(self):
        assert _format_memory(100) == "400 B"

    def test_kilobytes(self):
        assert "KB" in _format_memory(1000)

    def test_megabytes(self):
        assert "MB" in _format_memory(1_000_000)

    def test_gigabytes(self):
        assert "GB" in _format_memory(1_000_000_000)


class TestClassifyLayer:
    def test_known_types(self):
        assert _classify_layer("Linear") == "linear"
        assert _classify_layer("Embedding") == "embedding"
        assert _classify_layer("ReLU") == "activation"
        assert _classify_layer("LayerNorm") == "norm"
        assert _classify_layer("Conv2d") == "conv"
        assert _classify_layer("LIFNeuron") == "snn"
        assert _classify_layer("VSAReasoningHead") == "vsa"

    def test_unknown_type(self):
        assert _classify_layer("SomethingNew") == "unknown"


class _MockLinear:
    """Mock grilly Linear layer."""
    def __init__(self, in_f, out_f):
        self.in_features = in_f
        self.out_features = out_f
        self.weight = np.random.randn(out_f, in_f).astype(np.float32)
        self.bias = np.random.randn(out_f).astype(np.float32)
        self._parameters = {"weight": self.weight, "bias": self.bias}
        self._modules = {}

    def parameters(self):
        return [self.weight, self.bias]


class _MockEmbedding:
    """Mock grilly Embedding layer."""
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        self._parameters = {"weight": self.weight}
        self._modules = {}

    def parameters(self):
        return [self.weight]


class _MockSequential:
    """Mock grilly Sequential container."""
    def __init__(self, *modules):
        self._parameters = {}
        self._modules = {str(i): m for i, m in enumerate(modules)}

    def parameters(self):
        result = []
        for m in self._modules.values():
            result.extend(m.parameters())
        return result


class TestInferShapes:
    def test_linear(self):
        layer = _MockLinear(128, 64)
        in_s, out_s = _infer_shapes(layer)
        assert "128" in in_s
        assert "64" in out_s

    def test_embedding(self):
        layer = _MockEmbedding(32768, 768)
        in_s, out_s = _infer_shapes(layer)
        assert "int" in in_s.lower() or "B" in in_s
        assert "768" in out_s


class TestCountParams:
    def test_linear(self):
        layer = _MockLinear(10, 5)
        assert _count_params(layer) == 10 * 5 + 5  # weight + bias

    def test_embedding(self):
        layer = _MockEmbedding(100, 32)
        assert _count_params(layer) == 100 * 32


class TestDiscoverLayers:
    def test_sequential(self):
        model = _MockSequential(
            _MockLinear(784, 256),
            _MockLinear(256, 10),
        )
        layers = discover_layers(model)
        assert len(layers) >= 2
        assert all(isinstance(l, LayerInfo) for l in layers)

    def test_single_linear_in_container(self):
        """A bare layer has no _modules children; wrap in Sequential."""
        model = _MockSequential(_MockLinear(32, 16), _MockEmbedding(100, 32))
        layers = discover_layers(model)
        assert len(layers) >= 2
        type_names = [l.type_name for l in layers]
        assert "_MockLinear" in type_names or "Linear" in type_names


class TestRenderAscii:
    def test_returns_string(self):
        model = _MockSequential(_MockLinear(32, 16))
        result = render_ascii(model)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_model_name(self):
        model = _MockSequential(_MockLinear(32, 16))
        result = render_ascii(model)
        assert "_MockSequential" in result

    def test_contains_param_count(self):
        model = _MockSequential(_MockLinear(32, 16))
        result = render_ascii(model)
        assert "params" in result.lower()


class TestRenderHtml:
    def test_returns_html(self):
        model = _MockSequential(_MockLinear(32, 16))
        result = render_html(model)
        assert "<html" in result
        assert "<svg" in result
        assert "</html>" in result

    def test_contains_model_info(self):
        model = _MockSequential(_MockLinear(32, 16))
        result = render_html(model)
        assert "_MockSequential" in result

    def test_custom_title(self):
        model = _MockSequential(_MockLinear(32, 16))
        result = render_html(model, title="TestModel")
        assert "TestModel" in result


class TestVisualize:
    def test_ascii_default(self, capsys):
        model = _MockSequential(_MockLinear(32, 16))
        result = visualize(model)
        assert result is None  # prints to stdout
        captured = capsys.readouterr()
        assert "_MockSequential" in captured.out

    def test_ascii_return_str(self):
        model = _MockSequential(_MockLinear(32, 16))
        result = visualize(model, output="ascii", return_str=True)
        assert isinstance(result, str)
        assert "_MockSequential" in result

    def test_html_return_str(self):
        model = _MockSequential(_MockLinear(32, 16))
        result = visualize(model, output="html", return_str=True)
        assert "<html" in result
        assert "<svg" in result

    def test_svg_return_str(self):
        model = _MockSequential(_MockLinear(32, 16))
        result = visualize(model, output="svg", return_str=True)
        assert result.startswith("<svg")
        assert "</svg>" in result

    def test_save_to_file(self, tmp_path):
        model = _MockSequential(_MockLinear(32, 16))
        path = str(tmp_path / "model.html")
        result = visualize(model, output="html", save_path=path)
        assert result is not None
        assert os.path.exists(path)
        content = open(path, encoding="utf-8").read()
        assert "<html" in content

    def test_invalid_output_mode(self):
        model = _MockSequential(_MockLinear(32, 16))
        with pytest.raises(ValueError, match="Unknown output mode"):
            visualize(model, output="pdf", return_str=True)


class TestSummary:
    def test_returns_string(self, capsys):
        model = _MockSequential(
            _MockLinear(784, 256),
            _MockLinear(256, 10),
        )
        result = summary(model)
        assert isinstance(result, str)
        assert "Summary" in result
        assert "Total params" in result

    def test_contains_layers(self, capsys):
        model = _MockSequential(
            _MockLinear(784, 256),
            _MockLinear(256, 10),
        )
        result = summary(model)
        assert "Linear" in result
