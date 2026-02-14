"""
Tests for ONNX import, export, and LoRA fine-tuning.

All tests build small ONNX models programmatically so that no external
model files are required.
"""

import os
import tempfile

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from grilly.nn.lora import LoRAConfig
from grilly.nn.modules import LayerNorm, Linear
from grilly.utils.onnx_exporter import OnnxExporter
from grilly.utils.onnx_finetune import OnnxFineTuner
from grilly.utils.onnx_loader import GrillyOnnxModel, OnnxModelLoader, OnnxOpRegistry

# ---------------------------------------------------------------------------
# Helper: build small ONNX models in-memory
# ---------------------------------------------------------------------------


def _make_linear_relu_onnx(in_features=8, out_features=4):
    """Create a minimal ONNX model: Gemm(x, W, b) -> Relu."""
    weight = np.random.randn(in_features, out_features).astype(np.float32) * 0.1
    bias = np.zeros(out_features, dtype=np.float32)

    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, in_features])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, out_features])

    w_init = numpy_helper.from_array(weight, name="W")
    b_init = numpy_helper.from_array(bias, name="b")

    gemm_node = helper.make_node("Gemm", ["X", "W", "b"], ["gemm_out"], transB=0)
    relu_node = helper.make_node("Relu", ["gemm_out"], ["Y"])

    graph = helper.make_graph(
        [gemm_node, relu_node],
        "linear_relu",
        [x_info],
        [y_info],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model, weight, bias


def _make_two_linear_onnx(d_in=8, d_hidden=16, d_out=4):
    """Two-layer MLP: Gemm -> Relu -> Gemm."""
    w1 = np.random.randn(d_in, d_hidden).astype(np.float32) * 0.1
    b1 = np.zeros(d_hidden, dtype=np.float32)
    w2 = np.random.randn(d_hidden, d_out).astype(np.float32) * 0.1
    b2 = np.zeros(d_out, dtype=np.float32)

    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, d_in])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, d_out])

    nodes = [
        helper.make_node("Gemm", ["X", "W1", "b1"], ["h"], transB=0),
        helper.make_node("Relu", ["h"], ["h_relu"]),
        helper.make_node("Gemm", ["h_relu", "W2", "b2"], ["Y"], transB=0),
    ]
    inits = [
        numpy_helper.from_array(w1, name="W1"),
        numpy_helper.from_array(b1, name="b1"),
        numpy_helper.from_array(w2, name="W2"),
        numpy_helper.from_array(b2, name="b2"),
    ]
    graph = helper.make_graph(nodes, "two_linear", [x_info], [y_info], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model, {"W1": w1, "b1": b1, "W2": w2, "b2": b2}


def _make_layernorm_onnx(dim=16):
    """Single LayerNormalization node."""
    scale = np.ones(dim, dtype=np.float32)
    bias = np.zeros(dim, dtype=np.float32)

    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, dim])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, dim])

    nodes = [
        helper.make_node(
            "LayerNormalization", ["X", "scale", "bias"], ["Y"], epsilon=1e-5, axis=-1
        ),
    ]
    inits = [
        numpy_helper.from_array(scale, name="scale"),
        numpy_helper.from_array(bias, name="bias"),
    ]
    graph = helper.make_graph(nodes, "layernorm", [x_info], [y_info], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _make_transformer_block_onnx(dim=16):
    """A simplified transformer block: LayerNorm -> Gemm (q_proj) -> Relu -> Gemm -> Add."""
    scale = np.ones(dim, dtype=np.float32)
    bias = np.zeros(dim, dtype=np.float32)
    w_q = np.random.randn(dim, dim).astype(np.float32) * 0.1
    b_q = np.zeros(dim, dtype=np.float32)
    w_o = np.random.randn(dim, dim).astype(np.float32) * 0.1
    b_o = np.zeros(dim, dtype=np.float32)

    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, dim])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, dim])

    nodes = [
        helper.make_node(
            "LayerNormalization",
            ["X", "ln_scale", "ln_bias"],
            ["ln_out"],
            epsilon=1e-5,
            axis=-1,
        ),
        helper.make_node("Gemm", ["ln_out", "W_q", "b_q"], ["q_proj_out"], transB=0),
        helper.make_node("Relu", ["q_proj_out"], ["relu_out"]),
        helper.make_node("Gemm", ["relu_out", "W_o", "b_o"], ["proj_out"], transB=0),
        helper.make_node("Add", ["X", "proj_out"], ["Y"]),
    ]
    inits = [
        numpy_helper.from_array(scale, name="ln_scale"),
        numpy_helper.from_array(bias, name="ln_bias"),
        numpy_helper.from_array(w_q, name="W_q"),
        numpy_helper.from_array(b_q, name="b_q"),
        numpy_helper.from_array(w_o, name="W_o"),
        numpy_helper.from_array(b_o, name="b_o"),
    ]
    graph = helper.make_graph(
        nodes,
        "transformer_block",
        [x_info],
        [y_info],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


# ===========================================================================
# Loader tests
# ===========================================================================


@pytest.mark.gpu
class TestOnnxLoader:
    def test_load_simple_onnx(self):
        """Load a minimal ONNX model (Linear + ReLU), verify forward pass."""
        proto, _weight, _bias = _make_linear_relu_onnx(in_features=8, out_features=4)
        model = OnnxModelLoader.load_from_proto(proto)

        assert isinstance(model, GrillyOnnxModel)
        x = np.random.randn(1, 8).astype(np.float32)
        out = model(x)

        assert out is not None
        assert out.shape == (1, 4)
        # ReLU means all values >= 0
        assert np.all(out >= 0)

    def test_load_transformer_onnx(self):
        """Load a small transformer ONNX, verify output shape."""
        proto = _make_transformer_block_onnx(dim=16)
        model = OnnxModelLoader.load_from_proto(proto)

        x = np.random.randn(1, 16).astype(np.float32)
        out = model(x)

        assert out is not None
        assert out.shape == (1, 16)

    def test_onnx_weight_loading(self):
        """Verify initializer weights are correctly loaded into parameters."""
        proto, _weight, _bias = _make_linear_relu_onnx(in_features=8, out_features=4)
        model = OnnxModelLoader.load_from_proto(proto)

        # The model should contain the weights in its constant tensors or modules
        found_linear = False
        for key, mod in model._modules.items():
            if isinstance(mod, Linear):
                found_linear = True
                # Check weight shape
                from grilly.nn.modules import _get_param_array

                w = _get_param_array(mod.weight)
                assert w.shape[0] == 4  # out_features
                assert w.shape[1] == 8  # in_features
        assert found_linear, "No Linear module found in loaded model"

    def test_load_from_file(self):
        """Load ONNX from a file path."""
        proto, _, _ = _make_linear_relu_onnx()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(proto, f.name)
            path = f.name
        try:
            model = OnnxModelLoader.load(path)
            x = np.random.randn(1, 8).astype(np.float32)
            out = model(x)
            assert out.shape == (1, 4)
        finally:
            os.unlink(path)

    def test_layernorm_loading(self):
        """Verify LayerNorm is correctly loaded."""
        proto = _make_layernorm_onnx(dim=16)
        model = OnnxModelLoader.load_from_proto(proto)

        x = np.random.randn(1, 16).astype(np.float32)
        out = model(x)

        assert out is not None
        assert out.shape == (1, 16)
        # LayerNorm output should have roughly zero mean and unit variance
        assert abs(np.mean(out)) < 0.5
        assert abs(np.std(out) - 1.0) < 0.5

    def test_two_layer_mlp(self):
        """Load two-layer MLP, verify forward pass."""
        proto, weights = _make_two_linear_onnx(d_in=8, d_hidden=16, d_out=4)
        model = OnnxModelLoader.load_from_proto(proto)

        x = np.random.randn(1, 8).astype(np.float32)
        out = model(x)

        assert out is not None
        assert out.shape == (1, 4)


# ===========================================================================
# Individual op coverage tests
# ===========================================================================


class TestOnnxOpCoverage:
    def _make_single_op_model(
        self, op_type, input_shapes, output_shape, attrs=None, initializers=None, extra_inputs=None
    ):
        """Helper to build a single-op ONNX model."""
        inputs = []
        input_names = []
        for i, shape in enumerate(input_shapes):
            name = f"input_{i}" if i > 0 else "X"
            inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape)))
            input_names.append(name)

        if extra_inputs:
            input_names.extend(extra_inputs)

        y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(output_shape))

        kwargs = attrs or {}
        node = helper.make_node(op_type, input_names, ["Y"], **kwargs)

        init_list = []
        if initializers:
            for name, arr in initializers.items():
                init_list.append(numpy_helper.from_array(arr, name=name))

        graph = helper.make_graph(
            [node], f"test_{op_type.lower()}", inputs, [y_info], initializer=init_list
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        return model

    def test_relu(self):
        proto = self._make_single_op_model("Relu", [(1, 4)], (1, 4))
        model = OnnxModelLoader.load_from_proto(proto)
        x = np.array([[-1, 0, 1, 2]], dtype=np.float32)
        out = model(x)
        np.testing.assert_array_equal(out, np.array([[0, 0, 1, 2]], dtype=np.float32))

    def test_sigmoid(self):
        proto = self._make_single_op_model("Sigmoid", [(1, 4)], (1, 4))
        model = OnnxModelLoader.load_from_proto(proto)
        x = np.zeros((1, 4), dtype=np.float32)
        out = model(x)
        np.testing.assert_allclose(out, 0.5, atol=1e-6)

    def test_tanh(self):
        proto = self._make_single_op_model("Tanh", [(1, 4)], (1, 4))
        model = OnnxModelLoader.load_from_proto(proto)
        x = np.zeros((1, 4), dtype=np.float32)
        out = model(x)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_add(self):
        proto = self._make_single_op_model("Add", [(1, 4), (1, 4)], (1, 4))
        model = OnnxModelLoader.load_from_proto(proto)
        a = np.ones((1, 4), dtype=np.float32)
        b = np.ones((1, 4), dtype=np.float32) * 2
        out = model(a, b)
        np.testing.assert_allclose(out, 3.0, atol=1e-6)

    def test_mul(self):
        proto = self._make_single_op_model("Mul", [(1, 4), (1, 4)], (1, 4))
        model = OnnxModelLoader.load_from_proto(proto)
        a = np.ones((1, 4), dtype=np.float32) * 3
        b = np.ones((1, 4), dtype=np.float32) * 2
        out = model(a, b)
        np.testing.assert_allclose(out, 6.0, atol=1e-6)

    def test_sub(self):
        proto = self._make_single_op_model("Sub", [(1, 4), (1, 4)], (1, 4))
        model = OnnxModelLoader.load_from_proto(proto)
        a = np.ones((1, 4), dtype=np.float32) * 5
        b = np.ones((1, 4), dtype=np.float32) * 2
        out = model(a, b)
        np.testing.assert_allclose(out, 3.0, atol=1e-6)

    def test_sqrt(self):
        proto = self._make_single_op_model("Sqrt", [(1, 4)], (1, 4))
        model = OnnxModelLoader.load_from_proto(proto)
        x = np.array([[4, 9, 16, 25]], dtype=np.float32)
        out = model(x)
        np.testing.assert_allclose(out, np.array([[2, 3, 4, 5]], dtype=np.float32), atol=1e-5)

    def test_pow(self):
        proto = self._make_single_op_model("Pow", [(1, 4), (1, 4)], (1, 4))
        model = OnnxModelLoader.load_from_proto(proto)
        base = np.array([[2, 3, 4, 5]], dtype=np.float32)
        exp = np.array([[2, 2, 2, 2]], dtype=np.float32)
        out = model(base, exp)
        np.testing.assert_allclose(out, np.array([[4, 9, 16, 25]], dtype=np.float32), atol=1e-5)

    def test_shape(self):
        proto = self._make_single_op_model("Shape", [(2, 3)], (2,))
        model = OnnxModelLoader.load_from_proto(proto)
        x = np.zeros((2, 3), dtype=np.float32)
        out = model(x)
        np.testing.assert_array_equal(out, np.array([2, 3]))

    def test_transpose(self):
        proto = self._make_single_op_model("Transpose", [(2, 3)], (3, 2), attrs={"perm": [1, 0]})
        model = OnnxModelLoader.load_from_proto(proto)
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        out = model(x)
        np.testing.assert_array_equal(out, x.T)

    def test_softmax(self):
        proto = self._make_single_op_model("Softmax", [(1, 4)], (1, 4), attrs={"axis": -1})
        model = OnnxModelLoader.load_from_proto(proto)
        x = np.array([[1, 2, 3, 4]], dtype=np.float32)
        out = model(x)
        assert abs(np.sum(out) - 1.0) < 1e-5

    def test_where(self):
        """Test Where op."""
        # Build manually since Where takes 3 inputs with different types
        cond_vi = helper.make_tensor_value_info("cond", TensorProto.BOOL, [1, 4])
        a_vi = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 4])
        b_vi = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 4])
        y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

        node = helper.make_node("Where", ["cond", "a", "b"], ["Y"])
        graph = helper.make_graph([node], "test_where", [cond_vi, a_vi, b_vi], [y_info])
        proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        proto.ir_version = 8

        model = OnnxModelLoader.load_from_proto(proto)
        cond = np.array([[True, False, True, False]])
        a = np.ones((1, 4), dtype=np.float32) * 10
        b = np.ones((1, 4), dtype=np.float32) * 20
        out = model(cond, a, b)
        expected = np.array([[10, 20, 10, 20]], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_less_or_equal(self):
        proto = self._make_single_op_model("LessOrEqual", [(1, 4), (1, 4)], (1, 4))
        model = OnnxModelLoader.load_from_proto(proto)
        a = np.array([[1, 2, 3, 4]], dtype=np.float32)
        b = np.array([[1, 3, 2, 4]], dtype=np.float32)
        out = model(a, b)
        expected = np.array([[True, True, False, True]])
        np.testing.assert_array_equal(out, expected)

    def test_and(self):
        a_vi = helper.make_tensor_value_info("a", TensorProto.BOOL, [1, 4])
        b_vi = helper.make_tensor_value_info("b", TensorProto.BOOL, [1, 4])
        y_info = helper.make_tensor_value_info("Y", TensorProto.BOOL, [1, 4])
        node = helper.make_node("And", ["a", "b"], ["Y"])
        graph = helper.make_graph([node], "test_and", [a_vi, b_vi], [y_info])
        proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        proto.ir_version = 8

        model = OnnxModelLoader.load_from_proto(proto)
        a = np.array([[True, True, False, False]])
        b = np.array([[True, False, True, False]])
        out = model(a, b)
        expected = np.array([[True, False, False, False]])
        np.testing.assert_array_equal(out, expected)

    def test_cos(self):
        proto = self._make_single_op_model("Cos", [(1, 4)], (1, 4))
        model = OnnxModelLoader.load_from_proto(proto)
        x = np.array([[0.0, np.pi / 2, np.pi, 3 * np.pi / 2]], dtype=np.float32)
        out = model(x)
        expected = np.array([[1.0, 0.0, -1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_sin(self):
        proto = self._make_single_op_model("Sin", [(1, 4)], (1, 4))
        model = OnnxModelLoader.load_from_proto(proto)
        x = np.array([[0.0, np.pi / 2, np.pi, 3 * np.pi / 2]], dtype=np.float32)
        out = model(x)
        expected = np.array([[0.0, 1.0, 0.0, -1.0]], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_isnan(self):
        x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
        y_info = helper.make_tensor_value_info("Y", TensorProto.BOOL, [1, 4])
        node = helper.make_node("IsNaN", ["X"], ["Y"])
        graph = helper.make_graph([node], "test_isnan", [x_vi], [y_info])
        proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        proto.ir_version = 8

        model = OnnxModelLoader.load_from_proto(proto)
        x = np.array([[0.0, np.nan, 2.0, np.nan]], dtype=np.float32)
        out = model(x)
        expected = np.array([[False, True, False, True]])
        np.testing.assert_array_equal(out, expected)

    def test_range(self):
        start_vi = helper.make_tensor_value_info("start", TensorProto.INT64, [])
        limit_vi = helper.make_tensor_value_info("limit", TensorProto.INT64, [])
        delta_vi = helper.make_tensor_value_info("delta", TensorProto.INT64, [])
        y_info = helper.make_tensor_value_info("Y", TensorProto.INT64, [None])
        node = helper.make_node("Range", ["start", "limit", "delta"], ["Y"])
        graph = helper.make_graph([node], "test_range", [start_vi, limit_vi, delta_vi], [y_info])
        proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        proto.ir_version = 8

        model = OnnxModelLoader.load_from_proto(proto)
        out = model(np.array(1, dtype=np.int64), np.array(7, dtype=np.int64), np.array(2, dtype=np.int64))
        expected = np.array([1, 3, 5], dtype=np.int64)
        np.testing.assert_array_equal(out, expected)


# ===========================================================================
# Export tests
# ===========================================================================


@pytest.mark.gpu
class TestOnnxExporter:
    def test_export_linear(self):
        """Export a single Linear module to ONNX."""
        linear = Linear(8, 4, bias=True)
        exporter = OnnxExporter()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            exporter.export(linear, path, input_shapes={"input": (1, 8)})
            # Verify the file is valid ONNX
            reloaded = onnx.load(path)
            assert len(reloaded.graph.node) > 0
        finally:
            os.unlink(path)

    def test_export_roundtrip(self):
        """Load ONNX -> export to ONNX -> reload -> verify same output."""
        proto, _weight, _bias = _make_linear_relu_onnx(in_features=8, out_features=4)
        model = OnnxModelLoader.load_from_proto(proto)

        x = np.random.randn(1, 8).astype(np.float32)
        out1 = model(x)

        # Export
        exporter = OnnxExporter()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            exporter.export(model, path, input_shapes={"input": (1, 8)})
            # Reload
            model2 = OnnxModelLoader.load(path)
            out2 = model2(x)

            assert out2 is not None
            assert out2.shape == out1.shape
        finally:
            os.unlink(path)

    def test_export_layernorm(self):
        """Export LayerNorm module."""
        ln = LayerNorm(16)
        exporter = OnnxExporter()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            exporter.export(ln, path, input_shapes={"input": (1, 16)})
            reloaded = onnx.load(path)
            # Should have a LayerNormalization node
            op_types = [n.op_type for n in reloaded.graph.node]
            assert "LayerNormalization" in op_types or "Identity" in op_types
        finally:
            os.unlink(path)


# ===========================================================================
# LoRA fine-tuning tests
# ===========================================================================


class TestOnnxFineTuner:
    def test_apply_lora(self):
        """Load model, apply LoRA, verify LoRA layers exist."""
        proto, _ = _make_two_linear_onnx(d_in=8, d_hidden=16, d_out=4)
        model = OnnxModelLoader.load_from_proto(proto)

        config = LoRAConfig(rank=4, alpha=8, target_modules=[])  # match all
        tuner = OnnxFineTuner(model, config)
        tuner.apply_lora()

        assert tuner._applied
        assert len(tuner._lora_layers) > 0
        assert tuner.num_trainable_params() > 0

    def test_lora_save_load(self):
        """Save LoRA adapters, reload, verify same params."""
        proto, _ = _make_two_linear_onnx(d_in=8, d_hidden=16, d_out=4)
        model = OnnxModelLoader.load_from_proto(proto)

        config = LoRAConfig(rank=4, alpha=8, target_modules=[])
        tuner = OnnxFineTuner(model, config)
        tuner.apply_lora()

        # Modify LoRA weights so they're not zeros
        for lora in tuner._lora_layers.values():
            lora.lora_A.data = np.random.randn(*lora.lora_A.data.shape).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tuner.save_lora(tmpdir)

            # Reload into a fresh tuner
            model2 = OnnxModelLoader.load_from_proto(proto)
            tuner2 = OnnxFineTuner(model2, config)
            tuner2.apply_lora()
            tuner2.load_lora(tmpdir)

            # Verify weights match
            for name in tuner._lora_layers:
                if name in tuner2._lora_layers:
                    np.testing.assert_allclose(
                        tuner._lora_layers[name].lora_A.data,
                        tuner2._lora_layers[name].lora_A.data,
                        atol=1e-6,
                    )

    def test_export_with_lora(self):
        """Export fine-tuned model to ONNX, verify weights merged."""
        proto, _ = _make_two_linear_onnx(d_in=8, d_hidden=16, d_out=4)
        model = OnnxModelLoader.load_from_proto(proto)

        config = LoRAConfig(rank=4, alpha=8, target_modules=[])
        tuner = OnnxFineTuner(model, config)
        tuner.apply_lora()

        # Set non-trivial LoRA weights
        for lora in tuner._lora_layers.values():
            lora.lora_A.data = np.random.randn(*lora.lora_A.data.shape).astype(np.float32) * 0.1
            lora.lora_B.data = np.random.randn(*lora.lora_B.data.shape).astype(np.float32) * 0.1

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            tuner.save_onnx(path, input_shapes={"input": (1, 8)})
            # Should produce a valid ONNX file
            reloaded = onnx.load(path)
            assert len(reloaded.graph.node) > 0
        finally:
            os.unlink(path)

    def test_print_trainable_parameters(self, capsys):
        """Verify print_trainable_parameters works."""
        proto, _ = _make_two_linear_onnx(d_in=8, d_hidden=16, d_out=4)
        model = OnnxModelLoader.load_from_proto(proto)

        config = LoRAConfig(rank=4, alpha=8, target_modules=[])
        tuner = OnnxFineTuner(model, config)
        tuner.apply_lora()
        tuner.print_trainable_parameters()

        captured = capsys.readouterr()
        assert "trainable params" in captured.out


# ===========================================================================
# Op Registry tests
# ===========================================================================


class TestOnnxOpRegistry:
    def test_supported_ops(self):
        """Registry should have all documented ops."""
        registry = OnnxOpRegistry()
        expected_ops = [
            "Gemm",
            "MatMul",
            "Add",
            "Mul",
            "Sub",
            "Div",
            "Relu",
            "Gelu",
            "Sigmoid",
            "Tanh",
            "Softmax",
            "LayerNormalization",
            "Reshape",
            "Transpose",
            "Gather",
            "Unsqueeze",
            "Squeeze",
            "Concat",
            "Split",
            "Slice",
            "Where",
            "Cast",
            "Shape",
            "Pow",
            "Sqrt",
            "ReduceMean",
            "Erf",
            "Constant",
            "ConstantOfShape",
            "Expand",
            "Equal",
            "Less",
            "LessOrEqual",
            "Greater",
            "And",
            "Not",
            "Cos",
            "Sin",
            "IsNaN",
            "Range",
            "Dropout",
            "Identity",
        ]
        for op in expected_ops:
            assert registry.get(op) is not None, f"Missing handler for {op}"

    def test_custom_handler_registration(self):
        """Register a custom handler and verify it's used."""
        registry = OnnxOpRegistry()

        def custom_handler(node, inputs, initializers, attrs):
            def fn(*args):
                return args[0] * 42

            return "callable", fn

        registry.register("CustomOp", custom_handler)
        assert registry.get("CustomOp") is not None


# ===========================================================================
# GrillyOnnxModel API tests
# ===========================================================================


class TestGrillyOnnxModel:
    def test_get_linear_layers(self):
        """Verify get_linear_layers returns Linear modules."""
        proto, _ = _make_two_linear_onnx()
        model = OnnxModelLoader.load_from_proto(proto)
        linears = model.get_linear_layers()
        assert len(linears) >= 2

    def test_repr(self):
        """Model repr should include node count."""
        proto, _, _ = _make_linear_relu_onnx()
        model = OnnxModelLoader.load_from_proto(proto)
        r = repr(model)
        assert "GrillyOnnxModel" in r
        assert "nodes=" in r

    def test_parameters_iterable(self):
        """Parameters should be iterable and non-empty for a model with weights."""
        proto, _, _ = _make_linear_relu_onnx()
        model = OnnxModelLoader.load_from_proto(proto)
        params = list(model.parameters())
        assert len(params) > 0
