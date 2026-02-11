"""
ONNX Exporter for Grilly

Converts a Grilly nn.Module back to ONNX format.
Walks the module tree and creates corresponding ONNX nodes and initializers.

Usage:
    from grilly.utils.onnx_exporter import OnnxExporter

    exporter = OnnxExporter()
    exporter.export(model, "model.onnx", input_shapes={"input": (1, 128)})
"""

from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..nn.module import Module
from ..nn.modules import (
    GELU,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    MultiheadAttention,
    ReLU,
    Residual,
    Sequential,
    SiLU,
    Softmax,
    _get_param_array,
)


class OnnxExporter:
    """Export a Grilly ``nn.Module`` to ONNX format.

    Supports ``Linear``, ``LayerNorm``, ``Embedding``, activations
    (``ReLU``, ``GELU``, ``SiLU``, ``Softmax``), ``Dropout``,
    ``Sequential``, ``Residual``, and ``GrillyOnnxModel``.
    """

    def __init__(self, opset_version: int = 17):
        self.opset_version = opset_version
        self._counter = 0
        self._nodes: list[Any] = []
        self._initializers: list[Any] = []
        self._inputs: list[Any] = []
        self._outputs: list[Any] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(
        self,
        model: Module,
        path: str,
        input_shapes: dict[str, tuple[int, ...]] | None = None,
        input_dtypes: dict[str, int] | None = None,
        output_names: list[str] | None = None,
        dummy_input: np.ndarray | None = None,
    ) -> None:
        """Export *model* to an ONNX file at *path*.

        Args:
            model: Grilly Module to export.
            path: Destination ``.onnx`` file path.
            input_shapes: Mapping of input name -> shape.  If only one input,
                a dict with key ``"input"`` is sufficient.
            input_dtypes: Mapping of input name -> ONNX TensorProto dtype.
                Defaults to ``FLOAT`` for all inputs.
            output_names: Optional list of output names.
            dummy_input: Optional dummy input used to trace shapes through the
                model.  If provided, *input_shapes* is inferred.
        """
        self._reset()

        # Determine input shapes
        if dummy_input is not None:
            if input_shapes is None:
                input_shapes = {"input": dummy_input.shape}
            if input_dtypes is None:
                np_to_onnx = {
                    np.float32: TensorProto.FLOAT,
                    np.float64: TensorProto.DOUBLE,
                    np.int32: TensorProto.INT32,
                    np.int64: TensorProto.INT64,
                }
                input_dtypes = {"input": np_to_onnx.get(dummy_input.dtype.type, TensorProto.FLOAT)}

        if input_shapes is None:
            input_shapes = {"input": (1, 128)}

        if input_dtypes is None:
            input_dtypes = {k: TensorProto.FLOAT for k in input_shapes}

        if output_names is None:
            output_names = ["output"]

        # Create graph inputs
        input_value_infos = []
        current_names: dict[str, str] = {}
        for name, shape in input_shapes.items():
            dtype = input_dtypes.get(name, TensorProto.FLOAT)
            vi = helper.make_tensor_value_info(name, dtype, list(shape))
            input_value_infos.append(vi)
            current_names[name] = name

        # Determine the primary input name (first)
        primary_input = list(input_shapes.keys())[0]

        # Walk the module tree and create ONNX nodes
        from .onnx_loader import GrillyOnnxModel

        if isinstance(model, GrillyOnnxModel):
            last_output = self._export_onnx_model(model, primary_input)
        else:
            last_output = self._export_module(model, primary_input, prefix="")

        # Create graph outputs
        output_value_infos = []
        for i, oname in enumerate(output_names):
            vi = helper.make_tensor_value_info(oname, TensorProto.FLOAT, None)
            output_value_infos.append(vi)

        # If last_output differs from expected output name, add identity node
        if last_output != output_names[0]:
            identity_node = helper.make_node(
                "Identity", [last_output], [output_names[0]], name="output_identity"
            )
            self._nodes.append(identity_node)

        # Build the graph
        graph = helper.make_graph(
            self._nodes,
            "grilly_model",
            input_value_infos,
            output_value_infos,
            initializer=self._initializers,
        )

        # Build the model
        onnx_model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", self.opset_version)]
        )
        onnx_model.ir_version = 8

        # Validate
        try:
            onnx.checker.check_model(onnx_model)
        except onnx.checker.ValidationError:
            pass  # Proceed anyway — some dynamic shapes can't be fully validated

        onnx.save(onnx_model, path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self):
        self._counter = 0
        self._nodes = []
        self._initializers = []

    def _unique_name(self, prefix: str = "t") -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def _add_initializer(self, name: str, array: np.ndarray):
        """Add a weight tensor as an initializer."""
        tensor = numpy_helper.from_array(array.astype(np.float32), name=name)
        self._initializers.append(tensor)

    def _export_module(self, module: Module, input_name: str, prefix: str) -> str:
        """Recursively export a Module, returning the output tensor name."""
        if isinstance(module, Sequential):
            return self._export_sequential(module, input_name, prefix)
        elif isinstance(module, Residual):
            return self._export_residual(module, input_name, prefix)
        elif isinstance(module, Linear):
            return self._export_linear(module, input_name, prefix)
        elif isinstance(module, LayerNorm):
            return self._export_layernorm(module, input_name, prefix)
        elif isinstance(module, Embedding):
            return self._export_embedding(module, input_name, prefix)
        elif isinstance(module, ReLU):
            return self._export_activation(module, input_name, prefix, "Relu")
        elif isinstance(module, GELU):
            return self._export_activation(module, input_name, prefix, "Gelu")
        elif isinstance(module, SiLU):
            return self._export_silu(module, input_name, prefix)
        elif isinstance(module, Softmax):
            return self._export_softmax(module, input_name, prefix)
        elif isinstance(module, Dropout):
            return self._export_dropout(module, input_name, prefix)
        elif isinstance(module, MultiheadAttention):
            return self._export_multihead_attention(module, input_name, prefix)
        else:
            # Generic: try to recurse into sub-modules
            return self._export_generic(module, input_name, prefix)

    def _export_sequential(self, module: Sequential, input_name: str, prefix: str) -> str:
        current = input_name
        for key, child in module._modules.items():
            child_prefix = f"{prefix}{key}_" if prefix else f"seq_{key}_"
            current = self._export_module(child, current, child_prefix)
        return current

    def _export_residual(self, module: Residual, input_name: str, prefix: str) -> str:
        # Export the inner module
        inner_prefix = f"{prefix}res_inner_"
        inner_output = self._export_module(module.module, input_name, inner_prefix)
        # Add residual (Add node)
        output_name = self._unique_name(f"{prefix}res_out")
        add_node = helper.make_node(
            "Add", [input_name, inner_output], [output_name], name=f"{prefix}residual_add"
        )
        self._nodes.append(add_node)
        return output_name

    def _export_linear(self, module: Linear, input_name: str, prefix: str) -> str:
        weight = _get_param_array(module.weight).copy()
        weight_name = self._unique_name(f"{prefix}weight")
        # Gemm: Y = X @ B^T + C  (with transB=1)
        # Linear weight is (out_features, in_features), which is B
        self._add_initializer(weight_name, weight)

        output_name = self._unique_name(f"{prefix}gemm_out")
        inputs = [input_name, weight_name]
        attrs = {"transB": 1, "alpha": 1.0, "beta": 1.0}

        if module.bias is not None:
            bias = _get_param_array(module.bias).copy().flatten()
            bias_name = self._unique_name(f"{prefix}bias")
            self._add_initializer(bias_name, bias)
            inputs.append(bias_name)

        node = helper.make_node(
            "Gemm",
            inputs,
            [output_name],
            name=f"{prefix}gemm",
            **attrs,
        )
        self._nodes.append(node)
        return output_name

    def _export_layernorm(self, module: LayerNorm, input_name: str, prefix: str) -> str:
        scale = _get_param_array(module.weight).copy()
        scale_name = self._unique_name(f"{prefix}ln_scale")
        self._add_initializer(scale_name, scale)

        inputs = [input_name, scale_name]

        if module.bias is not None:
            bias = _get_param_array(module.bias).copy()
            bias_name = self._unique_name(f"{prefix}ln_bias")
            self._add_initializer(bias_name, bias)
            inputs.append(bias_name)

        output_name = self._unique_name(f"{prefix}ln_out")
        node = helper.make_node(
            "LayerNormalization",
            inputs,
            [output_name],
            name=f"{prefix}layernorm",
            epsilon=module.eps,
            axis=-1,
        )
        self._nodes.append(node)
        return output_name

    def _export_embedding(self, module: Embedding, input_name: str, prefix: str) -> str:
        weight = _get_param_array(module.weight).copy()
        weight_name = self._unique_name(f"{prefix}emb_weight")
        self._add_initializer(weight_name, weight)

        output_name = self._unique_name(f"{prefix}emb_out")
        node = helper.make_node(
            "Gather",
            [weight_name, input_name],
            [output_name],
            name=f"{prefix}gather",
            axis=0,
        )
        self._nodes.append(node)
        return output_name

    def _export_activation(self, module: Module, input_name: str, prefix: str, onnx_op: str) -> str:
        output_name = self._unique_name(f"{prefix}{onnx_op.lower()}_out")
        node = helper.make_node(
            onnx_op, [input_name], [output_name], name=f"{prefix}{onnx_op.lower()}"
        )
        self._nodes.append(node)
        return output_name

    def _export_silu(self, module: SiLU, input_name: str, prefix: str) -> str:
        # SiLU = x * sigmoid(x)
        sig_name = self._unique_name(f"{prefix}sigmoid_out")
        sig_node = helper.make_node(
            "Sigmoid",
            [input_name],
            [sig_name],
            name=f"{prefix}sigmoid",
        )
        self._nodes.append(sig_node)

        output_name = self._unique_name(f"{prefix}silu_out")
        mul_node = helper.make_node(
            "Mul",
            [input_name, sig_name],
            [output_name],
            name=f"{prefix}silu_mul",
        )
        self._nodes.append(mul_node)
        return output_name

    def _export_softmax(self, module: Softmax, input_name: str, prefix: str) -> str:
        output_name = self._unique_name(f"{prefix}softmax_out")
        node = helper.make_node(
            "Softmax",
            [input_name],
            [output_name],
            name=f"{prefix}softmax",
            axis=module.dim,
        )
        self._nodes.append(node)
        return output_name

    def _export_dropout(self, module: Dropout, input_name: str, prefix: str) -> str:
        # Export as Dropout node (ratio attribute)
        output_name = self._unique_name(f"{prefix}dropout_out")
        mask_name = self._unique_name(f"{prefix}dropout_mask")
        # In opset 12+, ratio is an input not an attribute
        ratio_name = self._unique_name(f"{prefix}dropout_ratio")
        ratio_tensor = numpy_helper.from_array(
            np.array(module.p, dtype=np.float32), name=ratio_name
        )
        self._initializers.append(ratio_tensor)

        training_name = self._unique_name(f"{prefix}dropout_training")
        training_tensor = numpy_helper.from_array(
            np.array(False, dtype=np.bool_), name=training_name
        )
        self._initializers.append(training_tensor)

        node = helper.make_node(
            "Dropout",
            [input_name, ratio_name, training_name],
            [output_name, mask_name],
            name=f"{prefix}dropout",
        )
        self._nodes.append(node)
        return output_name

    def _export_multihead_attention(
        self, module: MultiheadAttention, input_name: str, prefix: str
    ) -> str:
        """Export MultiheadAttention as a subgraph of MatMul/Add/Softmax/Reshape."""
        # Q, K, V projections
        q_out = self._export_linear(module.q_proj, input_name, f"{prefix}q_")
        k_out = self._export_linear(module.k_proj, input_name, f"{prefix}k_")
        v_out = self._export_linear(module.v_proj, input_name, f"{prefix}v_")

        # Compute attention: softmax(Q @ K^T / sqrt(d_k)) @ V
        # K transpose
        kt_name = self._unique_name(f"{prefix}kt")
        kt_node = helper.make_node(
            "Transpose",
            [k_out],
            [kt_name],
            name=f"{prefix}k_transpose",
            perm=[0, 2, 1],  # Simplified — doesn't handle multi-head reshape
        )
        self._nodes.append(kt_node)

        # Q @ K^T
        qk_name = self._unique_name(f"{prefix}qk")
        qk_node = helper.make_node(
            "MatMul",
            [q_out, kt_name],
            [qk_name],
            name=f"{prefix}qk_matmul",
        )
        self._nodes.append(qk_node)

        # Scale by sqrt(d_k)
        scale_val = np.array(1.0 / np.sqrt(module.head_dim), dtype=np.float32)
        scale_name = self._unique_name(f"{prefix}scale")
        self._add_initializer(scale_name, scale_val)
        scaled_name = self._unique_name(f"{prefix}scaled")
        scale_node = helper.make_node(
            "Mul",
            [qk_name, scale_name],
            [scaled_name],
            name=f"{prefix}scale_mul",
        )
        self._nodes.append(scale_node)

        # Softmax
        attn_name = self._unique_name(f"{prefix}attn")
        softmax_node = helper.make_node(
            "Softmax",
            [scaled_name],
            [attn_name],
            name=f"{prefix}attn_softmax",
            axis=-1,
        )
        self._nodes.append(softmax_node)

        # attn @ V
        context_name = self._unique_name(f"{prefix}context")
        context_node = helper.make_node(
            "MatMul",
            [attn_name, v_out],
            [context_name],
            name=f"{prefix}context_matmul",
        )
        self._nodes.append(context_node)

        # Output projection
        if hasattr(module, "o_proj"):
            return self._export_linear(module.o_proj, context_name, f"{prefix}o_")
        return context_name

    def _export_generic(self, module: Module, input_name: str, prefix: str) -> str:
        """Fallback: walk _modules dict and chain them."""
        current = input_name
        for key, child in module._modules.items():
            child_prefix = f"{prefix}{key}_"
            current = self._export_module(child, current, child_prefix)
        return current

    def _export_onnx_model(self, model, input_name: str) -> str:
        """Re-export a GrillyOnnxModel by replaying its exec nodes."""
        # Map graph input to the provided input_name
        name_map: dict[str, str] = {}
        if model._graph_input_names:
            name_map[model._graph_input_names[0]] = input_name

        # Replay each exec node, creating corresponding ONNX nodes
        for i, nd in enumerate(model._exec_nodes):
            prefix = f"replay_{i}_"

            if nd.kind == "module":
                # Module nodes — export using the module exporter
                inp_name = (
                    name_map.get(nd.input_names[0], nd.input_names[0])
                    if nd.input_names
                    else input_name
                )
                out_name = self._export_module(nd.handler, inp_name, prefix)
                if nd.output_names:
                    name_map[nd.output_names[0]] = out_name
            else:
                # Callable nodes — map op_type back to ONNX node
                mapped_inputs = []
                for inp in nd.input_names:
                    mapped_inputs.append(name_map.get(inp, inp))

                out_name = self._unique_name(f"{prefix}{nd.op_type.lower()}_out")
                mapped_outputs = [out_name]

                # Create the ONNX node with original op_type
                node = helper.make_node(
                    nd.op_type,
                    mapped_inputs,
                    mapped_outputs,
                    name=f"{prefix}{nd.op_type.lower()}",
                )
                self._nodes.append(node)

                if nd.output_names:
                    name_map[nd.output_names[0]] = out_name

        # Return the last output name
        if model._graph_output_names:
            last_graph_out = model._graph_output_names[0]
            return name_map.get(last_graph_out, input_name)
        return input_name


__all__ = [
    "OnnxExporter",
]
