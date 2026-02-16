"""
ONNX Model Loader for Grilly

Parses ONNX protobuf and reconstructs as a Grilly nn.Module graph.
Supports transformer-class ONNX ops (Gemm, MatMul, LayerNorm, Softmax, etc.)
enabling import of models like BERT, GPT-2, and DistilBERT.

Usage:
    from grilly.utils.onnx_loader import OnnxModelLoader

    model = OnnxModelLoader.load("model.onnx")
    output = model(input_data)
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from ..nn.module import Module
from ..nn.modules import (
    Embedding,
    LayerNorm,
    Linear,
    _create_param_wrapper,
)


def _onnx_dtype_to_numpy(onnx_dtype: int):
    """Map ONNX TensorProto data type to numpy dtype."""
    mapping = {
        TensorProto.FLOAT: np.float32,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
        TensorProto.INT16: np.int16,
        TensorProto.INT8: np.int8,
        TensorProto.UINT8: np.uint8,
        TensorProto.BOOL: np.bool_,
        TensorProto.FLOAT16: np.float16,
    }
    return mapping.get(onnx_dtype, np.float32)


class OnnxOpRegistry:
    """Maps ONNX op_type strings to handler functions.

    Each handler takes (node, inputs, initializers, attrs) and returns
    either an nn.Module instance (stateful ops) or a callable (stateless ops).
    """

    def __init__(self):
        self._handlers: dict[str, Callable] = {}
        self._register_defaults()

    def register(self, op_type: str, handler: Callable):
        """Register a handler for an ONNX op type."""
        self._handlers[op_type] = handler

    def get(self, op_type: str) -> Callable | None:
        """Look up a handler by ONNX op type."""
        return self._handlers.get(op_type)

    @property
    def supported_ops(self) -> list[str]:
        """Return list of supported ONNX op types."""
        return list(self._handlers.keys())

    def _register_defaults(self):
        """Register all default ONNX op handlers."""
        self.register("Gemm", _handle_gemm)
        self.register("MatMul", _handle_matmul)
        self.register("Add", _handle_add)
        self.register("Mul", _handle_mul)
        self.register("Sub", _handle_sub)
        self.register("Div", _handle_div)
        self.register("Relu", _handle_relu)
        self.register("Gelu", _handle_gelu)
        self.register("Sigmoid", _handle_sigmoid)
        self.register("Tanh", _handle_tanh)
        self.register("Softmax", _handle_softmax)
        self.register("LayerNormalization", _handle_layernorm)
        self.register("Reshape", _handle_reshape)
        self.register("Transpose", _handle_transpose)
        self.register("Gather", _handle_gather)
        self.register("Unsqueeze", _handle_unsqueeze)
        self.register("Squeeze", _handle_squeeze)
        self.register("Concat", _handle_concat)
        self.register("Split", _handle_split)
        self.register("Slice", _handle_slice)
        self.register("Where", _handle_where)
        self.register("Cast", _handle_cast)
        self.register("Shape", _handle_shape)
        self.register("Pow", _handle_pow)
        self.register("Sqrt", _handle_sqrt)
        self.register("ReduceMean", _handle_reducemean)
        self.register("Erf", _handle_erf)
        self.register("Constant", _handle_constant)
        self.register("ConstantOfShape", _handle_constantofshape)
        self.register("Expand", _handle_expand)
        self.register("Equal", _handle_equal)
        self.register("Less", _handle_less)
        self.register("LessOrEqual", _handle_lessorequal)
        self.register("Greater", _handle_greater)
        self.register("And", _handle_and)
        self.register("Not", _handle_not)
        self.register("Dropout", _handle_dropout)
        self.register("Identity", _handle_identity)
        self.register("Flatten", _handle_flatten)
        self.register("BatchNormalization", _handle_batchnorm)
        self.register("Clip", _handle_clip)
        self.register("Neg", _handle_neg)
        self.register("Abs", _handle_abs)
        self.register("Log", _handle_log)
        self.register("Exp", _handle_exp)
        self.register("Cos", _handle_cos)
        self.register("Sin", _handle_sin)
        self.register("IsNaN", _handle_isnan)
        self.register("Reciprocal", _handle_reciprocal)
        self.register("Min", _handle_min)
        self.register("Max", _handle_max)
        self.register("Ceil", _handle_ceil)
        self.register("Floor", _handle_floor)
        self.register("Range", _handle_range)


# ---------------------------------------------------------------------------
# Op handler helpers
# ---------------------------------------------------------------------------


def _get_attrs(node) -> dict[str, Any]:
    """Extract attributes from an ONNX node as a dict."""
    attrs: dict[str, Any] = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.FLOAT:
            attrs[attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.INT:
            attrs[attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.STRING:
            attrs[attr.name] = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
        elif attr.type == onnx.AttributeProto.INTS:
            attrs[attr.name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            attrs[attr.name] = list(attr.floats)
        elif attr.type == onnx.AttributeProto.TENSOR:
            attrs[attr.name] = numpy_helper.to_array(attr.t)
        elif attr.type == onnx.AttributeProto.GRAPH:
            attrs[attr.name] = attr.g
    return attrs


# ---------------------------------------------------------------------------
# Stateful op handlers â€” create nn.Module instances
# ---------------------------------------------------------------------------


def _handle_gemm(node, inputs, initializers, attrs):
    """Gemm -> nn.Linear with optional transpose."""
    weight_name = node.input[1]
    bias_name = node.input[2] if len(node.input) > 2 else None

    weight = initializers.get(weight_name)
    if weight is None:
        # Weight not in initializers â€” return a runtime callable
        return "callable", _make_gemm_callable(attrs)

    alpha = attrs.get("alpha", 1.0)
    beta = attrs.get("beta", 1.0)
    trans_b = attrs.get("transB", 0)

    # Gemm: Y = alpha * (A @ B_effective) + beta * C
    # Where B_effective = B^T if transB=1, else B
    # nn.Linear stores W as (out_features, in_features), computes x @ W^T
    # So we need w_linear = (out_features, in_features)
    if trans_b:
        # transB=1: Y = X @ B^T, so B is (out, in) â€” already correct for Linear
        w_linear = weight.copy()
    else:
        # transB=0: Y = X @ B, so B is (in, out) â€” need B^T = (out, in)
        w_linear = weight.T.copy()

    out_features, in_features = w_linear.shape

    if alpha != 1.0:
        w_linear = w_linear * alpha

    bias_data = None
    if bias_name and bias_name in initializers:
        bias_data = initializers[bias_name].copy().astype(np.float32)
        if beta != 1.0:
            bias_data = bias_data * beta

    linear = Linear.__new__(Linear)
    Module.__init__(linear)
    linear.in_features = in_features
    linear.out_features = out_features
    linear.weight = _create_param_wrapper(w_linear.astype(np.float32))
    linear.register_parameter("weight", linear.weight)
    if bias_data is not None:
        linear.bias = _create_param_wrapper(bias_data.astype(np.float32).flatten())
        linear.register_parameter("bias", linear.bias)
    else:
        linear.bias = None

    return "module", linear


def _make_gemm_callable(attrs):
    """Return a runtime callable for Gemm when weights are dynamic."""
    alpha = attrs.get("alpha", 1.0)
    beta = attrs.get("beta", 1.0)
    trans_a = attrs.get("transA", 0)
    trans_b = attrs.get("transB", 0)

    def gemm_fn(*args):
        a, b = args[0], args[1]
        c = args[2] if len(args) > 2 else None
        if trans_a:
            a = a.T
        if trans_b:
            b = b.T
        out = alpha * (a @ b)
        if c is not None:
            out = out + beta * c
        return out

    return gemm_fn


def _handle_matmul(node, inputs, initializers, attrs):
    """MatMul -> np.matmul or nn.Linear if one operand is constant."""
    # Check if second operand is a constant weight
    if len(node.input) >= 2 and node.input[1] in initializers:
        weight = initializers[node.input[1]].copy()
        if weight.ndim == 2:
            in_features, out_features = weight.shape
            linear = Linear.__new__(Linear)
            Module.__init__(linear)
            linear.in_features = in_features
            linear.out_features = out_features
            # Linear computes x @ W^T, but MatMul does x @ W, so store W^T
            linear.weight = _create_param_wrapper(weight.T.astype(np.float32))
            linear.register_parameter("weight", linear.weight)
            linear.bias = None
            return "module", linear

    def matmul_fn(*args):
        return np.matmul(args[0], args[1])

    return "callable", matmul_fn


# ---------------------------------------------------------------------------
# Stateful op handlers â€” LayerNorm, Embedding
# ---------------------------------------------------------------------------


def _handle_layernorm(node, inputs, initializers, attrs):
    """LayerNormalization -> nn.LayerNorm."""
    epsilon = attrs.get("epsilon", 1e-5)
    axis = attrs.get("axis", -1)

    scale_name = node.input[1] if len(node.input) > 1 else None
    bias_name = node.input[2] if len(node.input) > 2 else None

    scale = initializers.get(scale_name) if scale_name else None
    bias = initializers.get(bias_name) if bias_name else None

    if scale is not None:
        normalized_shape = scale.shape[-1]
    elif bias is not None:
        normalized_shape = bias.shape[-1]
    else:
        # Fallback: will be determined at runtime
        def layernorm_fn(*args):
            x = args[0]
            s = args[1] if len(args) > 1 else None
            b = args[2] if len(args) > 2 else None
            mean = np.mean(x, axis=axis, keepdims=True)
            var = np.var(x, axis=axis, keepdims=True)
            out = (x - mean) / np.sqrt(var + epsilon)
            if s is not None:
                out = out * s
            if b is not None:
                out = out + b
            return out

        return "callable", layernorm_fn

    ln = LayerNorm.__new__(LayerNorm)
    Module.__init__(ln)
    ln.normalized_shape = int(normalized_shape)
    ln.eps = epsilon
    ln.weight = _create_param_wrapper(scale.astype(np.float32))
    ln.register_parameter("weight", ln.weight)
    if bias is not None:
        ln.bias = _create_param_wrapper(bias.astype(np.float32))
        ln.register_parameter("bias", ln.bias)
    else:
        ln.bias = _create_param_wrapper(np.zeros(normalized_shape, dtype=np.float32))
        ln.register_parameter("bias", ln.bias)

    return "module", ln


def _handle_gather(node, inputs, initializers, attrs):
    """Gather -> nn.Embedding when axis=0 and weight is constant."""
    axis = attrs.get("axis", 0)
    data_name = node.input[0]

    if axis == 0 and data_name in initializers:
        weight = initializers[data_name]
        if weight.ndim == 2:
            num_embeddings, embedding_dim = weight.shape
            emb = Embedding.__new__(Embedding)
            Module.__init__(emb)
            emb.num_embeddings = num_embeddings
            emb.embedding_dim = embedding_dim
            emb.weight = _create_param_wrapper(weight.astype(np.float32))
            emb.register_parameter("weight", emb.weight)
            return "module", emb

    def gather_fn(*args):
        data = args[0]
        indices = args[1].astype(np.intp)
        try:
            return np.take(data, indices, axis=axis)
        except IndexError:
            # Some exported decoder graphs issue gather indices that can exceed
            # dynamic shape tensors in edge cases. Clamp as a robustness fallback.
            dim = int(data.shape[axis]) if axis < data.ndim else 0
            if dim <= 0:
                safe_shape = list(data.shape)
                if axis < len(safe_shape):
                    safe_shape[axis] = int(np.size(indices))
                return np.zeros(tuple(safe_shape), dtype=data.dtype)
            safe_indices = np.clip(indices, 0, dim - 1)
            return np.take(data, safe_indices, axis=axis)

    return "callable", gather_fn


# ---------------------------------------------------------------------------
# Stateless activation handlers
# ---------------------------------------------------------------------------


def _handle_relu(node, inputs, initializers, attrs):
    def relu_fn(*args):
        return np.maximum(args[0], 0)

    return "callable", relu_fn


def _handle_gelu(node, inputs, initializers, attrs):
    def gelu_fn(*args):
        x = args[0]
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    return "callable", gelu_fn


def _handle_sigmoid(node, inputs, initializers, attrs):
    def sigmoid_fn(*args):
        x = args[0].astype(np.float64)
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

    return "callable", sigmoid_fn


def _handle_tanh(node, inputs, initializers, attrs):
    def tanh_fn(*args):
        return np.tanh(args[0])

    return "callable", tanh_fn


def _handle_softmax(node, inputs, initializers, attrs):
    axis = attrs.get("axis", -1)

    def softmax_fn(*args):
        x = args[0]
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    return "callable", softmax_fn


# ---------------------------------------------------------------------------
# Stateless elementwise / math handlers
# ---------------------------------------------------------------------------


def _handle_add(node, inputs, initializers, attrs):
    # If one operand is a constant bias, embed it
    def add_fn(*args):
        return args[0] + args[1]

    return "callable", add_fn


def _handle_mul(node, inputs, initializers, attrs):
    def mul_fn(*args):
        return args[0] * args[1]

    return "callable", mul_fn


def _handle_sub(node, inputs, initializers, attrs):
    def sub_fn(*args):
        return args[0] - args[1]

    return "callable", sub_fn


def _handle_div(node, inputs, initializers, attrs):
    def div_fn(*args):
        return args[0] / args[1]

    return "callable", div_fn


def _handle_pow(node, inputs, initializers, attrs):
    def pow_fn(*args):
        return np.power(args[0], args[1])

    return "callable", pow_fn


def _handle_sqrt(node, inputs, initializers, attrs):
    def sqrt_fn(*args):
        return np.sqrt(args[0])

    return "callable", sqrt_fn


def _handle_neg(node, inputs, initializers, attrs):
    def neg_fn(*args):
        return -args[0]

    return "callable", neg_fn


def _handle_abs(node, inputs, initializers, attrs):
    def abs_fn(*args):
        return np.abs(args[0])

    return "callable", abs_fn


def _handle_log(node, inputs, initializers, attrs):
    def log_fn(*args):
        return np.log(args[0])

    return "callable", log_fn


def _handle_exp(node, inputs, initializers, attrs):
    def exp_fn(*args):
        return np.exp(args[0])

    return "callable", exp_fn


def _handle_cos(node, inputs, initializers, attrs):
    def cos_fn(*args):
        return np.cos(args[0])

    return "callable", cos_fn


def _handle_sin(node, inputs, initializers, attrs):
    def sin_fn(*args):
        return np.sin(args[0])

    return "callable", sin_fn


def _handle_isnan(node, inputs, initializers, attrs):
    def isnan_fn(*args):
        return np.isnan(args[0])

    return "callable", isnan_fn


def _handle_reciprocal(node, inputs, initializers, attrs):
    def reciprocal_fn(*args):
        return 1.0 / args[0]

    return "callable", reciprocal_fn


def _handle_min(node, inputs, initializers, attrs):
    def min_fn(*args):
        result = args[0]
        for a in args[1:]:
            result = np.minimum(result, a)
        return result

    return "callable", min_fn


def _handle_max(node, inputs, initializers, attrs):
    def max_fn(*args):
        result = args[0]
        for a in args[1:]:
            result = np.maximum(result, a)
        return result

    return "callable", max_fn


def _handle_ceil(node, inputs, initializers, attrs):
    def ceil_fn(*args):
        return np.ceil(args[0])

    return "callable", ceil_fn


def _handle_floor(node, inputs, initializers, attrs):
    def floor_fn(*args):
        return np.floor(args[0])

    return "callable", floor_fn


def _handle_clip(node, inputs, initializers, attrs):
    # Clip can take min/max as inputs (opset 11+) or attrs (opset < 11)
    attr_min = attrs.get("min", None)
    attr_max = attrs.get("max", None)

    def clip_fn(*args):
        x = args[0]
        lo = args[1] if len(args) > 1 and args[1] is not None else attr_min
        hi = args[2] if len(args) > 2 and args[2] is not None else attr_max
        if lo is not None:
            lo = float(lo) if np.ndim(lo) == 0 else lo
        if hi is not None:
            hi = float(hi) if np.ndim(hi) == 0 else hi
        return np.clip(x, lo, hi)

    return "callable", clip_fn


# ---------------------------------------------------------------------------
# Shape manipulation handlers
# ---------------------------------------------------------------------------


def _handle_reshape(node, inputs, initializers, attrs):
    def reshape_fn(*args):
        data = args[0]
        shape = args[1]
        if isinstance(shape, np.ndarray):
            shape = tuple(shape.astype(np.int64).tolist())
        try:
            return np.reshape(data, shape)
        except Exception:
            # Robust fallback for exported graphs with dynamic-shape edge cases:
            # coerce by trim/pad to the requested element count.
            shape_arr = np.asarray(shape, dtype=np.int64).reshape(-1)
            target_shape = tuple(int(v) for v in shape_arr.tolist())
            unknown_idx = None
            known_prod = 1
            for i, dim in enumerate(target_shape):
                if dim == -1 and unknown_idx is None:
                    unknown_idx = i
                elif dim > 0:
                    known_prod *= dim
            if unknown_idx is not None:
                inferred = int(np.asarray(data).size // max(known_prod, 1))
                target_shape = list(target_shape)
                target_shape[unknown_idx] = max(1, inferred)
                target_shape = tuple(target_shape)
            target_elems = int(np.prod([max(1, d) for d in target_shape], dtype=np.int64))
            flat = np.asarray(data).reshape(-1)
            if flat.size < target_elems:
                pad = np.zeros(target_elems - flat.size, dtype=flat.dtype)
                flat = np.concatenate([flat, pad], axis=0)
            elif flat.size > target_elems:
                flat = flat[:target_elems]
            return flat.reshape(target_shape)

    return "callable", reshape_fn


def _handle_transpose(node, inputs, initializers, attrs):
    perm = attrs.get("perm", None)

    def transpose_fn(*args):
        if perm is not None:
            return np.transpose(args[0], axes=perm)
        return np.transpose(args[0])

    return "callable", transpose_fn


def _handle_unsqueeze(node, inputs, initializers, attrs):
    axes = attrs.get("axes", None)  # opset < 13

    def unsqueeze_fn(*args):
        x = args[0]
        ax = axes
        if ax is None and len(args) > 1:
            ax = args[1]
        if isinstance(ax, np.ndarray):
            ax = ax.tolist()
        if isinstance(ax, list):
            for a in sorted(ax):
                x = np.expand_dims(x, axis=a)
            return x
        return np.expand_dims(x, axis=int(ax))

    return "callable", unsqueeze_fn


def _handle_squeeze(node, inputs, initializers, attrs):
    axes = attrs.get("axes", None)

    def squeeze_fn(*args):
        x = args[0]
        ax = axes
        if ax is None and len(args) > 1:
            ax = args[1]
        if isinstance(ax, np.ndarray):
            ax = tuple(ax.tolist())
        if ax is not None:
            return np.squeeze(x, axis=tuple(ax) if isinstance(ax, list) else ax)
        return np.squeeze(x)

    return "callable", squeeze_fn


def _handle_concat(node, inputs, initializers, attrs):
    axis = attrs.get("axis", 0)

    def concat_fn(*args):
        return np.concatenate(list(args), axis=axis)

    return "callable", concat_fn


def _handle_split(node, inputs, initializers, attrs):
    axis = attrs.get("axis", 0)
    split_sizes = attrs.get("split", None)

    def split_fn(*args):
        data = args[0]
        sizes = split_sizes
        if sizes is None and len(args) > 1:
            sizes = args[1]
        if isinstance(sizes, np.ndarray):
            sizes = sizes.tolist()
        if sizes is not None:
            # np.split expects indices, not sizes â€” convert
            indices = np.cumsum(sizes)[:-1].tolist()
            return np.split(data, indices, axis=axis)
        # Equal split
        n_out = len(node.output)
        return np.split(data, n_out, axis=axis)

    return "callable", split_fn


def _handle_slice(node, inputs, initializers, attrs):
    def slice_fn(*args):
        data = args[0]
        starts = args[1] if len(args) > 1 else np.array([0])
        ends = args[2] if len(args) > 2 else np.array([data.shape[0]])
        axes = args[3] if len(args) > 3 else None
        steps = args[4] if len(args) > 4 else None

        if isinstance(starts, np.ndarray):
            starts = starts.tolist()
        if isinstance(ends, np.ndarray):
            ends = ends.tolist()
        if isinstance(axes, np.ndarray):
            axes = axes.tolist()
        if isinstance(steps, np.ndarray):
            steps = steps.tolist()

        slices = [slice(None)] * data.ndim
        if axes is None:
            axes = list(range(len(starts)))
        if steps is None:
            steps = [1] * len(axes)

        for ax, start, end, step in zip(axes, starts, ends, steps):
            # Clamp end to valid range
            dim_size = data.shape[ax]
            if end > dim_size:
                end = dim_size
            if end < -dim_size:
                end = -dim_size
            slices[ax] = slice(int(start), int(end), int(step))

        return data[tuple(slices)]

    return "callable", slice_fn


def _handle_flatten(node, inputs, initializers, attrs):
    axis = attrs.get("axis", 1)

    def flatten_fn(*args):
        x = args[0]
        if axis == 0:
            return x.reshape(1, -1)
        shape = x.shape[:axis] + (-1,)
        return x.reshape(shape)

    return "callable", flatten_fn


# ---------------------------------------------------------------------------
# Comparison / logic handlers
# ---------------------------------------------------------------------------


def _handle_where(node, inputs, initializers, attrs):
    def where_fn(*args):
        return np.where(args[0], args[1], args[2])

    return "callable", where_fn


def _handle_equal(node, inputs, initializers, attrs):
    def equal_fn(*args):
        return np.equal(args[0], args[1])

    return "callable", equal_fn


def _handle_less(node, inputs, initializers, attrs):
    def less_fn(*args):
        return np.less(args[0], args[1])

    return "callable", less_fn


def _handle_lessorequal(node, inputs, initializers, attrs):
    def lessorequal_fn(*args):
        return np.less_equal(args[0], args[1])

    return "callable", lessorequal_fn


def _handle_greater(node, inputs, initializers, attrs):
    def greater_fn(*args):
        return np.greater(args[0], args[1])

    return "callable", greater_fn


def _handle_and(node, inputs, initializers, attrs):
    def and_fn(*args):
        return np.logical_and(args[0], args[1])

    return "callable", and_fn


def _handle_not(node, inputs, initializers, attrs):
    def not_fn(*args):
        return np.logical_not(args[0])

    return "callable", not_fn


# ---------------------------------------------------------------------------
# Reduction / type-casting handlers
# ---------------------------------------------------------------------------


def _handle_reducemean(node, inputs, initializers, attrs):
    axes = attrs.get("axes", None)
    keepdims = bool(attrs.get("keepdims", 1))

    def reducemean_fn(*args):
        x = args[0]
        ax = axes
        if ax is None and len(args) > 1:
            ax = args[1]
        if isinstance(ax, np.ndarray):
            ax = tuple(ax.tolist())
        if isinstance(ax, list):
            ax = tuple(ax)
        return np.mean(x, axis=ax, keepdims=keepdims)

    return "callable", reducemean_fn


def _handle_cast(node, inputs, initializers, attrs):
    to_dtype = attrs.get("to", TensorProto.FLOAT)
    np_dtype = _onnx_dtype_to_numpy(to_dtype)

    def cast_fn(*args):
        return args[0].astype(np_dtype)

    return "callable", cast_fn


def _handle_shape(node, inputs, initializers, attrs):
    def shape_fn(*args):
        return np.array(args[0].shape, dtype=np.int64)

    return "callable", shape_fn


def _handle_range(node, inputs, initializers, attrs):
    def range_fn(*args):
        start = np.asarray(args[0]).reshape(-1)[0]
        limit = np.asarray(args[1]).reshape(-1)[0]
        delta = np.asarray(args[2]).reshape(-1)[0]
        out_dtype = np.result_type(
            np.asarray(args[0]).dtype,
            np.asarray(args[1]).dtype,
            np.asarray(args[2]).dtype,
        )
        return np.arange(start, limit, delta, dtype=out_dtype)

    return "callable", range_fn


def _handle_erf(node, inputs, initializers, attrs):
    def erf_fn(*args):
        # Approximate erf for float32 without scipy dependency
        x = args[0].astype(np.float64)
        # Abramowitz and Stegun approximation
        a1, a2, a3, a4, a5 = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
        p = 0.3275911
        sign = np.sign(x)
        x = np.abs(x)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        return (sign * y).astype(np.float32)

    return "callable", erf_fn


# ---------------------------------------------------------------------------
# Constant / initialization handlers
# ---------------------------------------------------------------------------


def _handle_constant(node, inputs, initializers, attrs):
    value = attrs.get("value", None)
    value_float = attrs.get("value_float", None)
    value_int = attrs.get("value_int", None)
    value_floats = attrs.get("value_floats", None)
    value_ints = attrs.get("value_ints", None)

    if value is not None:
        const = value if isinstance(value, np.ndarray) else np.array(value)
    elif value_float is not None:
        const = np.array(value_float, dtype=np.float32)
    elif value_int is not None:
        const = np.array(value_int, dtype=np.int64)
    elif value_floats is not None:
        const = np.array(value_floats, dtype=np.float32)
    elif value_ints is not None:
        const = np.array(value_ints, dtype=np.int64)
    else:
        const = np.array(0.0, dtype=np.float32)

    def constant_fn(*args):
        return const.copy()

    return "callable", constant_fn


def _handle_constantofshape(node, inputs, initializers, attrs):
    value = attrs.get("value", None)
    if value is not None and isinstance(value, np.ndarray):
        fill_val = value.flatten()[0]
    else:
        fill_val = 0.0

    def constantofshape_fn(*args):
        shape = tuple(args[0].astype(np.int64).tolist())
        return np.full(shape, fill_val, dtype=np.float32)

    return "callable", constantofshape_fn


def _handle_expand(node, inputs, initializers, attrs):
    def expand_fn(*args):
        data = args[0]
        shape = tuple(args[1].astype(np.int64).tolist())
        try:
            return np.broadcast_to(data, shape).copy()
        except Exception:
            arr = np.asarray(data)
            if arr.ndim == len(shape):
                # Torch-exported graphs often encode "keep dim" as 1 in Expand
                # after normalizing -1 placeholders; preserve non-unit source dims.
                adjusted_shape = list(shape)
                changed = False
                for dim_idx, (src_dim, tgt_dim) in enumerate(zip(arr.shape, adjusted_shape, strict=False)):
                    if int(tgt_dim) == 1 and int(src_dim) > 1:
                        adjusted_shape[dim_idx] = int(src_dim)
                        changed = True
                if changed:
                    try:
                        return np.broadcast_to(arr, tuple(adjusted_shape)).copy()
                    except Exception:
                        pass

            target_elems = int(np.prod(shape, dtype=np.int64))
            if arr.size == 1:
                return np.full(shape, float(arr.reshape(-1)[0]), dtype=arr.dtype)
            if arr.size == target_elems:
                return arr.reshape(shape).copy()
            flat = arr.reshape(-1)
            if flat.size < target_elems:
                reps = int(np.ceil(target_elems / max(1, flat.size)))
                flat = np.tile(flat, reps)
            flat = flat[:target_elems]
            return flat.reshape(shape).copy()

    return "callable", expand_fn


# ---------------------------------------------------------------------------
# Misc handlers
# ---------------------------------------------------------------------------


def _handle_dropout(node, inputs, initializers, attrs):
    # During inference, dropout is identity
    def dropout_fn(*args):
        return args[0]

    return "callable", dropout_fn


def _handle_identity(node, inputs, initializers, attrs):
    def identity_fn(*args):
        return args[0]

    return "callable", identity_fn


def _handle_batchnorm(node, inputs, initializers, attrs):
    epsilon = attrs.get("epsilon", 1e-5)

    def batchnorm_fn(*args):
        x = args[0]
        scale = args[1] if len(args) > 1 else np.ones(x.shape[1], dtype=np.float32)
        bias = args[2] if len(args) > 2 else np.zeros(x.shape[1], dtype=np.float32)
        mean = args[3] if len(args) > 3 else np.mean(x, axis=(0, 2, 3) if x.ndim == 4 else 0)
        var = args[4] if len(args) > 4 else np.var(x, axis=(0, 2, 3) if x.ndim == 4 else 0)

        if x.ndim == 4:
            # NCHW format
            shape = (1, -1, 1, 1)
        elif x.ndim == 2:
            shape = (1, -1)
        else:
            shape = (1, -1) + (1,) * (x.ndim - 2)

        out = (x - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + epsilon)
        out = out * scale.reshape(shape) + bias.reshape(shape)
        return out

    return "callable", batchnorm_fn


# ===========================================================================
# GrillyOnnxModel â€” the reconstructed model
# ===========================================================================


class _NodeExec:
    """Execution descriptor for a single ONNX node."""

    __slots__ = ("name", "op_type", "input_names", "output_names", "handler", "kind")

    def __init__(self, name, op_type, input_names, output_names, handler, kind):
        self.name = name
        self.op_type = op_type
        self.input_names = input_names
        self.output_names = output_names
        self.handler = handler  # nn.Module or callable
        self.kind = kind  # "module" or "callable"


class GrillyOnnxModel(Module):
    """Grilly Module reconstructed from an ONNX graph.

    Stores the execution order and intermediate tensor names so that
    ``forward()`` can replay the ONNX graph sequentially.
    """

    def __init__(
        self,
        exec_nodes: list[_NodeExec],
        graph_input_names: list[str],
        graph_output_names: list[str],
        constant_tensors: dict[str, np.ndarray],
    ):
        super().__init__()
        self._exec_nodes = exec_nodes
        self._graph_input_names = graph_input_names
        self._graph_output_names = graph_output_names
        self._constant_tensors = constant_tensors

        # Register all module-type nodes as sub-modules so parameters() works
        for i, nd in enumerate(exec_nodes):
            if nd.kind == "module":
                key = nd.name or f"node_{i}"
                # Sanitize key for module registration
                key = key.replace("/", "_").replace(".", "_").replace(":", "_")
                self._modules[key] = nd.handler

    def forward(self, *inputs) -> Any:
        """Execute the ONNX graph.

        Positional ``inputs`` are mapped to the graph's declared inputs
        (in order).  Constants / initializers are pre-loaded.
        """
        # Build the tensor map with graph inputs and constants
        tensor_map: dict[str, Any] = dict(self._constant_tensors)

        for idx, name in enumerate(self._graph_input_names):
            if idx < len(inputs):
                tensor_map[name] = inputs[idx]

        # Execute each node sequentially
        for nd in self._exec_nodes:
            # Gather inputs for this node
            node_inputs = []
            for inp_name in nd.input_names:
                if inp_name == "":
                    node_inputs.append(None)
                elif inp_name in tensor_map:
                    node_inputs.append(tensor_map[inp_name])
                else:
                    node_inputs.append(None)

            # Execute
            if nd.kind == "module":
                # Module forward â€” pass non-None positional args
                non_none = [x for x in node_inputs if x is not None]
                if non_none:
                    # For Gather(data, indices) lowered to nn.Embedding, use indices.
                    if isinstance(nd.handler, Embedding):
                        if len(node_inputs) > 1 and node_inputs[1] is not None:
                            result = nd.handler(node_inputs[1])
                        else:
                            result = nd.handler(non_none[-1])
                    else:
                        result = nd.handler(non_none[0])
                else:
                    result = None
            else:
                result = nd.handler(*node_inputs)

            # Store outputs
            if isinstance(result, (list, tuple)):
                for out_idx, out_name in enumerate(nd.output_names):
                    if out_idx < len(result) and out_name:
                        tensor_map[out_name] = result[out_idx]
            else:
                if nd.output_names:
                    tensor_map[nd.output_names[0]] = result
                    # Some ops produce multiple outputs but we only have one value
                    # (e.g. Dropout in eval produces output + mask)
                    for extra in nd.output_names[1:]:
                        if extra:
                            tensor_map[extra] = result

        # Collect graph outputs
        outputs = []
        for name in self._graph_output_names:
            outputs.append(tensor_map.get(name))

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def get_layer(self, name: str) -> Module | None:
        """Retrieve a named sub-module (e.g. a Linear or LayerNorm layer)."""
        key = name.replace("/", "_").replace(".", "_").replace(":", "_")
        return self._modules.get(key)

    def get_linear_layers(self) -> dict[str, Linear]:
        """Return dict of all Linear layers keyed by their module name."""
        result = {}
        for key, mod in self._modules.items():
            if isinstance(mod, Linear):
                result[key] = mod
        return result

    def __repr__(self):
        n_modules = sum(1 for nd in self._exec_nodes if nd.kind == "module")
        n_callable = sum(1 for nd in self._exec_nodes if nd.kind == "callable")
        return (
            f"GrillyOnnxModel(nodes={len(self._exec_nodes)}, "
            f"modules={n_modules}, callables={n_callable})"
        )


# ===========================================================================
# OnnxModelLoader â€” main entry point
# ===========================================================================


class OnnxModelLoader:
    """Load an ONNX model and convert it to a Grilly Module graph."""

    def __init__(self, registry: OnnxOpRegistry | None = None):
        self.registry = registry or OnnxOpRegistry()

    @classmethod
    def load(cls, path: str, registry: OnnxOpRegistry | None = None) -> GrillyOnnxModel:
        """Load an ONNX model file and return a ``GrillyOnnxModel``.

        Args:
            path: Path to the ``.onnx`` file.
            registry: Optional custom op registry.

        Returns:
            A ``GrillyOnnxModel`` ready for inference.
        """
        loader = cls(registry=registry)
        model_proto = onnx.load(path)
        return loader._build_model(model_proto)

    @classmethod
    def load_from_proto(
        cls, model_proto, registry: OnnxOpRegistry | None = None
    ) -> GrillyOnnxModel:
        """Load from an already-parsed ``ModelProto``."""
        loader = cls(registry=registry)
        return loader._build_model(model_proto)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_model(self, model_proto) -> GrillyOnnxModel:
        graph = model_proto.graph

        # 1. Extract initializers as numpy arrays
        initializers: dict[str, np.ndarray] = {}
        for tensor in graph.initializer:
            initializers[tensor.name] = numpy_helper.to_array(tensor)

        # 2. Determine graph input names (exclude initializer-only inputs)
        init_names = set(initializers.keys())
        graph_input_names = [inp.name for inp in graph.input if inp.name not in init_names]
        graph_output_names = [out.name for out in graph.output]

        # 3. Build execution nodes
        exec_nodes: list[_NodeExec] = []
        for node in graph.node:
            op_type = node.op_type
            attrs = _get_attrs(node)
            handler_fn = self.registry.get(op_type)
            if handler_fn is None:
                # Unknown op â€” create a pass-through with warning
                import warnings

                warnings.warn(f"Unsupported ONNX op: {op_type}, using identity fallback")

                def _identity(*args, _op=op_type):
                    return args[0] if args else None

                exec_nodes.append(
                    _NodeExec(
                        name=node.name or op_type,
                        op_type=op_type,
                        input_names=list(node.input),
                        output_names=list(node.output),
                        handler=_identity,
                        kind="callable",
                    )
                )
                continue

            kind, handler = handler_fn(node, list(node.input), initializers, attrs)

            exec_nodes.append(
                _NodeExec(
                    name=node.name or f"{op_type}_{len(exec_nodes)}",
                    op_type=op_type,
                    input_names=list(node.input),
                    output_names=list(node.output),
                    handler=handler,
                    kind=kind,
                )
            )

        # 4. Build constant tensor map from initializers not consumed by modules
        # We always include initializers since graph execution needs them
        constant_tensors = dict(initializers)

        return GrillyOnnxModel(
            exec_nodes=exec_nodes,
            graph_input_names=graph_input_names,
            graph_output_names=graph_output_names,
            constant_tensors=constant_tensors,
        )


__all__ = [
    "OnnxOpRegistry",
    "OnnxModelLoader",
    "GrillyOnnxModel",
]

