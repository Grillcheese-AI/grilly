"""
AdditionLinear — multiplication-free linear layer using L1 distance.

output[b, o] = -sum_k |weight[o, k] - input[b, k]| + bias[o]

Uses: addition-linear.glsl (compute shader).
Backward uses the subgradient: d/dx |a - b| = -sign(a - b).
"""

import numpy as np

from ._helpers import (
    _USE_CPP_BRIDGE,
    _bridge,
    _bridge_to_numpy,
    _create_param_wrapper,
    _get_param_array,
)
from .module import Module


class AdditionLinear(Module):
    """Multiplication-free linear layer (L1 / Manhattan distance).

    ``y = -||W - x||_1 + b`` per output neuron.  Uses the ``addition-linear``
    Vulkan compute shader when available, with a NumPy CPU fallback.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        limit = np.sqrt(6.0 / (in_features + out_features))
        weight_data = np.random.uniform(
            -limit, limit, (out_features, in_features)
        ).astype(np.float32)

        self.weight = _create_param_wrapper(weight_data)
        self.register_parameter("weight", self.weight)

        if bias:
            self.bias_param = _create_param_wrapper(
                np.zeros(out_features, dtype=np.float32)
            )
            self.register_parameter("bias", self.bias_param)
        else:
            self.bias_param = None

        self._last_input = None

    @property
    def d_in(self) -> int:
        """Alias for ``in_features`` (PyTorch-style naming)."""
        return self.in_features

    def forward(self, x) -> np.ndarray:
        x_arr = np.ascontiguousarray(x, dtype=np.float32)
        self._last_input = x_arr

        w = _get_param_array(self.weight)
        b = _get_param_array(self.bias_param) if self.bias_param is not None else None

        if _USE_CPP_BRIDGE and hasattr(_bridge, "addition_linear"):
            result = _bridge_to_numpy(_bridge.addition_linear(x_arr, w, b))
            if result is not None:
                return result

        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        original_shape = x_arr.shape
        if x_arr.ndim > 2:
            batch_seq = int(np.prod(original_shape[:-1]))
            x_2d = x_arr.reshape(batch_seq, original_shape[-1])
        else:
            batch_seq = original_shape[0]
            x_2d = x_arr

        out = np.empty((batch_seq, self.out_features), dtype=np.float32)
        for o in range(self.out_features):
            dist = np.sum(np.abs(w[o] - x_2d), axis=1)
            out[:, o] = -dist
            if b is not None:
                out[:, o] += b[o]

        if len(original_shape) > 2:
            return out.reshape(original_shape[:-1] + (self.out_features,))
        return out

    def backward(self, grad_output: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """Backward pass — subgradient of L1 distance.

        grad_input[s, k] = -sum_o( grad_out[s, o] * sign(W[o, k] - x[s, k]) )
        grad_W[o, k]     = -sum_s( grad_out[s, o] * sign(W[o, k] - x[s, k]) )
        grad_b[o]        = sum_s( grad_out[s, o] )
        """
        if x is None:
            x = self._last_input
        if x is None:
            raise RuntimeError("backward requires input; call forward first or pass x=")

        x_arr = np.asarray(x, dtype=np.float32)
        w = _get_param_array(self.weight)
        go = np.asarray(grad_output, dtype=np.float32)

        if x_arr.ndim > 2:
            S = int(np.prod(x_arr.shape[:-1]))
            x_2d = x_arr.reshape(S, -1)
            go_2d = go.reshape(S, -1)
        else:
            x_2d = x_arr if x_arr.ndim == 2 else x_arr.reshape(1, -1)
            go_2d = go if go.ndim == 2 else go.reshape(1, -1)

        S, d_in = x_2d.shape
        d_out = self.out_features

        grad_input = np.zeros_like(x_2d)
        grad_w = np.zeros_like(w)

        for o in range(d_out):
            sgn = np.sign(w[o][np.newaxis, :] - x_2d)  # (S, d_in)
            g_col = go_2d[:, o:o+1]  # (S, 1)
            grad_input -= g_col * sgn
            grad_w[o] -= np.sum(g_col * sgn, axis=0)

        if self.weight is not None:
            if not hasattr(self.weight, "grad") or self.weight.grad is None:
                self.weight.grad = grad_w
            else:
                self.weight.grad += grad_w

        if self.bias_param is not None:
            grad_b = np.sum(go_2d, axis=0)
            if not hasattr(self.bias_param, "grad") or self.bias_param.grad is None:
                self.bias_param.grad = grad_b
            else:
                self.bias_param.grad += grad_b

        if x_arr.ndim > 2:
            return grad_input.reshape(x_arr.shape)
        return grad_input

    def __repr__(self):
        return (
            f"AdditionLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias_param is not None})"
        )
