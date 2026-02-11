"""
Common PyTorch Operations (80% Coverage)

Provides implementations of the most commonly used PyTorch operations
using Vulkan backend. Covers approximately 80% of common PyTorch usage.
"""

import numpy as np

from .device_manager import get_device_manager
from .pytorch_compat import Tensor, tensor, to_numpy


def _get_backend():
    """Get Vulkan backend"""
    device_manager = get_device_manager()
    return device_manager.vulkan


# ============================================================================
# Basic Operations
# ============================================================================


def add(input: Tensor | np.ndarray, other: Tensor | np.ndarray, alpha: float = 1.0) -> Tensor:
    """Add tensors: output = input + alpha * other"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    other_arr = to_numpy(other) if not isinstance(other, np.ndarray) else other
    return tensor(input_arr + alpha * other_arr)


def mul(input: Tensor | np.ndarray, other: Tensor | np.ndarray) -> Tensor:
    """Multiply tensors element-wise"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    other_arr = to_numpy(other) if not isinstance(other, np.ndarray) else other
    return tensor(input_arr * other_arr)


def matmul(input: Tensor | np.ndarray, other: Tensor | np.ndarray) -> Tensor:
    """Matrix multiplication"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    other_arr = to_numpy(other) if not isinstance(other, np.ndarray) else other
    backend = _get_backend()

    # Try GPU shader if available
    if hasattr(backend, "linear"):
        try:
            # Use linear backend (matmul is similar)
            result = backend.linear(input_arr, other_arr.T, None)
            return tensor(result)
        except Exception:
            pass

    # CPU fallback
    return tensor(input_arr @ other_arr)


def bmm(input: Tensor | np.ndarray, mat2: Tensor | np.ndarray) -> Tensor:
    """Batch matrix multiplication"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    mat2_arr = to_numpy(mat2) if not isinstance(mat2, np.ndarray) else mat2

    # Batch matmul: (batch, n, m) @ (batch, m, p) -> (batch, n, p)
    return tensor(np.einsum("bij,bjk->bik", input_arr, mat2_arr))


# ============================================================================
# Activation Functions
# ============================================================================


def relu(input: Tensor | np.ndarray) -> Tensor:
    """ReLU activation"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    backend = _get_backend()

    if hasattr(backend, "activation_relu"):
        try:
            return tensor(backend.activation_relu(input_arr))
        except Exception:
            pass

    return tensor(np.maximum(input_arr, 0))


def gelu(input: Tensor | np.ndarray) -> Tensor:
    """GELU activation"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    backend = _get_backend()

    if hasattr(backend, "activation_gelu"):
        try:
            return tensor(backend.activation_gelu(input_arr))
        except Exception:
            pass

    # CPU fallback
    return tensor(
        0.5 * input_arr * (1 + np.tanh(np.sqrt(2 / np.pi) * (input_arr + 0.044715 * input_arr**3)))
    )


def softmax(input: Tensor | np.ndarray, dim: int = -1) -> Tensor:
    """Softmax activation"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    backend = _get_backend()

    if hasattr(backend, "activation_softmax"):
        try:
            return tensor(backend.activation_softmax(input_arr))
        except Exception:
            pass

    # CPU fallback
    exp = np.exp(input_arr - np.max(input_arr, axis=dim, keepdims=True))
    return tensor(exp / np.sum(exp, axis=dim, keepdims=True))


def sigmoid(input: Tensor | np.ndarray) -> Tensor:
    """Sigmoid activation"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    return tensor(1.0 / (1.0 + np.exp(-input_arr)))


def tanh(input: Tensor | np.ndarray) -> Tensor:
    """Tanh activation"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    return tensor(np.tanh(input_arr))


# ============================================================================
# Normalization
# ============================================================================


def layer_norm(
    input: Tensor | np.ndarray,
    normalized_shape: tuple[int, ...],
    weight: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Layer normalization"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    backend = _get_backend()

    if hasattr(backend, "layernorm") and weight is not None and bias is not None:
        try:
            return tensor(backend.layernorm(input_arr, weight, bias, eps))
        except Exception:
            pass

    # CPU fallback
    mean = np.mean(input_arr, axis=-1, keepdims=True)
    var = np.var(input_arr, axis=-1, keepdims=True)
    normalized = (input_arr - mean) / np.sqrt(var + eps)
    if weight is not None and bias is not None:
        normalized = normalized * weight + bias
    return tensor(normalized)


def batch_norm(
    input: Tensor | np.ndarray,
    running_mean: np.ndarray | None = None,
    running_var: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """Batch normalization"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input

    if training:
        # Compute batch statistics
        mean = np.mean(input_arr, axis=(0, 2, 3) if input_arr.ndim == 4 else 0, keepdims=True)
        var = np.var(input_arr, axis=(0, 2, 3) if input_arr.ndim == 4 else 0, keepdims=True)

        # Update running stats
        if running_mean is not None and running_var is not None:
            running_mean[:] = (1 - momentum) * running_mean + momentum * mean.squeeze()
            running_var[:] = (1 - momentum) * running_var + momentum * var.squeeze()
        use_mean = mean
        use_var = var
    else:
        if running_mean is not None and running_var is not None:
            use_mean = (
                running_mean.reshape(1, -1, 1, 1)
                if input_arr.ndim == 4
                else running_mean.reshape(1, -1)
            )
            use_var = (
                running_var.reshape(1, -1, 1, 1)
                if input_arr.ndim == 4
                else running_var.reshape(1, -1)
            )
        else:
            use_mean = np.mean(
                input_arr, axis=(0, 2, 3) if input_arr.ndim == 4 else 0, keepdims=True
            )
            use_var = np.var(input_arr, axis=(0, 2, 3) if input_arr.ndim == 4 else 0, keepdims=True)

    # Normalize
    normalized = (input_arr - use_mean) / np.sqrt(use_var + eps)
    if weight is not None and bias is not None:
        # Reshape weight and bias for broadcasting
        if input_arr.ndim == 4:
            # (batch, channels, height, width)
            weight_reshaped = weight.reshape(1, -1, 1, 1)
            bias_reshaped = bias.reshape(1, -1, 1, 1)
        else:
            # (batch, features)
            weight_reshaped = weight.reshape(1, -1)
            bias_reshaped = bias.reshape(1, -1)
        normalized = normalized * weight_reshaped + bias_reshaped

    return tensor(normalized)


# ============================================================================
# Dropout
# ============================================================================


def dropout(
    input: Tensor | np.ndarray, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Dropout"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    backend = _get_backend()

    if hasattr(backend, "dropout"):
        try:
            return tensor(backend.dropout(input_arr, p, training))
        except Exception:
            pass

    # CPU fallback
    if not training or p == 0:
        return tensor(input_arr)

    mask = np.random.binomial(1, 1 - p, input_arr.shape).astype(np.float32)
    scale = 1.0 / (1 - p)
    result = input_arr * mask * scale
    return tensor(result)


# ============================================================================
# Convolution (2D)
# ============================================================================


def conv2d(
    input: Tensor | np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
) -> Tensor:
    """
    2D Convolution (CPU fallback for now)

    Args:
        input: Input tensor (batch, in_channels, height, width)
        weight: Weight tensor (out_channels, in_channels, kernel_h, kernel_w)
        bias: Optional bias (out_channels,)
        stride: Stride (h, w)
        padding: Padding (h, w)
        dilation: Dilation (h, w)
        groups: Number of groups
    """
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input

    # CPU implementation (simplified)
    batch, in_channels, in_h, in_w = input_arr.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1

    # Add padding
    if padding[0] > 0 or padding[1] > 0:
        input_arr = np.pad(
            input_arr,
            ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
            mode="constant",
        )

    output = np.zeros((batch, out_channels, out_h, out_w), dtype=np.float32)

    for b in range(batch):
        for oc in range(out_channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * stride[0]
                    w_start = ow * stride[1]
                    h_end = h_start + kernel_h
                    w_end = w_start + kernel_w

                    output[b, oc, oh, ow] = np.sum(
                        input_arr[b, :, h_start:h_end, w_start:w_end] * weight[oc]
                    )
                    if bias is not None:
                        output[b, oc, oh, ow] += bias[oc]

    return tensor(output)


# ============================================================================
# Pooling
# ============================================================================


def max_pool2d(
    input: Tensor | np.ndarray,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = (0, 0),
) -> Tensor:
    """2D Max pooling"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    batch, channels, in_h, in_w = input_arr.shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    out_h = (in_h + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - kernel_w) // stride_w + 1

    if pad_h > 0 or pad_w > 0:
        input_arr = np.pad(
            input_arr,
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=-np.inf,
        )

    output = np.zeros((batch, channels, out_h, out_w), dtype=np.float32)

    for b in range(batch):
        for c in range(channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * stride_h
                    w_start = ow * stride_w
                    h_end = h_start + kernel_h
                    w_end = w_start + kernel_w

                    output[b, c, oh, ow] = np.max(input_arr[b, c, h_start:h_end, w_start:w_end])

    return tensor(output)


def avg_pool2d(
    input: Tensor | np.ndarray,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = (0, 0),
) -> Tensor:
    """2D Average pooling"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    batch, channels, in_h, in_w = input_arr.shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    out_h = (in_h + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - kernel_w) // stride_w + 1

    if pad_h > 0 or pad_w > 0:
        input_arr = np.pad(
            input_arr, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )

    output = np.zeros((batch, channels, out_h, out_w), dtype=np.float32)

    for b in range(batch):
        for c in range(channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * stride_h
                    w_start = ow * stride_w
                    h_end = h_start + kernel_h
                    w_end = w_start + kernel_w

                    output[b, c, oh, ow] = np.mean(input_arr[b, c, h_start:h_end, w_start:w_end])

    return tensor(output)


# ============================================================================
# Loss Functions
# ============================================================================


def mse_loss(
    input: Tensor | np.ndarray, target: Tensor | np.ndarray, reduction: str = "mean"
) -> Tensor:
    """Mean Squared Error loss"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    target_arr = to_numpy(target) if not isinstance(target, np.ndarray) else target

    loss = (input_arr - target_arr) ** 2

    if reduction == "mean":
        return tensor(np.mean(loss))
    elif reduction == "sum":
        return tensor(np.sum(loss))
    else:
        return tensor(loss)


def cross_entropy_loss(
    input: Tensor | np.ndarray,
    target: Tensor | np.ndarray,
    weight: np.ndarray | None = None,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> Tensor:
    """Cross entropy loss"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    target_arr = to_numpy(target) if not isinstance(target, np.ndarray) else target

    # Softmax
    exp = np.exp(input_arr - np.max(input_arr, axis=-1, keepdims=True))
    probs = exp / np.sum(exp, axis=-1, keepdims=True)

    # Log probabilities
    log_probs = np.log(probs + 1e-8)

    # One-hot encode targets
    if target_arr.ndim == 1:
        batch_size, num_classes = input_arr.shape
        one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
        one_hot[np.arange(batch_size), target_arr] = 1.0
    else:
        one_hot = target_arr

    # Compute loss
    loss = -np.sum(one_hot * log_probs, axis=-1)

    if weight is not None:
        loss = loss * weight[target_arr.astype(int)]

    if reduction == "mean":
        return tensor(np.mean(loss))
    elif reduction == "sum":
        return tensor(np.sum(loss))
    else:
        return tensor(loss)


# ============================================================================
# Utility Functions
# ============================================================================


def flatten(input: Tensor | np.ndarray, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flatten tensor"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input

    if end_dim == -1:
        end_dim = len(input_arr.shape) - 1

    shape = list(input_arr.shape)
    new_shape = shape[:start_dim] + [-1] + shape[end_dim + 1 :]

    return tensor(input_arr.reshape(new_shape))


def reshape(input: Tensor | np.ndarray, shape: tuple[int, ...]) -> Tensor:
    """Reshape tensor"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    return tensor(input_arr.reshape(shape))


def transpose(input: Tensor | np.ndarray, dim0: int, dim1: int) -> Tensor:
    """Transpose tensor dimensions"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    axes = list(range(len(input_arr.shape)))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return tensor(np.transpose(input_arr, axes))


def unsqueeze(input: Tensor | np.ndarray, dim: int) -> Tensor:
    """Add dimension"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    return tensor(np.expand_dims(input_arr, dim))


def squeeze(input: Tensor | np.ndarray, dim: int | None = None) -> Tensor:
    """Remove dimension"""
    input_arr = to_numpy(input) if not isinstance(input, np.ndarray) else input
    if dim is not None:
        return tensor(np.squeeze(input_arr, axis=dim))
    return tensor(np.squeeze(input_arr))
