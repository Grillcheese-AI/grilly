"""
Weight Initialization Utilities

Similar to torch.nn.init
"""

import numpy as np


def xavier_uniform_(tensor: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """
    Fill tensor with values from Xavier uniform distribution.

    Args:
        tensor: Tensor to initialize
        gain: Scaling factor

    Returns:
        Initialized tensor
    """
    if tensor.ndim < 2:
        raise ValueError("Xavier initialization requires at least 2 dimensions")

    fan_in = tensor.shape[-1]
    fan_out = tensor.shape[-2] if tensor.ndim > 1 else tensor.shape[0]

    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    tensor[:] = np.random.uniform(-limit, limit, tensor.shape).astype(tensor.dtype)
    return tensor


def xavier_normal_(tensor: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """
    Fill tensor with values from Xavier normal distribution.

    Args:
        tensor: Tensor to initialize
        gain: Scaling factor

    Returns:
        Initialized tensor
    """
    if tensor.ndim < 2:
        raise ValueError("Xavier initialization requires at least 2 dimensions")

    fan_in = tensor.shape[-1]
    fan_out = tensor.shape[-2] if tensor.ndim > 1 else tensor.shape[0]

    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    tensor[:] = np.random.randn(*tensor.shape).astype(tensor.dtype) * std
    return tensor


def kaiming_uniform_(
    tensor: np.ndarray, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
) -> np.ndarray:
    """
    Fill tensor with values from Kaiming uniform distribution.

    Args:
        tensor: Tensor to initialize
        a: Negative slope of rectifier (for leaky ReLU)
        mode: 'fan_in' or 'fan_out'
        nonlinearity: 'relu' or 'leaky_relu'

    Returns:
        Initialized tensor
    """
    if tensor.ndim < 2:
        raise ValueError("Kaiming initialization requires at least 2 dimensions")

    fan_in = tensor.shape[-1]
    fan_out = tensor.shape[-2] if tensor.ndim > 1 else tensor.shape[0]

    fan = fan_in if mode == "fan_in" else fan_out

    if nonlinearity == "leaky_relu":
        gain = np.sqrt(2.0 / (1 + a**2))
    else:  # relu
        gain = np.sqrt(2.0)

    std = gain / np.sqrt(fan)
    limit = np.sqrt(3.0) * std
    tensor[:] = np.random.uniform(-limit, limit, tensor.shape).astype(tensor.dtype)
    return tensor


def kaiming_normal_(
    tensor: np.ndarray, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
) -> np.ndarray:
    """
    Fill tensor with values from Kaiming normal distribution.

    Args:
        tensor: Tensor to initialize
        a: Negative slope of rectifier (for leaky ReLU)
        mode: 'fan_in' or 'fan_out'
        nonlinearity: 'relu' or 'leaky_relu'

    Returns:
        Initialized tensor
    """
    if tensor.ndim < 2:
        raise ValueError("Kaiming initialization requires at least 2 dimensions")

    fan_in = tensor.shape[-1]
    fan_out = tensor.shape[-2] if tensor.ndim > 1 else tensor.shape[0]

    fan = fan_in if mode == "fan_in" else fan_out

    if nonlinearity == "leaky_relu":
        gain = np.sqrt(2.0 / (1 + a**2))
    else:  # relu
        gain = np.sqrt(2.0)

    std = gain / np.sqrt(fan)
    tensor[:] = np.random.randn(*tensor.shape).astype(tensor.dtype) * std
    return tensor


def _writable_array(tensor):
    """Return a writable ndarray view for ``tensor``.

    The torch_api ``Tensor`` / autograd ``Variable`` / ``Parameter`` wrappers
    don't all support ``__setitem__`` directly — but they expose the backing
    ndarray via ``.data``. Detect those and return the underlying buffer so
    in-place init works for both raw arrays and wrapped tensors.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "data") and isinstance(getattr(tensor, "data", None), np.ndarray):
        return tensor.data
    # Last-resort: hope it's array-like and writable.
    return np.asarray(tensor)


def normal_(tensor, mean: float = 0.0, std: float = 1.0):
    """In-place normal distribution (PyTorch ``nn.init.normal_``)."""
    arr = _writable_array(tensor)
    arr[:] = np.random.normal(mean, std, arr.shape).astype(arr.dtype, copy=False)
    return tensor


def uniform_(tensor, a: float = 0.0, b: float = 1.0):
    """In-place uniform distribution (PyTorch ``nn.init.uniform_``)."""
    arr = _writable_array(tensor)
    arr[:] = np.random.uniform(a, b, arr.shape).astype(arr.dtype, copy=False)
    return tensor
