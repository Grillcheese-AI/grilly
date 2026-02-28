"""
Bridge layer: routes Python nn module ops through grilly_core C++ extension.

Instead of the legacy Python ctypes Vulkan path (struct.pack -> vkMapMemory ->
ctypes.memmove -> dispatch -> fence wait -> repeat per op), the bridge calls
pybind11-bound C++ ops that use VMA persistent mapping (single memcpy, zero
vkMap/vkUnmap) and BufferPool bucketed allocation.

Higher-level modules (nn/modules.py, nn/conv.py, nn/normalization.py) call
bridge functions with a try/fallback pattern:
    result = _bridge.linear(x, weight, bias)
    if result is not None: return result
    # else fall through to legacy backend
"""

import os
import numpy as np

try:
    import grilly_core as _core
    _NATIVE = True
except ImportError:
    _core = None
    _NATIVE = False

# ── Lazy singleton device ────────────────────────────────────────────────

_device = None


def _get_device():
    """Lazily initialize the C++ Vulkan device and load shaders."""
    global _device
    if _device is not None:
        return _device
    if not _NATIVE:
        return None
    try:
        _device = _core.Device()
        # Find shaders relative to this file: backend/_bridge.py -> ../../shaders/spv/
        shader_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "shaders", "spv"
        )
        if os.path.isdir(shader_dir):
            _device.load_shaders(shader_dir)
        return _device
    except Exception:
        _device = None
        return None


def is_available():
    """Check if the C++ bridge is available and working."""
    return _get_device() is not None


# ── Helpers ──────────────────────────────────────────────────────────────

def _ensure_f32_contiguous(arr):
    """Ensure array is float32 and C-contiguous for the C++ side."""
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


# ── Linear ───────────────────────────────────────────────────────────────

def linear(x, weight, bias=None):
    """GPU linear: output = x @ W^T + bias. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        x = _ensure_f32_contiguous(x)
        weight = _ensure_f32_contiguous(weight)
        bias = _ensure_f32_contiguous(bias)
        return _core.linear(dev, x, weight, bias)
    except Exception:
        return None


# ── Activations ──────────────────────────────────────────────────────────

def relu(x):
    """GPU ReLU. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.relu(dev, _ensure_f32_contiguous(x))
    except Exception:
        return None


def gelu(x):
    """GPU GELU. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.gelu(dev, _ensure_f32_contiguous(x))
    except Exception:
        return None


def silu(x):
    """GPU SiLU. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        return _core.silu(dev, _ensure_f32_contiguous(x))
    except Exception:
        return None


# ── Conv2d ───────────────────────────────────────────────────────────────

def conv2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0),
           dilation=(1, 1), groups=1):
    """GPU Conv2d. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        x = _ensure_f32_contiguous(x)
        weight = _ensure_f32_contiguous(weight)
        bias = _ensure_f32_contiguous(bias)
        return _core.conv2d(dev, x, weight, bias,
                            list(stride), list(padding),
                            list(dilation), groups)
    except Exception:
        return None


# ── LayerNorm ────────────────────────────────────────────────────────────

def layernorm(x, gamma, beta, eps=1e-5):
    """GPU LayerNorm. Returns None on failure."""
    dev = _get_device()
    if dev is None:
        return None
    try:
        x = _ensure_f32_contiguous(x)
        gamma = _ensure_f32_contiguous(gamma)
        beta = _ensure_f32_contiguous(beta)
        return _core.layernorm(dev, x, gamma, beta, eps)
    except Exception:
        return None
