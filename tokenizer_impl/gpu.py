"""GPU tokenizer backend (extension points; kernels land in shaders + C++ bridge)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# Set GRILLY_GPU_TOKENIZER=1 when native GPU tokenization is wired (future).
GPU_TOKENIZER_AVAILABLE = os.environ.get("GRILLY_GPU_TOKENIZER", "").strip() in {
    "1",
    "true",
    "yes",
    "on",
}


def numpy_to_input_ids_buffers(
    encoded: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Normalize batch output to numpy int32 arrays (CPU staging before GPU upload)."""
    out: dict[str, np.ndarray] = {}
    if "input_ids" in encoded:
        ids = encoded["input_ids"]
        if hasattr(ids, "numpy"):
            ids = ids.numpy()
        out["input_ids"] = np.asarray(ids, dtype=np.int32)
    if "attention_mask" in encoded:
        m = encoded["attention_mask"]
        if hasattr(m, "numpy"):
            m = m.numpy()
        out["attention_mask"] = np.asarray(m, dtype=np.int32)
    return out


def wrap_ids_as_vulkan_tensors(
    encoded_numpy: dict[str, np.ndarray],
    *,
    device: Any = None,
) -> dict[str, Any]:
    """Upload token buffers with :class:`grilly.utils.tensor_conversion.VulkanTensor` when available.

    Raises ``RuntimeError`` if Vulkan is unavailable or GPU tokenizer path not enabled.
    """
    if not GPU_TOKENIZER_AVAILABLE:
        raise RuntimeError(
            "GPU-resident tokenizer buffers are not enabled yet "
            "(set GRILLY_GPU_TOKENIZER=1 when implemented, or use numpy_to_input_ids_buffers)."
        )
    try:
        from grilly.utils.tensor_conversion import VulkanTensor, gpu_mode
    except ImportError as e:
        raise RuntimeError("VulkanTensor not available") from e

    if device is None:
        gpu_mode(True)
    out: dict[str, Any] = {}
    for k, arr in encoded_numpy.items():
        out[k] = VulkanTensor(arr)
    return out
