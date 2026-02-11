"""
Device Management Utilities
"""

_current_device: str | None = "vulkan"


def get_device() -> str:
    """
    Get current device.

    Returns:
        Current device name ('vulkan', 'cuda', 'cpu', 'llama-cpp')
    """
    return _current_device


def set_device(device: str) -> None:
    """
    Set current device.

    Args:
        device: Device name ('vulkan', 'cuda', 'cpu', 'llama-cpp')
    """
    global _current_device
    device = device.lower()
    if device in ("vulkan", "cuda", "cpu", "llama-cpp"):
        _current_device = device
    else:
        raise ValueError(f"Unknown device: {device}")


def device_count() -> int:
    """
    Get number of available devices.

    Returns:
        Number of devices
    """
    try:
        device = get_device()

        if device == "vulkan":
            try:
                from ..backend.base import VULKAN_AVAILABLE

                if VULKAN_AVAILABLE:
                    # Vulkan is available, return 1 for now
                    # Full implementation would enumerate physical devices
                    return 1
            except Exception:
                pass
        elif device == "cuda":
            try:
                import torch

                return torch.cuda.device_count()
            except Exception:
                pass

        return 1
    except Exception:
        return 1
