"""``torch.cuda``-free availability helper (Vulkan)."""

from __future__ import annotations

try:
    from grilly.backend.base import VULKAN_AVAILABLE
except Exception:
    VULKAN_AVAILABLE = False


def is_available() -> bool:
    return bool(VULKAN_AVAILABLE)


__all__ = ["is_available"]
