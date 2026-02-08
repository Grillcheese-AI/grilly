"""
Base constants and utilities for Vulkan backend.
"""

import numpy as np

_VULKAN_FALLBACKS = {
    "VK_STRUCTURE_TYPE_APPLICATION_INFO": None,
    "VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO": None,
    "VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO": None,
    "VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO": None,
    "VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO": None,
    "VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO": None,
    "VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO": None,
    "VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO": None,
    "VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO": None,
    "VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO": None,
    "VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO": None,
    "VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET": None,
    "VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_INFO": None,
    "VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO": None,
    "VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO": None,
    "VK_STRUCTURE_TYPE_SUBMIT_INFO": None,
    "VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO": None,
    "VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO": None,
    "VK_STRUCTURE_TYPE_PUSH_CONSTANT_RANGE": None,
    "VK_API_VERSION_1_0": 0,
    "VK_QUEUE_COMPUTE_BIT": 0,
    "VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT": 0,
    "VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT": 0,
    "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER": 0,
    "VK_SHADER_STAGE_COMPUTE_BIT": 0,
    "VK_SHARING_MODE_EXCLUSIVE": 0,
    "VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT": 0,
    "VK_MEMORY_PROPERTY_HOST_COHERENT_BIT": 0,
    "VK_BUFFER_USAGE_STORAGE_BUFFER_BIT": 0,
    "VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT": 0,
    "VK_COMMAND_BUFFER_LEVEL_PRIMARY": 0,
    "VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT": 0,
    "VK_PIPELINE_BIND_POINT_COMPUTE": 0,
}

try:
    from vulkan import *
    VULKAN_AVAILABLE = True
    # After 'from vulkan import *', all Vulkan constants are in the namespace
    # and can be imported by other modules using 'from base import VK_...'
except ImportError:
    VULKAN_AVAILABLE = False
    pass

# Create dummy constants for type checking when Vulkan is not available
# or when a mocked `vulkan` module does not define them (e.g. RTD autodoc).
if "VK_MAKE_VERSION" not in globals():
    VK_MAKE_VERSION = lambda a, b, c: 0
for _name, _default in _VULKAN_FALLBACKS.items():
    if _name not in globals():
        globals()[_name] = _default

import logging

_logger = logging.getLogger(__name__)

# Import buffer pool components (lazy - may not be available)
try:
    from .buffer_pool import (
        BufferPool, VMABufferPool, PooledBuffer, VMABuffer, PYVMA_AVAILABLE
    )
    BUFFER_POOL_AVAILABLE = True
except ImportError:
    BUFFER_POOL_AVAILABLE = False
    PYVMA_AVAILABLE = False
    BufferPool = None
    VMABufferPool = None
    PooledBuffer = None
    VMABuffer = None


class _DirectBuffer:
    """Wrapper for direct buffer allocation when pool is unavailable."""
    __slots__ = ('handle', 'memory', 'size')

    def __init__(self, handle, memory, size):
        self.handle = handle
        self.memory = memory
        self.size = size

    def release(self):
        pass

    def destroy(self, device):
        if self.handle:
            vkDestroyBuffer(device, self.handle, None)
            self.handle = None
        if self.memory:
            vkFreeMemory(device, self.memory, None)
            self.memory = None


class BufferMixin:
    """Mixin providing pooled buffer management for backend modules.

    Subclasses must have ``self.core`` (a VulkanCore instance).
    """

    _pool = None  # Per-instance pool, lazily initialized

    @property
    def buffer_pool(self):
        """Get or lazily initialize the buffer pool.

        Prefers VMA when PyVMA is installed; falls back to legacy BufferPool.
        """
        if self._pool is None and BUFFER_POOL_AVAILABLE:
            try:
                if PYVMA_AVAILABLE:
                    self._pool = VMABufferPool(self.core)
                else:
                    self._pool = BufferPool(self.core)
            except Exception as exc:
                _logger.debug("Buffer pool init failed: %s", exc)
                # Fall back to legacy pool if VMA fails
                if PYVMA_AVAILABLE:
                    try:
                        self._pool = BufferPool(self.core)
                    except Exception:
                        pass
        return self._pool

    # ------------------------------------------------------------------
    # Buffer acquire / release
    # ------------------------------------------------------------------
    def _acquire_buffer(self, size: int, usage: int = None):
        """Acquire a buffer from the pool, or allocate directly."""
        if usage is None:
            usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        pool = self.buffer_pool
        if pool is not None:
            return pool.acquire(size, usage)
        handle, memory = self.core._create_buffer(size, usage)
        return _DirectBuffer(handle, memory, size)

    def _release_buffer(self, buf):
        """Return a single buffer to the pool (or destroy it)."""
        self._release_buffers([buf])

    def _release_buffers(self, buffers):
        """Return multiple buffers to the pool (or destroy them)."""
        for buf in buffers:
            if VMABuffer is not None and isinstance(buf, VMABuffer):
                buf.release()
            elif PooledBuffer is not None and isinstance(buf, PooledBuffer):
                buf.release()
            elif isinstance(buf, _DirectBuffer):
                buf.destroy(self.core.device)
            elif isinstance(buf, tuple) and len(buf) == 2:
                handle, memory = buf
                vkDestroyBuffer(self.core.device, handle, None)
                vkFreeMemory(self.core.device, memory, None)

    # ------------------------------------------------------------------
    # Upload / download helpers
    # ------------------------------------------------------------------
    def _is_vma_buffer(self, buf) -> bool:
        return VMABuffer is not None and isinstance(buf, VMABuffer)

    def _upload_buffer(self, buf, data: np.ndarray):
        """Upload numpy data to a buffer."""
        if self._is_vma_buffer(buf):
            pool = self.buffer_pool
            if pool is not None and isinstance(pool, VMABufferPool):
                pool.upload_data(buf, data)
                return
        self.core._upload_buffer(buf.handle, buf.memory, data)

    def _download_buffer(self, buf, size: int, dtype=np.float32) -> np.ndarray:
        """Download data from a buffer."""
        if self._is_vma_buffer(buf):
            pool = self.buffer_pool
            if pool is not None and isinstance(pool, VMABufferPool):
                return pool.download_data(buf, size, dtype)
        return self.core._download_buffer(buf.memory, size, dtype)

    def _get_buffer_handle(self, buf):
        """Get Vulkan-compatible buffer handle."""
        if self._is_vma_buffer(buf):
            return buf.get_vulkan_handle()
        return buf.handle

    # ------------------------------------------------------------------
    # GPU-resident tensor helpers (Phase 3)
    # ------------------------------------------------------------------
    def _prepare_input(self, data, size=None):
        """Accept numpy array or VulkanTensor, return (buffer, needs_release).

        When *data* is a VulkanTensor already on GPU, returns its existing
        buffer without copying.  Otherwise allocates a pooled buffer and
        uploads.
        """
        # Avoid hard import at module level – VulkanTensor lives in utils
        from ..utils.tensor_conversion import VulkanTensor
        if isinstance(data, VulkanTensor) and data.on_gpu:
            buf = data._pooled_buffer if data._pooled_buffer is not None else None
            if buf is not None:
                return buf, False
            # Has raw GPU buffer but not pooled – wrap for API compat
            return _DirectBuffer(data._gpu_buffer, data._gpu_memory, data.nbytes), False
        arr = np.asarray(data, dtype=np.float32).flatten()
        buf = self._acquire_buffer(arr.nbytes if size is None else size)
        self._upload_buffer(buf, arr)
        return buf, True

    def _prepare_output(self, size):
        """Create an output buffer of *size* bytes."""
        return self._acquire_buffer(size)


__all__ = ['VULKAN_AVAILABLE', 'BufferMixin', '_DirectBuffer', 'BUFFER_POOL_AVAILABLE']
