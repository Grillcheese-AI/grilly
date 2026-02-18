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

    def VK_MAKE_VERSION(a, b, c):
        return 0


for _name, _default in _VULKAN_FALLBACKS.items():
    if _name not in globals():
        globals()[_name] = _default

import logging

_logger = logging.getLogger(__name__)

# Import buffer pool components (lazy - may not be available)
try:
    from .buffer_pool import PYVMA_AVAILABLE, BufferPool, PooledBuffer, VMABuffer, VMABufferPool

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

    __slots__ = ("handle", "memory", "size")

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
    _weight_cache = None  # Lazily initialized weight cache
    _WEIGHT_CACHE_MAX = 32

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
            max_bucket_power = getattr(pool, "MAX_BUCKET_POWER", None)
            if max_bucket_power is not None:
                max_bucket_size = 1 << int(max_bucket_power)
                if size > max_bucket_size:
                    # Oversized allocations must bypass power-of-two pool bucketing.
                    # Otherwise a capped bucket can under-allocate and overflow on upload.
                    handle, memory = self.core._create_buffer(size, usage)
                    return _DirectBuffer(handle, memory, size)
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

    def _upload_buffer_raw(self, buf, data: np.ndarray):
        """Upload raw numpy data (any dtype) without fp32 conversion."""
        if self._is_vma_buffer(buf):
            pool = self.buffer_pool
            if pool is not None and isinstance(pool, VMABufferPool):
                pool.upload_data_raw(buf, data)
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
    # Weight pinning cache
    # ------------------------------------------------------------------
    def _get_or_upload_weight(self, data):
        """Return cached GPU buffer for weight data. Upload only on cache miss.

        Cache key uses (id, shape, ctypes.data pointer) â€” the pointer changes
        when the optimizer modifies weights in-place, forcing a re-upload.

        Returns:
            (buffer, release: bool) â€” release is always False (cache owns buffer)
        """
        if self._weight_cache is None:
            self._weight_cache = {}

        key = (id(data), data.shape, data.ctypes.data)

        if key in self._weight_cache:
            return self._weight_cache[key][0], False

        # Cache miss: upload
        flat = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
        buf = self._acquire_buffer(flat.nbytes)
        self._upload_buffer(buf, flat)

        # Evict oldest if over limit
        if len(self._weight_cache) >= self._WEIGHT_CACHE_MAX:
            oldest_key = next(iter(self._weight_cache))
            old_buf, _ = self._weight_cache.pop(oldest_key)
            self._release_buffer(old_buf)

        self._weight_cache[key] = (buf, flat.nbytes)
        return buf, False

    def clear_weight_cache(self):
        """Release all cached weight buffers."""
        if self._weight_cache:
            for buf, _ in self._weight_cache.values():
                self._release_buffer(buf)
            self._weight_cache.clear()

    # ------------------------------------------------------------------
    # DEVICE_LOCAL buffer helpers
    # ------------------------------------------------------------------
    def _acquire_device_local_buffer(self, size: int, usage: int = None):
        """Acquire a DEVICE_LOCAL (VRAM) buffer. Falls back to HOST_VISIBLE."""
        pool = self.buffer_pool
        if pool is not None and hasattr(pool, "acquire_device_local"):
            return pool.acquire_device_local(size, usage)
        return self._acquire_buffer(size, usage)

    def _upload_to_device_local(self, device_local_buf, data, recorder=None):
        """Stage-upload *data* into a DEVICE_LOCAL buffer.

        1. Acquire staging buffer (CPU_TO_GPU, mapped).
        2. memcpy data into staging via persistent mapping.
        3. Record vkCmdCopyBuffer(staging → device_local).
        4. If *recorder* is None, submit immediately and release staging.
           Otherwise the caller owns submit and must release staging later.

        Returns:
            staging_buf if recorder is provided (caller must release), else None.
        """
        flat = np.ascontiguousarray(data, dtype=np.float32).ravel()
        pool = self.buffer_pool

        staging = pool.acquire_staging(flat.nbytes, for_upload=True)
        pool.upload_data(staging, flat)

        staging_handle = self._get_buffer_handle(staging)
        dl_handle = self._get_buffer_handle(device_local_buf)

        if recorder is None:
            with self.core.record_commands() as rec:
                rec.copy_buffer(staging_handle, dl_handle, flat.nbytes)
            self._release_buffer(staging)
            return None
        else:
            recorder.copy_buffer(staging_handle, dl_handle, flat.nbytes)
            return staging  # caller must release after submit

    def _download_from_device_local(self, device_local_buf, size, dtype=np.float32,
                                     recorder=None):
        """Stage-download data from a DEVICE_LOCAL buffer.

        1. Acquire readback buffer (GPU_TO_CPU, mapped).
        2. Record transfer_barrier + vkCmdCopyBuffer(device_local → readback).
        3. Submit + wait.
        4. memcpy from readback → numpy.

        Returns:
            numpy array with downloaded data.
        """
        pool = self.buffer_pool
        readback = pool.acquire_staging(size, for_upload=False)

        dl_handle = self._get_buffer_handle(device_local_buf)
        rb_handle = self._get_buffer_handle(readback)

        if recorder is None:
            with self.core.record_commands() as rec:
                rec.transfer_barrier()
                rec.copy_buffer(dl_handle, rb_handle, size)
        else:
            recorder.transfer_barrier()
            recorder.copy_buffer(dl_handle, rb_handle, size)
            recorder.submit_and_wait()

        result = pool.download_data(readback, size, dtype)
        self._release_buffer(readback)
        return result

    def _get_or_upload_weight_device_local(self, data):
        """Like _get_or_upload_weight but places the weight in DEVICE_LOCAL VRAM.

        Returns:
            (buffer, release: bool) — release is always False (cache owns buffer).
        """
        if self._weight_cache is None:
            self._weight_cache = {}

        key = (id(data), data.shape, data.ctypes.data)

        if key in self._weight_cache:
            return self._weight_cache[key][0], False

        flat = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
        buf = self._acquire_device_local_buffer(flat.nbytes)

        if getattr(buf, "is_device_local", False):
            self._upload_to_device_local(buf, flat)
        else:
            self._upload_buffer(buf, flat)

        if len(self._weight_cache) >= self._WEIGHT_CACHE_MAX:
            oldest_key = next(iter(self._weight_cache))
            old_buf, _ = self._weight_cache.pop(oldest_key)
            self._release_buffer(old_buf)

        self._weight_cache[key] = (buf, flat.nbytes)
        return buf, False

    def _wrap_output_tensor_device_local(self, buf, shape):
        """Wrap a DEVICE_LOCAL buffer in a VulkanTensor."""
        from ..utils.tensor_conversion import VulkanTensor

        vt = VulkanTensor.empty(shape)
        vt._pooled_buffer = buf
        vt._gpu_buffer = self._get_buffer_handle(buf)
        vt._gpu_memory = getattr(buf, "memory", None)
        vt._core = self.core
        vt._gpu_valid = True
        vt._cpu_valid = False
        vt._is_device_local = True
        return vt

    # ------------------------------------------------------------------
    # GPU-resident tensor helpers (Phase 3)
    # ------------------------------------------------------------------
    def _prepare_input(self, data, size=None):
        """Accept numpy array or VulkanTensor, return (buffer, needs_release).

        When *data* is a VulkanTensor already on GPU, returns its existing
        buffer without copying. Otherwise allocates a pooled buffer and uploads.
        """
        # Avoid hard import at module level - VulkanTensor lives in utils
        from ..utils.tensor_conversion import VulkanTensor

        if isinstance(data, VulkanTensor):
            if data.on_gpu:
                buf = data._pooled_buffer if data._pooled_buffer is not None else None
                if buf is not None:
                    return buf, False
                # Has raw GPU buffer but not pooled - wrap for API compat
                return _DirectBuffer(data._gpu_buffer, data._gpu_memory, data.nbytes), False
            # CPU-backed lazy VulkanTensor
            arr = np.asarray(data.numpy(), dtype=np.float32).reshape(-1)
        else:
            arr = np.asarray(data)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            arr = np.ascontiguousarray(arr).reshape(-1)

        buf = self._acquire_buffer(arr.nbytes if size is None else size)
        self._upload_buffer(buf, arr)
        return buf, True

    def _wrap_output_tensor(self, buf, shape):
        """Wrap a pooled output buffer in a VulkanTensor (no download)."""
        from ..utils.tensor_conversion import VulkanTensor

        vt = VulkanTensor.empty(shape)
        vt._pooled_buffer = buf
        vt._gpu_buffer = self._get_buffer_handle(buf)
        vt._gpu_memory = getattr(buf, "memory", None)
        vt._core = self.core  # Fast download path (avoids Compute() re-init)
        vt._gpu_valid = True
        vt._cpu_valid = False
        return vt

    def _prepare_output(self, size):
        """Create an output buffer of *size* bytes."""
        return self._acquire_buffer(size)


__all__ = ["VULKAN_AVAILABLE", "BufferMixin", "_DirectBuffer", "BUFFER_POOL_AVAILABLE"]

