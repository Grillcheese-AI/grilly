"""
GPU Buffer Pool for Vulkan Operations using VMA (Vulkan Memory Allocator)

Implements efficient buffer reuse with AMD-optimized memory management.
Uses PyVMA (Python wrapper for VMA) when available.

Key Features:
1. VMA-backed allocation with sub-allocation from large memory blocks
2. AMD/NVIDIA/Intel optimized memory heap selection
3. Persistent mapping support for frequent CPU<->GPU transfers
4. Size-based bucketing with LRU eviction
5. Thread-safe operations
6. Automatic cleanup on context destruction

Installation:
    See grilly/scripts/install_pyvma.py for automated installation.
    Or manually: pip install pyvma (after building vk_mem_alloc.lib)

See: https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from .base import VULKAN_AVAILABLE

if TYPE_CHECKING:
    from .core import VulkanCore

logger = logging.getLogger(__name__)

# Check for PyVMA availability
PYVMA_AVAILABLE = False
pyvma = None
pyvma_lib = None

try:
    import pyvma2 as pyvma

    # pyvma2 exports: ffi, lib, vma (lib alias)
    pyvma_lib = getattr(pyvma, "vma", None) or getattr(pyvma, "lib", None)
    PYVMA_AVAILABLE = pyvma_lib is not None and hasattr(pyvma_lib, "vmaCreateAllocator")
    if PYVMA_AVAILABLE:
        logger.debug("PyVMA2 (VMA 3.4) available - using VMA for buffer allocation")
except ImportError:
    logger.debug("PyVMA2 not available - using direct Vulkan allocation")

if VULKAN_AVAILABLE:
    from vulkan import (
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_SHARING_MODE_EXCLUSIVE,
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        vkDestroyBuffer,
        vkFreeMemory,
    )


class VMABuffer:
    """
    A buffer allocated via VMA (Vulkan Memory Allocator).

    Provides efficient sub-allocation and AMD-optimized memory selection.
    Uses VMA's mapping functions for CPU<->GPU transfers.
    """

    __slots__ = (
        "handle",
        "allocation",
        "allocation_info",
        "size",
        "mapped_ptr",
        "in_use",
        "last_used",
        "usage_flags",
        "_weak_pool",
        "bucket_size",
        "_vk_handle",
        "is_device_local",
    )

    def __init__(
        self,
        handle,
        allocation,
        allocation_info,
        size: int,
        bucket_size: int,
        pool: VMABufferPool,
        usage_flags: int,
    ):
        """Initialize the instance."""

        self.handle = handle  # VMA buffer handle (pyvma ffi)
        self._vk_handle = None  # Vulkan handle (vulkan ffi) - lazy converted
        self.allocation = allocation
        self.allocation_info = allocation_info
        self.size = size
        self.bucket_size = bucket_size
        self._weak_pool = weakref.ref(pool) if pool else None
        self.in_use = True
        self.last_used = time.time()
        self.usage_flags = usage_flags
        self.is_device_local = False
        # VMA allocates with MAPPED_BIT — capture the persistent mapping.
        try:
            self.mapped_ptr = allocation_info.pMappedData
        except Exception:
            self.mapped_ptr = None

    @property
    def pool(self):
        """Execute pool."""

        return self._weak_pool() if self._weak_pool else None

    @property
    def memory(self):
        """Compatibility property - returns allocation for VMA mapping"""
        return self.allocation

    def get_vulkan_handle(self):
        """Get buffer handle compatible with vulkan package"""
        if self._vk_handle is None and self.handle is not None:
            import vulkan as vk

            # Convert pyvma handle to vulkan package handle
            handle_int = int(pyvma.ffi.cast("uintptr_t", self.handle))
            self._vk_handle = vk.ffi.cast("VkBuffer", handle_int)
        return self._vk_handle

    def release(self):
        """Return buffer to pool for reuse"""
        if self.in_use:
            self.in_use = False
            self.last_used = time.time()
            pool = self.pool
            if pool:
                pool._return_buffer(self)

    def __del__(self):
        """Auto-release on garbage collection"""
        if getattr(self, "in_use", False):
            self.release()


class VMABufferPool:
    """
    GPU Buffer Pool using VMA (Vulkan Memory Allocator).

    VMA provides:
    - Automatic sub-allocation from large memory blocks
    - AMD/NVIDIA/Intel optimized memory heap selection
    - Persistent mapping support
    - Better fragmentation handling

    Example:
        >>> pool = VMABufferPool(core)
        >>> buf = pool.acquire(1024)  # Gets VMA-allocated buffer
        >>> # ... use buffer ...
        >>> buf.release()  # Returns to pool for reuse
    """

    MIN_BUCKET_POWER = 8  # 256 bytes
    MAX_BUCKET_POWER = 28  # 256 MB
    MAX_BUFFERS_PER_BUCKET = 32
    MAX_POOL_MEMORY = 512 * 1024 * 1024  # 512MB

    def __init__(self, core: VulkanCore, max_memory: int = None):
        """
        Initialize VMA buffer pool.

        Args:
            core: VulkanCore instance
            max_memory: Maximum total memory to keep in pool (bytes)
        """
        self.core = core
        self.max_memory = max_memory or self.MAX_POOL_MEMORY
        self._allocator = None
        self._buckets: dict[int, list[VMABuffer]] = defaultdict(list)
        self._device_local_buckets: dict[int, list[VMABuffer]] = defaultdict(list)
        self._total_pooled_memory = 0
        self._lock = threading.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "allocations": 0,
            "evictions": 0,
            "total_acquired": 0,
            "total_released": 0,
        }

        # Initialize VMA allocator
        self._init_vma()

    def _init_vma(self):
        """Initialize VMA allocator"""
        if not PYVMA_AVAILABLE:
            logger.warning("PyVMA not available, VMA buffer pool will not function")
            return

        try:
            import vulkan as vk

            # VMA 3.x requires a full set of Vulkan function pointers
            core_functions = [
                "vkGetInstanceProcAddr",
                "vkGetDeviceProcAddr",
                "vkGetPhysicalDeviceProperties",
                "vkGetPhysicalDeviceMemoryProperties",
                "vkAllocateMemory",
                "vkFreeMemory",
                "vkMapMemory",
                "vkUnmapMemory",
                "vkFlushMappedMemoryRanges",
                "vkInvalidateMappedMemoryRanges",
                "vkBindBufferMemory",
                "vkBindImageMemory",
                "vkGetBufferMemoryRequirements",
                "vkGetImageMemoryRequirements",
                "vkCreateBuffer",
                "vkDestroyBuffer",
                "vkCreateImage",
                "vkDestroyImage",
                "vkCmdCopyBuffer",
            ]

            init_functions = {
                fn: pyvma.ffi.cast("PFN_" + fn, getattr(vk.lib, fn)) for fn in core_functions
            }

            # Vulkan 1.1 core promotes KHR functions (drop the suffix).
            # VMA's struct uses the KHR-suffixed field names, so map core→KHR.
            khr_from_core = {
                "vkGetBufferMemoryRequirements2KHR": "vkGetBufferMemoryRequirements2",
                "vkGetImageMemoryRequirements2KHR": "vkGetImageMemoryRequirements2",
                "vkBindBufferMemory2KHR": "vkBindBufferMemory2",
                "vkBindImageMemory2KHR": "vkBindImageMemory2",
                "vkGetPhysicalDeviceMemoryProperties2KHR": "vkGetPhysicalDeviceMemoryProperties2",
            }
            for khr_name, core_name in khr_from_core.items():
                try:
                    fn_ptr = getattr(vk.lib, core_name, None)
                    if fn_ptr is not None:
                        init_functions[khr_name] = pyvma.ffi.cast("PFN_" + khr_name, fn_ptr)
                        logger.debug(f"{khr_name} (via core {core_name})")
                except Exception:
                    logger.debug(f"{khr_name} not available")

            vulkan_functions = pyvma.ffi.new("VmaVulkanFunctions*", init_functions)

            # VMA 3.x requires instance + vulkanApiVersion
            VK_API_VERSION_1_1 = (1 << 22) | (1 << 12) | 0  # VK_MAKE_VERSION(1,1,0)
            create_info = pyvma.ffi.new(
                "VmaAllocatorCreateInfo*",
                {
                    "physicalDevice": pyvma.ffi.cast("void*", self.core.physical_device),
                    "device": pyvma.ffi.cast("void*", self.core.device),
                    "instance": pyvma.ffi.cast("void*", self.core.instance),
                    "vulkanApiVersion": VK_API_VERSION_1_1,
                    "pVulkanFunctions": vulkan_functions,
                },
            )

            pAllocator = pyvma.ffi.new("VmaAllocator*")
            result = pyvma_lib.vmaCreateAllocator(create_info, pAllocator)

            if result != 0:  # VK_SUCCESS = 0
                raise RuntimeError(f"vmaCreateAllocator failed with code {result}")

            self._allocator = pAllocator[0]
            logger.info("VMA allocator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize VMA allocator: {e}")
            self._allocator = None

    def _size_to_bucket(self, size: int) -> int:
        """Round size up to nearest power of 2 bucket."""
        if size <= 0:
            return 1 << self.MIN_BUCKET_POWER
        power = max(self.MIN_BUCKET_POWER, (size - 1).bit_length())
        power = min(power, self.MAX_BUCKET_POWER)
        return 1 << power

    def acquire(self, size: int, usage: int = None) -> VMABuffer:
        """
        Acquire a buffer from the pool.

        Args:
            size: Required buffer size in bytes
            usage: Vulkan buffer usage flags (default: storage buffer)

        Returns:
            VMABuffer ready for use
        """
        if usage is None:
            usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

        if self._allocator is None:
            raise RuntimeError(
                "VMA allocator not initialized. Install pyvma: python -m grilly.scripts.install_pyvma"
            )

        bucket_size = self._size_to_bucket(size)

        with self._lock:
            self._stats["total_acquired"] += 1

            # Try to find existing buffer in bucket
            bucket = self._buckets[bucket_size]
            for i, buf in enumerate(bucket):
                if not buf.in_use and buf.usage_flags == usage:
                    buf.in_use = True
                    buf.size = size
                    buf.last_used = time.time()
                    bucket.pop(i)
                    self._total_pooled_memory -= bucket_size
                    self._stats["hits"] += 1
                    return buf

            # No suitable buffer found, create new one via VMA
            self._stats["misses"] += 1
            self._stats["allocations"] += 1
            self._evict_if_needed(bucket_size)

            # Create buffer via VMA using raw CFFI
            buffer_info = pyvma.ffi.new(
                "VkBufferCreateInfo*",
                {
                    "sType": VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    "size": bucket_size,
                    "usage": usage,
                    "sharingMode": VK_SHARING_MODE_EXCLUSIVE,
                },
            )

            # VMA constants
            VMA_MEMORY_USAGE_CPU_TO_GPU = 3
            VMA_ALLOCATION_CREATE_MAPPED_BIT = 0x00000004

            alloc_info = pyvma.ffi.new(
                "VmaAllocationCreateInfo*",
                {
                    "usage": VMA_MEMORY_USAGE_CPU_TO_GPU,
                    "flags": VMA_ALLOCATION_CREATE_MAPPED_BIT,
                },
            )

            pBuffer = pyvma.ffi.new("VkBuffer*")
            pAllocation = pyvma.ffi.new("VmaAllocation*")
            pAllocationInfo = pyvma.ffi.new("VmaAllocationInfo*")

            result = pyvma_lib.vmaCreateBuffer(
                self._allocator, buffer_info, alloc_info, pBuffer, pAllocation, pAllocationInfo
            )

            if result != 0:
                raise RuntimeError(f"vmaCreateBuffer failed with code {result}")

            return VMABuffer(
                handle=pBuffer[0],
                allocation=pAllocation[0],
                allocation_info=pAllocationInfo[0],
                size=size,
                bucket_size=bucket_size,
                pool=self,
                usage_flags=usage,
            )

    def acquire_device_local(self, size: int, usage: int = None) -> VMABuffer:
        """Acquire DEVICE_LOCAL buffer (VRAM only, no CPU mapping).

        Uses VMA_MEMORY_USAGE_GPU_ONLY so the buffer lives in fast GDDR6/HBM
        rather than the small HOST_VISIBLE BAR.  Transfer bits are added
        automatically so the buffer can be a staging copy target/source.

        Falls back to HOST_VISIBLE ``acquire()`` if VRAM allocation fails.
        """
        if usage is None:
            usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        # Ensure transfer bits for staging copies
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001
        VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002
        usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT

        if self._allocator is None:
            return self.acquire(size, usage)

        bucket_size = self._size_to_bucket(size)

        with self._lock:
            self._stats["total_acquired"] += 1

            # Check device-local reuse buckets
            bucket = self._device_local_buckets[bucket_size]
            for i, buf in enumerate(bucket):
                if not buf.in_use and buf.usage_flags == usage:
                    buf.in_use = True
                    buf.size = size
                    buf.last_used = time.time()
                    bucket.pop(i)
                    self._total_pooled_memory -= bucket_size
                    self._stats["hits"] += 1
                    return buf

            self._stats["misses"] += 1
            self._stats["allocations"] += 1
            self._evict_if_needed(bucket_size)

        # Allocate DEVICE_LOCAL via VMA (no MAPPED_BIT — can't map VRAM)
        VMA_MEMORY_USAGE_GPU_ONLY = 1

        buffer_info = pyvma.ffi.new(
            "VkBufferCreateInfo*",
            {
                "sType": VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                "size": bucket_size,
                "usage": usage,
                "sharingMode": VK_SHARING_MODE_EXCLUSIVE,
            },
        )

        alloc_info = pyvma.ffi.new(
            "VmaAllocationCreateInfo*",
            {
                "usage": VMA_MEMORY_USAGE_GPU_ONLY,
                "flags": 0,  # No MAPPED_BIT for DEVICE_LOCAL
            },
        )

        pBuffer = pyvma.ffi.new("VkBuffer*")
        pAllocation = pyvma.ffi.new("VmaAllocation*")
        pAllocationInfo = pyvma.ffi.new("VmaAllocationInfo*")

        try:
            result = pyvma_lib.vmaCreateBuffer(
                self._allocator, buffer_info, alloc_info, pBuffer, pAllocation, pAllocationInfo
            )
            if result != 0:
                raise RuntimeError(f"vmaCreateBuffer (DEVICE_LOCAL) failed: {result}")

            buf = VMABuffer(
                handle=pBuffer[0],
                allocation=pAllocation[0],
                allocation_info=pAllocationInfo[0],
                size=size,
                bucket_size=bucket_size,
                pool=self,
                usage_flags=usage,
            )
            buf.is_device_local = True
            buf.mapped_ptr = None  # DEVICE_LOCAL cannot be mapped
            return buf

        except Exception as exc:
            logger.debug("DEVICE_LOCAL alloc failed (%s), falling back to HOST_VISIBLE", exc)
            return self.acquire(size, usage)

    def acquire_staging(self, size: int, for_upload: bool = True) -> VMABuffer:
        """Acquire staging buffer for CPU<->GPU copies.

        Args:
            size: Buffer size in bytes.
            for_upload: True  = CPU→GPU staging  (CPU_TO_GPU + TRANSFER_SRC)
                        False = GPU→CPU readback (GPU_TO_CPU + TRANSFER_DST)
        """
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001
        VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002
        VMA_MEMORY_USAGE_CPU_TO_GPU = 3
        VMA_MEMORY_USAGE_GPU_TO_CPU = 4
        VMA_ALLOCATION_CREATE_MAPPED_BIT = 0x00000004

        if for_upload:
            vma_usage = VMA_MEMORY_USAGE_CPU_TO_GPU
            vk_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
        else:
            vma_usage = VMA_MEMORY_USAGE_GPU_TO_CPU
            vk_usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT

        if self._allocator is None:
            raise RuntimeError("VMA allocator not initialized")

        bucket_size = self._size_to_bucket(size)

        buffer_info = pyvma.ffi.new(
            "VkBufferCreateInfo*",
            {
                "sType": VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                "size": bucket_size,
                "usage": vk_usage,
                "sharingMode": VK_SHARING_MODE_EXCLUSIVE,
            },
        )

        alloc_ci = pyvma.ffi.new(
            "VmaAllocationCreateInfo*",
            {
                "usage": vma_usage,
                "flags": VMA_ALLOCATION_CREATE_MAPPED_BIT,
            },
        )

        pBuffer = pyvma.ffi.new("VkBuffer*")
        pAllocation = pyvma.ffi.new("VmaAllocation*")
        pAllocationInfo = pyvma.ffi.new("VmaAllocationInfo*")

        result = pyvma_lib.vmaCreateBuffer(
            self._allocator, buffer_info, alloc_ci, pBuffer, pAllocation, pAllocationInfo
        )
        if result != 0:
            raise RuntimeError(f"vmaCreateBuffer (staging) failed: {result}")

        buf = VMABuffer(
            handle=pBuffer[0],
            allocation=pAllocation[0],
            allocation_info=pAllocationInfo[0],
            size=size,
            bucket_size=bucket_size,
            pool=self,
            usage_flags=vk_usage,
        )
        buf.is_device_local = False
        return buf

    def _return_buffer(self, buffer: VMABuffer):
        """Return a buffer to the pool for reuse."""
        # Flush/invalidate VMA memory so the next acquire sees clean state.
        # This prevents stale data from a previous dispatch leaking into a
        # reused buffer (the root cause of the old "backward stability" bug).
        # Skip for DEVICE_LOCAL — they can't be mapped/flushed.
        if not getattr(buffer, "is_device_local", False):
            try:
                if self._allocator and buffer.allocation:
                    pyvma_lib.vmaInvalidateAllocation(
                        self._allocator, buffer.allocation, 0, buffer.bucket_size
                    )
            except Exception:
                pass  # vmaInvalidateAllocation may not exist in older VMA builds

        with self._lock:
            self._stats["total_released"] += 1
            # Route to correct bucket set
            if getattr(buffer, "is_device_local", False):
                bucket = self._device_local_buckets[buffer.bucket_size]
            else:
                bucket = self._buckets[buffer.bucket_size]

            if len(bucket) >= self.MAX_BUFFERS_PER_BUCKET:
                oldest = min(bucket, key=lambda b: b.last_used)
                bucket.remove(oldest)
                self._destroy_buffer(oldest)
                self._stats["evictions"] += 1

            if self._total_pooled_memory + buffer.bucket_size > self.max_memory:
                self._evict_lru(buffer.bucket_size)

            bucket.append(buffer)
            self._total_pooled_memory += buffer.bucket_size

    def _evict_if_needed(self, needed_size: int):
        """Evict buffers if adding needed_size would exceed limit"""
        while self._total_pooled_memory + needed_size > self.max_memory:
            if not self._evict_lru(needed_size):
                break

    def _evict_lru(self, min_size: int) -> bool:
        """Evict least recently used buffer."""
        oldest_buf = None
        oldest_bucket_size = None
        oldest_time = float("inf")
        oldest_is_device_local = False

        for bucket_dict, is_dl in ((self._buckets, False), (self._device_local_buckets, True)):
            for bucket_size, bucket in bucket_dict.items():
                for buf in bucket:
                    if buf.last_used < oldest_time:
                        oldest_time = buf.last_used
                        oldest_buf = buf
                        oldest_bucket_size = bucket_size
                        oldest_is_device_local = is_dl

        if oldest_buf is None:
            return False

        if oldest_is_device_local:
            self._device_local_buckets[oldest_bucket_size].remove(oldest_buf)
        else:
            self._buckets[oldest_bucket_size].remove(oldest_buf)
        self._total_pooled_memory -= oldest_bucket_size
        self._destroy_buffer(oldest_buf)
        self._stats["evictions"] += 1
        return True

    def _destroy_buffer(self, buffer: VMABuffer):
        """Destroy a VMA buffer"""
        try:
            if self._allocator and buffer.handle and buffer.allocation:
                pyvma_lib.vmaDestroyBuffer(self._allocator, buffer.handle, buffer.allocation)
        except Exception as e:
            logger.debug(f"Error destroying VMA buffer: {e}")

    def upload_data(self, buffer: VMABuffer, data: np.ndarray):
        """Upload data to VMA buffer using VMA's memory mapping"""
        if self._allocator is None:
            raise RuntimeError("VMA allocator not initialized")

        flat = np.ascontiguousarray(data, dtype=np.float32).ravel()

        if buffer.mapped_ptr is not None:
            # Fast path: persistent mapping — no map/unmap overhead.
            pyvma.ffi.memmove(buffer.mapped_ptr, pyvma.ffi.from_buffer(flat), flat.nbytes)
            # Flush for non-HOST_COHERENT heaps (safe no-op on coherent memory).
            pyvma_lib.vmaFlushAllocation(self._allocator, buffer.allocation, 0, flat.nbytes)
            return

        # Fallback: explicit map/unmap
        ppData = pyvma.ffi.new("void**")
        result = pyvma_lib.vmaMapMemory(self._allocator, buffer.allocation, ppData)
        if result != 0:
            raise RuntimeError(f"vmaMapMemory failed with code {result}")

        pyvma.ffi.memmove(ppData[0], pyvma.ffi.from_buffer(flat), flat.nbytes)

        pyvma_lib.vmaUnmapMemory(self._allocator, buffer.allocation)

    def upload_data_raw(self, buffer: VMABuffer, data: np.ndarray):
        """Upload data without dtype conversion (for fp16, int8, etc.)."""
        if self._allocator is None:
            raise RuntimeError("VMA allocator not initialized")

        flat = np.ascontiguousarray(data).ravel()

        if buffer.mapped_ptr is not None:
            pyvma.ffi.memmove(buffer.mapped_ptr, pyvma.ffi.from_buffer(flat), flat.nbytes)
            pyvma_lib.vmaFlushAllocation(self._allocator, buffer.allocation, 0, flat.nbytes)
            return

        ppData = pyvma.ffi.new("void**")
        result = pyvma_lib.vmaMapMemory(self._allocator, buffer.allocation, ppData)
        if result != 0:
            raise RuntimeError(f"vmaMapMemory failed with code {result}")
        pyvma.ffi.memmove(ppData[0], pyvma.ffi.from_buffer(flat), flat.nbytes)
        pyvma_lib.vmaUnmapMemory(self._allocator, buffer.allocation)

    def download_data(self, buffer: VMABuffer, size: int, dtype=np.float32) -> np.ndarray:
        """Download data from VMA buffer using VMA's memory mapping"""
        if self._allocator is None:
            raise RuntimeError("VMA allocator not initialized")

        # Map memory
        ppData = pyvma.ffi.new("void**")
        result = pyvma_lib.vmaMapMemory(self._allocator, buffer.allocation, ppData)
        if result != 0:
            raise RuntimeError(f"vmaMapMemory failed with code {result}")

        # Copy to numpy array
        mapped = ppData[0]
        result_array = np.frombuffer(pyvma.ffi.buffer(mapped, size), dtype=dtype).copy()

        # Unmap
        pyvma_lib.vmaUnmapMemory(self._allocator, buffer.allocation)

        return result_array

    def clear(self):
        """Clear all pooled buffers"""
        with self._lock:
            for bucket in self._buckets.values():
                for buf in bucket:
                    self._destroy_buffer(buf)
                bucket.clear()
            for bucket in self._device_local_buckets.values():
                for buf in bucket:
                    self._destroy_buffer(buf)
                bucket.clear()
            self._total_pooled_memory = 0

    def get_stats(self) -> dict:
        """Get pool statistics"""
        with self._lock:
            stats = dict(self._stats)
            stats["total_pooled_memory"] = self._total_pooled_memory
            stats["buckets"] = {
                size: len(bucket) for size, bucket in self._buckets.items() if bucket
            }
            stats["hit_rate"] = stats["hits"] / max(1, stats["hits"] + stats["misses"])
            stats["vma_enabled"] = self._allocator is not None
            return stats

    def __repr__(self):
        """Return a debug representation."""

        stats = self.get_stats()
        return (
            f"VMABufferPool(pooled={stats['total_pooled_memory'] // 1024}KB, "
            f"hit_rate={stats['hit_rate']:.1%}, "
            f"allocs={stats['allocations']}, vma={stats['vma_enabled']})"
        )

    def shutdown(self):
        """Explicit cleanup - call before destroying the Vulkan device."""
        try:
            self.clear()
            if self._allocator and pyvma_lib is not None:
                pyvma_lib.vmaDestroyAllocator(self._allocator)
                self._allocator = None
        except Exception:
            pass

    def __del__(self):
        """Release resources during finalization.

        VMA native calls on an already-destroyed Vulkan device cause
        access violations that Python cannot catch, so we skip cleanup
        here and rely on the OS to reclaim memory at process exit.
        Use ``shutdown()`` for explicit cleanup before device destruction.
        """
        self._allocator = None


# Legacy PooledBuffer for backward compatibility (direct Vulkan allocation)
class PooledBuffer:
    """
    A buffer using direct Vulkan allocation (legacy fallback).
    Used when PyVMA is not available.
    """

    __slots__ = (
        "handle",
        "memory",
        "size",
        "bucket_size",
        "in_use",
        "last_used",
        "usage_flags",
        "_weak_pool",
    )

    def __init__(
        self, handle, memory, size: int, bucket_size: int, pool: BufferPool, usage_flags: int
    ):
        """Initialize the instance."""

        self.handle = handle
        self.memory = memory
        self.size = size
        self.bucket_size = bucket_size
        self._weak_pool = weakref.ref(pool) if pool else None
        self.in_use = True
        self.last_used = time.time()
        self.usage_flags = usage_flags

    @property
    def pool(self):
        """Execute pool."""

        return self._weak_pool() if self._weak_pool else None

    def release(self):
        """Return buffer to pool for reuse"""
        if self.in_use:
            self.in_use = False
            self.last_used = time.time()
            pool = self.pool
            if pool:
                pool._return_buffer(self)

    def __del__(self):
        """Release resources during finalization."""

        if getattr(self, "in_use", False):
            self.release()


class BufferPool:
    """
    Legacy GPU Buffer Pool using direct Vulkan allocation.
    Used as fallback when PyVMA is not available.

    NOTE: This has known issues with AMD GPUs. Use VMABufferPool instead.
    """

    MIN_BUCKET_POWER = 8
    MAX_BUCKET_POWER = 28
    MAX_BUFFERS_PER_BUCKET = 32
    MAX_POOL_MEMORY = 512 * 1024 * 1024

    def __init__(self, core: VulkanCore, max_memory: int = None):
        """Initialize the instance."""

        self.core = core
        self.max_memory = max_memory or self.MAX_POOL_MEMORY
        self._buckets: dict[int, list[PooledBuffer]] = defaultdict(list)
        self._total_pooled_memory = 0
        self._lock = threading.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "allocations": 0,
            "evictions": 0,
            "total_acquired": 0,
            "total_released": 0,
        }

    def _size_to_bucket(self, size: int) -> int:
        """Execute size to bucket."""

        if size <= 0:
            return 1 << self.MIN_BUCKET_POWER
        power = max(self.MIN_BUCKET_POWER, (size - 1).bit_length())
        power = min(power, self.MAX_BUCKET_POWER)
        return 1 << power

    def acquire(self, size: int, usage: int = None) -> PooledBuffer:
        """Execute acquire."""

        if usage is None:
            usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

        bucket_size = self._size_to_bucket(size)

        with self._lock:
            self._stats["total_acquired"] += 1
            bucket = self._buckets[bucket_size]

            for i, buf in enumerate(bucket):
                if not buf.in_use and buf.usage_flags == usage:
                    buf.in_use = True
                    buf.size = size
                    buf.last_used = time.time()
                    bucket.pop(i)
                    self._total_pooled_memory -= bucket_size
                    self._stats["hits"] += 1
                    return buf

            self._stats["misses"] += 1
            self._stats["allocations"] += 1
            self._evict_if_needed(bucket_size)

            handle, memory = self.core._create_buffer(bucket_size, usage)
            return PooledBuffer(
                handle=handle,
                memory=memory,
                size=size,
                bucket_size=bucket_size,
                pool=self,
                usage_flags=usage,
            )

    def _return_buffer(self, buffer: PooledBuffer):
        """Execute return buffer."""

        with self._lock:
            self._stats["total_released"] += 1
            bucket = self._buckets[buffer.bucket_size]

            if len(bucket) >= self.MAX_BUFFERS_PER_BUCKET:
                oldest = min(bucket, key=lambda b: b.last_used)
                bucket.remove(oldest)
                self._destroy_buffer(oldest)
                self._stats["evictions"] += 1

            if self._total_pooled_memory + buffer.bucket_size > self.max_memory:
                self._evict_lru(buffer.bucket_size)

            bucket.append(buffer)
            self._total_pooled_memory += buffer.bucket_size

    def _evict_if_needed(self, needed_size: int):
        """Execute evict if needed."""

        while self._total_pooled_memory + needed_size > self.max_memory:
            if not self._evict_lru(needed_size):
                break

    def _evict_lru(self, min_size: int) -> bool:
        """Execute evict lru."""

        oldest_buf = None
        oldest_bucket_size = None
        oldest_time = float("inf")

        for bucket_size, bucket in self._buckets.items():
            for buf in bucket:
                if buf.last_used < oldest_time:
                    oldest_time = buf.last_used
                    oldest_buf = buf
                    oldest_bucket_size = bucket_size

        if oldest_buf is None:
            return False

        self._buckets[oldest_bucket_size].remove(oldest_buf)
        self._total_pooled_memory -= oldest_bucket_size
        self._destroy_buffer(oldest_buf)
        self._stats["evictions"] += 1
        return True

    def _destroy_buffer(self, buffer: PooledBuffer):
        """Execute destroy buffer."""

        try:
            # Check device is still valid before destruction
            if self.core.device is None:
                return
            if buffer.handle:
                vkDestroyBuffer(self.core.device, buffer.handle, None)
            if buffer.memory:
                vkFreeMemory(self.core.device, buffer.memory, None)
        except Exception:
            pass

    def clear(self):
        """Execute clear."""

        with self._lock:
            for bucket in self._buckets.values():
                for buf in bucket:
                    self._destroy_buffer(buf)
                bucket.clear()
            self._total_pooled_memory = 0

    def get_stats(self) -> dict:
        """Execute get stats."""

        with self._lock:
            stats = dict(self._stats)
            stats["total_pooled_memory"] = self._total_pooled_memory
            stats["buckets"] = {
                size: len(bucket) for size, bucket in self._buckets.items() if bucket
            }
            stats["hit_rate"] = stats["hits"] / max(1, stats["hits"] + stats["misses"])
            stats["vma_enabled"] = False
            return stats

    def __repr__(self):
        """Return a debug representation."""

        stats = self.get_stats()
        return (
            f"BufferPool(pooled={stats['total_pooled_memory'] // 1024}KB, "
            f"hit_rate={stats['hit_rate']:.1%}, "
            f"allocs={stats['allocations']})"
        )

    def shutdown(self):
        """Explicit cleanup - call before destroying the Vulkan device."""
        try:
            self.clear()
        except Exception:
            pass

    def __del__(self):
        """Release resources during finalization.

        Native Vulkan calls on an already-destroyed device cause access
        violations, so skip cleanup here. Use ``shutdown()`` for explicit
        cleanup before device destruction.
        """
        pass


# Global pool instance
_global_pool = None
_pool_lock = threading.Lock()


def _cleanup_global_pool():
    """Cleanup global pool on shutdown"""
    global _global_pool
    if _global_pool is not None:
        try:
            _global_pool.clear()
            _global_pool = None
        except Exception:
            pass


import atexit

atexit.register(_cleanup_global_pool)


def get_buffer_pool(core: VulkanCore = None, use_vma: bool = None):
    """
    Get or create the global buffer pool.

    Args:
        core: VulkanCore instance (required on first call)
        use_vma: If True, use VMA pool when available. Defaults to True
                 when PyVMA is installed (synchronization is handled by
                 vkQueueWaitIdle in _dispatch_compute).

    Returns:
        VMABufferPool if VMA available and use_vma=True, else BufferPool
    """
    if use_vma is None:
        use_vma = PYVMA_AVAILABLE
    global _global_pool

    with _pool_lock:
        # Check if existing pool's core is still valid (same device)
        if _global_pool is not None and core is not None:
            if _global_pool.core is not core:
                # Core changed (new Compute instance), clear and reset pool
                try:
                    _global_pool.clear()
                except Exception:
                    pass
                _global_pool = None

        if _global_pool is None:
            if core is None:
                raise ValueError("VulkanCore required for first buffer pool initialization")

            if use_vma and PYVMA_AVAILABLE:
                _global_pool = VMABufferPool(core)
                logger.info("Using VMA buffer pool (AMD/NVIDIA optimized)")
            else:
                _global_pool = BufferPool(core)
                logger.debug("Using legacy buffer pool")
        return _global_pool


def acquire_buffer(size: int, usage: int = None, core: VulkanCore = None):
    """
    Convenience function to acquire a buffer from the global pool.

    Args:
        size: Required buffer size
        usage: Vulkan usage flags
        core: VulkanCore instance (for lazy pool initialization)

    Returns:
        VMABuffer or PooledBuffer
    """
    pool = get_buffer_pool(core)
    return pool.acquire(size, usage)


def release_buffer(buffer):
    """
    Convenience function to release a buffer back to the pool.

    Args:
        buffer: VMABuffer or PooledBuffer to release
    """
    buffer.release()


def is_vma_available() -> bool:
    """Check if VMA is available for buffer allocation"""
    return PYVMA_AVAILABLE
