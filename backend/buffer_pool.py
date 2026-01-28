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

import numpy as np
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Any
import time
import weakref
import logging

from .base import VULKAN_AVAILABLE

logger = logging.getLogger(__name__)

# Check for PyVMA availability
PYVMA_AVAILABLE = False
pyvma = None
pyvma_lib = None

try:
    import pyvma
    # PyVMA exports: ffi, vma (which is the lib)
    pyvma_lib = getattr(pyvma, 'vma', None) or getattr(pyvma, 'lib', None)
    PYVMA_AVAILABLE = pyvma_lib is not None and hasattr(pyvma_lib, 'vmaCreateAllocator')
    if PYVMA_AVAILABLE:
        logger.debug("PyVMA available - using VMA for buffer allocation")
except ImportError:
    logger.debug("PyVMA not available - using direct Vulkan allocation")

if VULKAN_AVAILABLE:
    from vulkan import (
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE,
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        vkDestroyBuffer,
        vkFreeMemory,
        VkBufferCreateInfo,
    )


class VMABuffer:
    """
    A buffer allocated via VMA (Vulkan Memory Allocator).

    Provides efficient sub-allocation and AMD-optimized memory selection.
    Uses VMA's mapping functions for CPU<->GPU transfers.
    """
    __slots__ = ('handle', 'allocation', 'allocation_info', 'size', 'mapped_ptr',
                 'in_use', 'last_used', 'usage_flags', '_weak_pool', 'bucket_size',
                 '_vk_handle')

    def __init__(self, handle, allocation, allocation_info, size: int,
                 bucket_size: int, pool: 'VMABufferPool', usage_flags: int):
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
        self.mapped_ptr = None

    @property
    def pool(self):
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
            handle_int = int(pyvma.ffi.cast('uintptr_t', self.handle))
            self._vk_handle = vk.ffi.cast('VkBuffer', handle_int)
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
        if getattr(self, 'in_use', False):
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

    MIN_BUCKET_POWER = 8   # 256 bytes
    MAX_BUCKET_POWER = 28  # 256 MB
    MAX_BUFFERS_PER_BUCKET = 32
    MAX_POOL_MEMORY = 512 * 1024 * 1024  # 512MB

    def __init__(self, core: 'VulkanCore', max_memory: int = None):
        """
        Initialize VMA buffer pool.

        Args:
            core: VulkanCore instance
            max_memory: Maximum total memory to keep in pool (bytes)
        """
        self.core = core
        self.max_memory = max_memory or self.MAX_POOL_MEMORY
        self._allocator = None
        self._buckets: Dict[int, List[VMABuffer]] = defaultdict(list)
        self._total_pooled_memory = 0
        self._lock = threading.Lock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'allocations': 0,
            'evictions': 0,
            'total_acquired': 0,
            'total_released': 0,
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

            # Create Vulkan functions struct manually (without KHR extensions)
            # This avoids the ProcedureNotFoundError for optional extensions
            core_functions = [
                'vkGetPhysicalDeviceProperties',
                'vkGetPhysicalDeviceMemoryProperties',
                'vkAllocateMemory',
                'vkFreeMemory',
                'vkMapMemory',
                'vkUnmapMemory',
                'vkBindBufferMemory',
                'vkBindImageMemory',
                'vkGetBufferMemoryRequirements',
                'vkGetImageMemoryRequirements',
                'vkCreateBuffer',
                'vkDestroyBuffer',
                'vkCreateImage',
                'vkDestroyImage'
            ]

            init_functions = {
                fn: pyvma.ffi.cast('PFN_' + fn, getattr(vk.lib, fn))
                for fn in core_functions
            }

            # Try to add KHR extension functions if available (optional)
            khr_functions = [
                'vkGetBufferMemoryRequirements2KHR',
                'vkGetImageMemoryRequirements2KHR'
            ]
            for fn_name in khr_functions:
                try:
                    fn_ptr = vk.lib.vkGetDeviceProcAddr(
                        self.core.device,
                        pyvma.ffi.new('char[]', fn_name.encode('ascii'))
                    )
                    if fn_ptr != pyvma.ffi.NULL:
                        init_functions[fn_name] = pyvma.ffi.cast('PFN_' + fn_name, fn_ptr)
                        logger.debug(f"KHR extension {fn_name} available")
                except Exception:
                    logger.debug(f"KHR extension {fn_name} not available (optional)")

            vulkan_functions = pyvma.ffi.new('VmaVulkanFunctions*', init_functions)

            # Create allocator with custom Vulkan functions
            # Note: older VMA versions don't have 'instance' field
            create_info = pyvma.ffi.new('VmaAllocatorCreateInfo*', {
                'physicalDevice': pyvma.ffi.cast('void*', self.core.physical_device),
                'device': pyvma.ffi.cast('void*', self.core.device),
                'pVulkanFunctions': vulkan_functions,
            })

            pAllocator = pyvma.ffi.new('VmaAllocator*')
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
            raise RuntimeError("VMA allocator not initialized. Install pyvma: python -m grilly.scripts.install_pyvma")

        bucket_size = self._size_to_bucket(size)

        with self._lock:
            self._stats['total_acquired'] += 1

            # Try to find existing buffer in bucket
            bucket = self._buckets[bucket_size]
            for i, buf in enumerate(bucket):
                if not buf.in_use and buf.usage_flags == usage:
                    buf.in_use = True
                    buf.size = size
                    buf.last_used = time.time()
                    bucket.pop(i)
                    self._total_pooled_memory -= bucket_size
                    self._stats['hits'] += 1
                    return buf

            # No suitable buffer found, create new one via VMA
            self._stats['misses'] += 1
            self._stats['allocations'] += 1
            self._evict_if_needed(bucket_size)

            # Create buffer via VMA using raw CFFI
            buffer_info = pyvma.ffi.new('VkBufferCreateInfo*', {
                'sType': VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                'size': bucket_size,
                'usage': usage,
                'sharingMode': VK_SHARING_MODE_EXCLUSIVE,
            })

            # VMA constants
            VMA_MEMORY_USAGE_CPU_TO_GPU = 3
            VMA_ALLOCATION_CREATE_MAPPED_BIT = 0x00000004

            alloc_info = pyvma.ffi.new('VmaAllocationCreateInfo*', {
                'usage': VMA_MEMORY_USAGE_CPU_TO_GPU,
                'flags': VMA_ALLOCATION_CREATE_MAPPED_BIT,
            })

            pBuffer = pyvma.ffi.new('VkBuffer*')
            pAllocation = pyvma.ffi.new('VmaAllocation*')
            pAllocationInfo = pyvma.ffi.new('VmaAllocationInfo*')

            result = pyvma_lib.vmaCreateBuffer(
                self._allocator, buffer_info, alloc_info,
                pBuffer, pAllocation, pAllocationInfo
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
                usage_flags=usage
            )

    def _return_buffer(self, buffer: VMABuffer):
        """Return a buffer to the pool for reuse."""
        with self._lock:
            self._stats['total_released'] += 1
            bucket = self._buckets[buffer.bucket_size]

            if len(bucket) >= self.MAX_BUFFERS_PER_BUCKET:
                oldest = min(bucket, key=lambda b: b.last_used)
                bucket.remove(oldest)
                self._destroy_buffer(oldest)
                self._stats['evictions'] += 1

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
        oldest_time = float('inf')

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
        self._stats['evictions'] += 1
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

        # Map memory
        ppData = pyvma.ffi.new('void**')
        result = pyvma_lib.vmaMapMemory(self._allocator, buffer.allocation, ppData)
        if result != 0:
            raise RuntimeError(f"vmaMapMemory failed with code {result}")

        # Copy data
        pyvma.ffi.memmove(ppData[0], data.tobytes(), data.nbytes)

        # Unmap
        pyvma_lib.vmaUnmapMemory(self._allocator, buffer.allocation)

    def download_data(self, buffer: VMABuffer, size: int, dtype=np.float32) -> np.ndarray:
        """Download data from VMA buffer using VMA's memory mapping"""
        if self._allocator is None:
            raise RuntimeError("VMA allocator not initialized")

        # Map memory
        ppData = pyvma.ffi.new('void**')
        result = pyvma_lib.vmaMapMemory(self._allocator, buffer.allocation, ppData)
        if result != 0:
            raise RuntimeError(f"vmaMapMemory failed with code {result}")

        # Copy to numpy array
        mapped = ppData[0]
        result_array = np.frombuffer(
            pyvma.ffi.buffer(mapped, size),
            dtype=dtype
        ).copy()

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
            self._total_pooled_memory = 0

    def get_stats(self) -> dict:
        """Get pool statistics"""
        with self._lock:
            stats = dict(self._stats)
            stats['total_pooled_memory'] = self._total_pooled_memory
            stats['buckets'] = {
                size: len(bucket)
                for size, bucket in self._buckets.items()
                if bucket
            }
            stats['hit_rate'] = (
                stats['hits'] / max(1, stats['hits'] + stats['misses'])
            )
            stats['vma_enabled'] = self._allocator is not None
            return stats

    def __repr__(self):
        stats = self.get_stats()
        return (
            f"VMABufferPool(pooled={stats['total_pooled_memory']//1024}KB, "
            f"hit_rate={stats['hit_rate']:.1%}, "
            f"allocs={stats['allocations']}, vma={stats['vma_enabled']})"
        )

    def __del__(self):
        """Cleanup on destruction - clear buffers before destroying allocator"""
        try:
            self.clear()
            if self._allocator and pyvma_lib is not None:
                pyvma_lib.vmaDestroyAllocator(self._allocator)
                self._allocator = None
        except Exception:
            pass  # Ignore errors during shutdown


# Legacy PooledBuffer for backward compatibility (direct Vulkan allocation)
class PooledBuffer:
    """
    A buffer using direct Vulkan allocation (legacy fallback).
    Used when PyVMA is not available.
    """
    __slots__ = ('handle', 'memory', 'size', 'bucket_size', 'in_use',
                 'last_used', 'usage_flags', '_weak_pool')

    def __init__(self, handle, memory, size: int, bucket_size: int,
                 pool: 'BufferPool', usage_flags: int):
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
        if getattr(self, 'in_use', False):
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

    def __init__(self, core: 'VulkanCore', max_memory: int = None):
        self.core = core
        self.max_memory = max_memory or self.MAX_POOL_MEMORY
        self._buckets: Dict[int, List[PooledBuffer]] = defaultdict(list)
        self._total_pooled_memory = 0
        self._lock = threading.Lock()
        self._stats = {
            'hits': 0, 'misses': 0, 'allocations': 0,
            'evictions': 0, 'total_acquired': 0, 'total_released': 0,
        }

    def _size_to_bucket(self, size: int) -> int:
        if size <= 0:
            return 1 << self.MIN_BUCKET_POWER
        power = max(self.MIN_BUCKET_POWER, (size - 1).bit_length())
        power = min(power, self.MAX_BUCKET_POWER)
        return 1 << power

    def acquire(self, size: int, usage: int = None) -> PooledBuffer:
        if usage is None:
            usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

        bucket_size = self._size_to_bucket(size)

        with self._lock:
            self._stats['total_acquired'] += 1
            bucket = self._buckets[bucket_size]

            for i, buf in enumerate(bucket):
                if not buf.in_use and buf.usage_flags == usage:
                    buf.in_use = True
                    buf.size = size
                    buf.last_used = time.time()
                    bucket.pop(i)
                    self._total_pooled_memory -= bucket_size
                    self._stats['hits'] += 1
                    return buf

            self._stats['misses'] += 1
            self._stats['allocations'] += 1
            self._evict_if_needed(bucket_size)

            handle, memory = self.core._create_buffer(bucket_size, usage)
            return PooledBuffer(
                handle=handle, memory=memory, size=size,
                bucket_size=bucket_size, pool=self, usage_flags=usage
            )

    def _return_buffer(self, buffer: PooledBuffer):
        with self._lock:
            self._stats['total_released'] += 1
            bucket = self._buckets[buffer.bucket_size]

            if len(bucket) >= self.MAX_BUFFERS_PER_BUCKET:
                oldest = min(bucket, key=lambda b: b.last_used)
                bucket.remove(oldest)
                self._destroy_buffer(oldest)
                self._stats['evictions'] += 1

            if self._total_pooled_memory + buffer.bucket_size > self.max_memory:
                self._evict_lru(buffer.bucket_size)

            bucket.append(buffer)
            self._total_pooled_memory += buffer.bucket_size

    def _evict_if_needed(self, needed_size: int):
        while self._total_pooled_memory + needed_size > self.max_memory:
            if not self._evict_lru(needed_size):
                break

    def _evict_lru(self, min_size: int) -> bool:
        oldest_buf = None
        oldest_bucket_size = None
        oldest_time = float('inf')

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
        self._stats['evictions'] += 1
        return True

    def _destroy_buffer(self, buffer: PooledBuffer):
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
        with self._lock:
            for bucket in self._buckets.values():
                for buf in bucket:
                    self._destroy_buffer(buf)
                bucket.clear()
            self._total_pooled_memory = 0

    def get_stats(self) -> dict:
        with self._lock:
            stats = dict(self._stats)
            stats['total_pooled_memory'] = self._total_pooled_memory
            stats['buckets'] = {
                size: len(bucket)
                for size, bucket in self._buckets.items()
                if bucket
            }
            stats['hit_rate'] = (
                stats['hits'] / max(1, stats['hits'] + stats['misses'])
            )
            stats['vma_enabled'] = False
            return stats

    def __repr__(self):
        stats = self.get_stats()
        return (
            f"BufferPool(pooled={stats['total_pooled_memory']//1024}KB, "
            f"hit_rate={stats['hit_rate']:.1%}, "
            f"allocs={stats['allocations']})"
        )

    def __del__(self):
        self.clear()


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


def get_buffer_pool(core: 'VulkanCore' = None, use_vma: bool = False):
    """
    Get or create the global buffer pool.

    Args:
        core: VulkanCore instance (required on first call)
        use_vma: If True, use VMA pool when available.
                 NOTE: VMA disabled by default due to stability issues with
                 backward operations. Use legacy BufferPool for now.

    Returns:
        VMABufferPool if VMA available and use_vma=True, else BufferPool
    """
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


def acquire_buffer(size: int, usage: int = None, core: 'VulkanCore' = None):
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
