"""
GPU Buffer Pool for Vulkan Operations

Implements buffer reuse to avoid repeated allocation/deallocation overhead.
Uses size-based bucketing for efficient buffer matching.

Key Features:
1. Size-bucketed buffer pools (powers of 2)
2. LRU eviction for memory management
3. Thread-safe operations
4. Automatic cleanup on context destruction
"""

import numpy as np
import threading
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import time
import weakref

from .base import VULKAN_AVAILABLE

if VULKAN_AVAILABLE:
    from vulkan import (
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    )


class PooledBuffer:
    """
    A buffer acquired from the pool.

    Automatically returns to pool when released or garbage collected.
    """
    __slots__ = ('handle', 'memory', 'size', 'bucket_size', 'pool', 'in_use',
                 'last_used', 'usage_flags', '_weak_pool')

    def __init__(self, handle, memory, size: int, bucket_size: int,
                 pool: 'BufferPool', usage_flags: int):
        self.handle = handle
        self.memory = memory
        self.size = size  # Actual requested size
        self.bucket_size = bucket_size  # Allocated bucket size (>= size)
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
        """Auto-release on garbage collection"""
        if self.in_use:
            self.release()


class BufferPool:
    """
    GPU Buffer Pool with size-based bucketing.

    Buffers are organized into buckets by size (powers of 2).
    When a buffer is requested, the smallest bucket >= requested size is used.

    Example:
        >>> pool = BufferPool(core)
        >>> buf = pool.acquire(1024)  # Gets from 1024-byte bucket
        >>> # ... use buffer ...
        >>> buf.release()  # Returns to pool for reuse
    """

    # Bucket sizes: 256B to 256MB (powers of 2)
    MIN_BUCKET_POWER = 8   # 256 bytes
    MAX_BUCKET_POWER = 28  # 256 MB

    # Maximum buffers per bucket
    MAX_BUFFERS_PER_BUCKET = 32

    # Maximum total memory in pool (512MB default)
    MAX_POOL_MEMORY = 512 * 1024 * 1024

    def __init__(self, core: 'VulkanCore', max_memory: int = None):
        """
        Initialize buffer pool.

        Args:
            core: VulkanCore instance for buffer creation
            max_memory: Maximum total memory to keep in pool (bytes)
        """
        self.core = core
        self.max_memory = max_memory or self.MAX_POOL_MEMORY

        # Bucket pools: bucket_size -> list of available PooledBuffers
        self._buckets: Dict[int, List[PooledBuffer]] = defaultdict(list)

        # Track total pooled memory
        self._total_pooled_memory = 0

        # Lock for thread safety
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'allocations': 0,
            'evictions': 0,
            'total_acquired': 0,
            'total_released': 0,
        }

    def _size_to_bucket(self, size: int) -> int:
        """
        Round size up to nearest power of 2 bucket.

        Args:
            size: Requested buffer size in bytes

        Returns:
            Bucket size (power of 2)
        """
        if size <= 0:
            return 1 << self.MIN_BUCKET_POWER

        # Find next power of 2
        power = max(self.MIN_BUCKET_POWER, (size - 1).bit_length())
        power = min(power, self.MAX_BUCKET_POWER)
        return 1 << power

    def acquire(self, size: int, usage: int = None) -> PooledBuffer:
        """
        Acquire a buffer from the pool.

        If a suitable buffer exists in the pool, it's reused.
        Otherwise, a new buffer is created.

        Args:
            size: Required buffer size in bytes
            usage: Vulkan buffer usage flags (default: storage buffer)

        Returns:
            PooledBuffer ready for use
        """
        if usage is None:
            usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

        bucket_size = self._size_to_bucket(size)

        with self._lock:
            self._stats['total_acquired'] += 1

            # Try to find existing buffer in bucket
            bucket = self._buckets[bucket_size]

            # Look for buffer with matching usage flags
            for i, buf in enumerate(bucket):
                if not buf.in_use and buf.usage_flags == usage:
                    # Reuse this buffer
                    buf.in_use = True
                    buf.size = size
                    buf.last_used = time.time()
                    bucket.pop(i)
                    self._total_pooled_memory -= bucket_size
                    self._stats['hits'] += 1
                    return buf

            # No suitable buffer found, create new one
            self._stats['misses'] += 1
            self._stats['allocations'] += 1

            # Evict old buffers if needed to stay under memory limit
            self._evict_if_needed(bucket_size)

            # Create new buffer
            handle, memory = self.core._create_buffer(bucket_size, usage)

            return PooledBuffer(
                handle=handle,
                memory=memory,
                size=size,
                bucket_size=bucket_size,
                pool=self,
                usage_flags=usage
            )

    def _return_buffer(self, buffer: PooledBuffer):
        """
        Return a buffer to the pool for reuse.

        Called automatically by PooledBuffer.release()
        """
        with self._lock:
            self._stats['total_released'] += 1

            bucket = self._buckets[buffer.bucket_size]

            # Check if bucket is full
            if len(bucket) >= self.MAX_BUFFERS_PER_BUCKET:
                # Destroy oldest buffer
                oldest = min(bucket, key=lambda b: b.last_used)
                bucket.remove(oldest)
                self._destroy_buffer(oldest)
                self._stats['evictions'] += 1

            # Check if we're over memory limit
            if self._total_pooled_memory + buffer.bucket_size > self.max_memory:
                self._evict_lru(buffer.bucket_size)

            # Add to pool
            bucket.append(buffer)
            self._total_pooled_memory += buffer.bucket_size

    def _evict_if_needed(self, needed_size: int):
        """Evict buffers if adding needed_size would exceed limit"""
        while self._total_pooled_memory + needed_size > self.max_memory:
            if not self._evict_lru(needed_size):
                break  # No more buffers to evict

    def _evict_lru(self, min_size: int) -> bool:
        """
        Evict least recently used buffer(s) to free at least min_size bytes.

        Returns:
            True if a buffer was evicted, False if pool is empty
        """
        # Find oldest buffer across all buckets
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

        # Remove and destroy
        self._buckets[oldest_bucket_size].remove(oldest_buf)
        self._total_pooled_memory -= oldest_bucket_size
        self._destroy_buffer(oldest_buf)
        self._stats['evictions'] += 1

        return True

    def _destroy_buffer(self, buffer: PooledBuffer):
        """Destroy a Vulkan buffer"""
        try:
            from vulkan import vkDestroyBuffer, vkFreeMemory
            if buffer.handle:
                vkDestroyBuffer(self.core.device, buffer.handle, None)
            if buffer.memory:
                vkFreeMemory(self.core.device, buffer.memory, None)
        except Exception:
            pass  # Ignore cleanup errors

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
            return stats

    def __repr__(self):
        stats = self.get_stats()
        return (
            f"BufferPool(pooled={stats['total_pooled_memory']//1024}KB, "
            f"hit_rate={stats['hit_rate']:.1%}, "
            f"allocs={stats['allocations']})"
        )

    def __del__(self):
        """Cleanup on destruction"""
        self.clear()


# Global pool instance (lazy initialization)
_global_pool: Optional[BufferPool] = None
_pool_lock = threading.Lock()


def get_buffer_pool(core: 'VulkanCore' = None) -> BufferPool:
    """
    Get or create the global buffer pool.

    Args:
        core: VulkanCore instance (required on first call)

    Returns:
        BufferPool instance
    """
    global _global_pool

    with _pool_lock:
        if _global_pool is None:
            if core is None:
                raise ValueError("VulkanCore required for first buffer pool initialization")
            _global_pool = BufferPool(core)
        return _global_pool


def acquire_buffer(size: int, usage: int = None, core: 'VulkanCore' = None) -> PooledBuffer:
    """
    Convenience function to acquire a buffer from the global pool.

    Args:
        size: Required buffer size
        usage: Vulkan usage flags
        core: VulkanCore instance (for lazy pool initialization)

    Returns:
        PooledBuffer
    """
    pool = get_buffer_pool(core)
    return pool.acquire(size, usage)


def release_buffer(buffer: PooledBuffer):
    """
    Convenience function to release a buffer back to the pool.

    Args:
        buffer: PooledBuffer to release
    """
    buffer.release()
