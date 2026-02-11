"""
Benchmark: Buffer pool hit rate, allocation speed, and VRAM usage.
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from benchmarks.utils import (
    format_size,
    format_time,
    get_gpu_backend,
    print_header,
)


def bench_pool_hit_rate(backend, num_ops=100):
    """Simulate a realistic workload and measure pool hit rate."""
    if not hasattr(backend, "fnn"):
        return None

    fnn = backend.fnn

    # Simulate varying buffer sizes like a real model would use
    sizes = [
        128 * 4,  # small activation
        512 * 4,  # medium
        1024 * 4,  # large
        2048 * 4,  # XL
        4096 * 4,  # common embedding dim
        512 * 512 * 4,  # weight matrix
    ]

    # Run operations that exercise the pool
    for i in range(num_ops):
        size = sizes[i % len(sizes)]
        dim = int(np.sqrt(size // 4))
        if dim < 2:
            dim = 2
        x = np.random.randn(dim, dim).astype(np.float32)
        try:
            fnn.activation_relu(x)
        except Exception:
            pass

    pool = fnn.buffer_pool
    if pool is not None:
        return pool.get_stats()
    return None


def bench_allocation_speed(backend, num_allocs=50):
    """Benchmark buffer allocation speed: pool vs direct."""
    if not hasattr(backend, "fnn"):
        return None, None

    fnn = backend.fnn
    size = 4096 * 4  # 16KB buffer

    # Pooled allocation (warm pool)
    pool = fnn.buffer_pool
    if pool is None:
        return None, None

    # Warm the pool
    bufs = []
    for _ in range(5):
        buf = fnn._acquire_buffer(size)
        bufs.append(buf)
    for buf in bufs:
        fnn._release_buffer(buf)

    # Time pooled acquire/release
    t0 = time.perf_counter()
    for _ in range(num_allocs):
        buf = fnn._acquire_buffer(size)
        fnn._release_buffer(buf)
    pool_time = (time.perf_counter() - t0) * 1000  # ms

    # Time direct allocation (bypass pool)
    from grilly.backend.base import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

    try:
        from vulkan import vkDestroyBuffer, vkFreeMemory

        t0 = time.perf_counter()
        for _ in range(num_allocs):
            handle, memory = backend.core._create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            vkDestroyBuffer(backend.core.device, handle, None)
            vkFreeMemory(backend.core.device, memory, None)
        direct_time = (time.perf_counter() - t0) * 1000
    except Exception:
        direct_time = None

    return pool_time, direct_time


def main():
    print_header("Buffer Pool Benchmark")

    backend = get_gpu_backend()
    if backend is None:
        print("GPU backend unavailable, skipping.")
        return

    # Pool hit rate
    print_header("Pool Hit Rate (100 ops)")
    stats = bench_pool_hit_rate(backend, num_ops=100)
    if stats:
        print(f"  Hits:        {stats.get('hits', 0)}")
        print(f"  Misses:      {stats.get('misses', 0)}")
        print(f"  Hit Rate:    {stats.get('hit_rate', 0):.1%}")
        print(f"  Allocations: {stats.get('allocations', 0)}")
        print(f"  Evictions:   {stats.get('evictions', 0)}")
        print(f"  Pooled Mem:  {format_size(stats.get('total_pooled_memory', 0))}")
        print(f"  VMA Enabled: {stats.get('vma_enabled', False)}")
        if stats.get("buckets"):
            print("  Buckets:")
            for bsize, count in sorted(stats["buckets"].items()):
                print(f"    {format_size(bsize)}: {count} buffers")
    else:
        print("  Buffer pool not available")

    # Allocation speed
    print_header("Allocation Speed (50 allocs of 16KB)")
    pool_time, direct_time = bench_allocation_speed(backend, num_allocs=50)
    if pool_time is not None:
        print(f"  Pool:   {format_time(pool_time)} total, {format_time(pool_time / 50)} /alloc")
        if direct_time is not None:
            print(
                f"  Direct: {format_time(direct_time)} total, {format_time(direct_time / 50)} /alloc"
            )
            speedup = direct_time / pool_time if pool_time > 0 else 0
            print(f"  Pool speedup: {speedup:.1f}x")
    else:
        print("  Buffer pool not available")


if __name__ == "__main__":
    main()
