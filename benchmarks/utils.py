"""
Benchmark utilities for timing, VRAM monitoring, and result formatting.
"""

import time

import numpy as np


def time_fn(fn, *args, warmup=2, repeats=5, **kwargs):
    """Time a function with warmup and averaging.

    Returns:
        dict with 'mean', 'std', 'min', 'max' times in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "result": result,
    }


def time_cpu(fn, *args, warmup=2, repeats=5, **kwargs):
    """Time a CPU-only function."""
    return time_fn(fn, *args, warmup=warmup, repeats=repeats, **kwargs)


def compute_speedup(gpu_ms, cpu_ms):
    """Compute speedup ratio. Returns 'N/A' if either is zero."""
    if cpu_ms <= 0 or gpu_ms <= 0:
        return "N/A"
    return cpu_ms / gpu_ms


def format_time(ms):
    """Format milliseconds to human-readable string."""
    if ms < 1:
        return f"{ms * 1000:.1f} us"
    if ms < 1000:
        return f"{ms:.2f} ms"
    return f"{ms / 1000:.2f} s"


def format_size(nbytes):
    """Format byte count to human-readable string."""
    if nbytes < 1024:
        return f"{nbytes} B"
    if nbytes < 1024**2:
        return f"{nbytes / 1024:.1f} KB"
    if nbytes < 1024**3:
        return f"{nbytes / 1024**2:.1f} MB"
    return f"{nbytes / 1024**3:.2f} GB"


def print_header(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_row(label, gpu_ms, cpu_ms=None, extra=""):
    """Print a formatted benchmark row."""
    speedup = ""
    if cpu_ms is not None and cpu_ms > 0 and gpu_ms > 0:
        s = cpu_ms / gpu_ms
        speedup = f"  {s:.1f}x"
    cpu_str = format_time(cpu_ms) if cpu_ms is not None else "N/A"
    print(f"  {label:<30} GPU: {format_time(gpu_ms):>10}  CPU: {cpu_str:>10}{speedup}  {extra}")


def print_summary_table(results):
    """Print a summary table of benchmark results.

    Args:
        results: list of dicts with keys 'label', 'gpu_ms', 'cpu_ms' (optional),
                 'shape' (optional), 'extra' (optional)
    """
    print(f"\n{'Label':<30} {'GPU':>10} {'CPU':>10} {'Speedup':>8} {'Shape'}")
    print("-" * 75)
    for r in results:
        gpu_str = format_time(r["gpu_ms"])
        cpu_str = format_time(r.get("cpu_ms", 0)) if r.get("cpu_ms") else "N/A"
        speedup = ""
        if r.get("cpu_ms") and r["gpu_ms"] > 0:
            speedup = f"{r['cpu_ms'] / r['gpu_ms']:.1f}x"
        shape = r.get("shape", "")
        print(f"  {r['label']:<28} {gpu_str:>10} {cpu_str:>10} {speedup:>8} {shape}")


def get_gpu_backend():
    """Get Grilly GPU backend, or None if unavailable."""
    try:
        from grilly import Compute

        backend = Compute()
        return backend
    except Exception as e:
        print(f"GPU backend unavailable: {e}")
        return None


def check_gpu_available():
    """Check if GPU backend is available and print info."""
    backend = get_gpu_backend()
    if backend is None:
        print("WARNING: GPU backend not available. Only CPU benchmarks will run.")
        return False
    print("GPU backend initialized successfully")
    if hasattr(backend, "core") and hasattr(backend.core, "device_name"):
        print(f"  Device: {backend.core.device_name}")
    return True


def get_buffer_pool_stats(backend):
    """Get buffer pool statistics if available."""
    if backend is None:
        return None
    try:
        if hasattr(backend, "fnn") and hasattr(backend.fnn, "buffer_pool"):
            pool = backend.fnn.buffer_pool
            if pool is not None:
                return pool.get_stats()
    except Exception:
        pass
    return None
