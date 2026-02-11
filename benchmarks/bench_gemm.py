"""
GEMM (General Matrix Multiply) benchmark for Grilly.

Compares GPU-accelerated GEMM via Vulkan compute shaders against
CPU-based numpy matrix multiplication across a range of matrix sizes.

Usage:
    python benchmarks/bench_gemm.py
"""

import os
import sys

import numpy as np

# Ensure the repo root is on the path so grilly and benchmarks are importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.utils import (
    format_size,
    get_gpu_backend,
    print_header,
    print_row,
    print_summary_table,
    time_cpu,
    time_fn,
)

# Square matrix sizes to benchmark (N x N).
SIZES = [16, 64, 128, 256, 512, 1024, 2048]


def generate_matrices(n, dtype=np.float32):
    """Generate two random float32 matrices of shape (n, n)."""
    rng = np.random.default_rng(seed=42)
    A = rng.standard_normal((n, n)).astype(dtype)
    B = rng.standard_normal((n, n)).astype(dtype)
    return A, B


def verify_correctness(backend, sizes=None):
    """Verify that GPU GEMM results match CPU (numpy) results for each size.

    Returns:
        list of (n, passed, max_diff) tuples.
    """
    if sizes is None:
        sizes = SIZES

    print_header("GEMM Correctness Verification")
    results = []

    for n in sizes:
        A, B = generate_matrices(n)
        C_cpu = A @ B
        C_gpu = backend.fnn.gemm(A, B)

        # Use a tolerance that accounts for float32 accumulation differences.
        # Larger matrices accumulate more floating-point error.
        atol = 1e-2 * max(1, n / 64)
        rtol = 1e-3

        passed = np.allclose(C_gpu, C_cpu, atol=atol, rtol=rtol)
        max_diff = float(np.max(np.abs(C_gpu - C_cpu)))

        status = "PASS" if passed else "FAIL"
        print(f"  {n:>5}x{n:<5}  {status}  max_diff={max_diff:.6e}")
        results.append((n, passed, max_diff))

    return results


def benchmark_gemm(backend, sizes=None, warmup=2, repeats=5):
    """Run GPU vs CPU GEMM benchmarks across all sizes.

    Returns:
        list of result dicts suitable for print_summary_table.
    """
    if sizes is None:
        sizes = SIZES

    print_header("GEMM Performance: GPU vs CPU")
    summary = []

    for n in sizes:
        A, B = generate_matrices(n)
        data_bytes = 2 * A.nbytes  # two input matrices
        label = f"{n}x{n} GEMM"

        # GPU timing
        gpu_stats = time_fn(backend.fnn.gemm, A, B, warmup=warmup, repeats=repeats)
        gpu_ms = gpu_stats["mean"]

        # CPU timing (numpy, often BLAS-accelerated)
        cpu_stats = time_cpu(lambda: A @ B, warmup=warmup, repeats=repeats)
        cpu_ms = cpu_stats["mean"]

        # FLOP count for square GEMM: 2*N^3 (multiply + add per element)
        flops = 2.0 * n * n * n
        gpu_gflops = (flops / (gpu_ms / 1000.0)) / 1e9 if gpu_ms > 0 else 0
        cpu_gflops = (flops / (cpu_ms / 1000.0)) / 1e9 if cpu_ms > 0 else 0

        extra = f"GPU: {gpu_gflops:.1f} GFLOP/s  CPU: {cpu_gflops:.1f} GFLOP/s"
        print_row(label, gpu_ms, cpu_ms, extra=extra)

        summary.append(
            {
                "label": label,
                "gpu_ms": gpu_ms,
                "cpu_ms": cpu_ms,
                "shape": f"({n},{n})x({n},{n}) [{format_size(data_bytes)}]",
                "gpu_gflops": gpu_gflops,
                "cpu_gflops": cpu_gflops,
            }
        )

    return summary


def main():
    backend = get_gpu_backend()
    if backend is None:
        print("ERROR: GPU backend is required for this benchmark. Exiting.")
        sys.exit(1)

    # -- Correctness --
    correctness = verify_correctness(backend, sizes=SIZES)
    all_passed = all(passed for _, passed, _ in correctness)
    if not all_passed:
        print("\nWARNING: Some correctness checks failed! Results may be unreliable.")
    else:
        print("\nAll correctness checks passed.")

    # -- Performance --
    results = benchmark_gemm(backend, sizes=SIZES)

    # -- Summary table --
    print_header("Summary")
    print_summary_table(results)

    # -- Peak throughput --
    if results:
        best = max(results, key=lambda r: r.get("gpu_gflops", 0))
        print(f"\n  Peak GPU throughput: {best['gpu_gflops']:.1f} GFLOP/s at {best['label']}")
        best_cpu = max(results, key=lambda r: r.get("cpu_gflops", 0))
        print(f"  Peak CPU throughput: {best_cpu['cpu_gflops']:.1f} GFLOP/s at {best_cpu['label']}")

    print()


if __name__ == "__main__":
    main()
