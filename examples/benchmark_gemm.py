"""GEMM throughput benchmark: GPU (Vulkan) vs CPU (NumPy)."""

import time

import grilly
import numpy as np

backend = grilly.Compute()

sizes = [128, 256, 512, 1024, 2048]
warmup = 3
trials = 10

print(f"{'Size':>6}  {'GPU ms':>8}  {'GFLOP/s':>9}  {'CPU ms':>8}  {'Speedup':>8}")
print("-" * 52)

for n in sizes:
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)
    flops = 2.0 * n * n * n  # matmul FLOPs

    # --- GPU (Vulkan GEMM) ---
    for _ in range(warmup):
        backend.fnn.gemm(a, b)

    t0 = time.perf_counter()
    for _ in range(trials):
        backend.fnn.gemm(a, b)
    gpu_ms = (time.perf_counter() - t0) / trials * 1000
    gflops = flops / (gpu_ms / 1000) / 1e9

    # --- CPU (NumPy) ---
    for _ in range(warmup):
        a @ b

    t0 = time.perf_counter()
    for _ in range(trials):
        a @ b
    cpu_ms = (time.perf_counter() - t0) / trials * 1000

    speedup = cpu_ms / gpu_ms if gpu_ms > 0 else float("inf")
    print(f"{n:>6}  {gpu_ms:>7.2f}  {gflops:>8.1f}  {cpu_ms:>7.2f}  {speedup:>7.2f}x")
