"""Test hamming-top1 shader: atomicMin argmin on GPU.
Compares GPU top-1 result against CPU reference, then benchmarks.
"""
import numpy as np
import time
import grilly_core

d = grilly_core.Device()
d.load_shaders('shaders/spv')

dim = 10240

print("=" * 60)
print("  hamming-top1: GPU atomicMin argmin benchmark")
print("=" * 60)

# ── Test 1: Correctness ──────────────────────────────────────
print("\n[1] Correctness: GPU top-1 vs CPU argmin")
for n in [100, 1000, 10_000]:
    cache = (np.random.randint(0, 2, size=(n, dim), dtype=np.int8) * 2 - 1)
    query = (np.random.randint(0, 2, size=dim, dtype=np.int8) * 2 - 1)
    bench = grilly_core.HammingSearchBench(d, cache, dim)
    del cache

    # CPU reference: full distance array → argmin
    cpu_dists = bench.search(query)
    cpu_best_idx = int(np.argmin(cpu_dists))
    cpu_best_dist = int(cpu_dists[cpu_best_idx])

    # GPU top-1
    gpu_result = bench.search_top1(query)
    gpu_idx = gpu_result['index']
    gpu_dist = gpu_result['distance']

    match = (gpu_idx == cpu_best_idx and gpu_dist == cpu_best_dist)
    print(f"  N={n:>6,}  CPU: idx={cpu_best_idx} dist={cpu_best_dist}"
          f"  GPU: idx={gpu_idx} dist={gpu_dist}"
          f"  {'OK' if match else 'MISMATCH!'}")
    assert match, f"GPU/CPU mismatch at N={n}!"
    del bench

print("  All correctness tests passed!")

# ── Test 2: Latency benchmark ────────────────────────────────
print("\n[2] Latency: wall-clock + GPU timestamps")
for n in [1000, 10_000, 50_000, 100_000, 490_000]:
    cache = (np.random.randint(0, 2, size=(n, dim), dtype=np.int8) * 2 - 1)
    query = (np.random.randint(0, 2, size=dim, dtype=np.int8) * 2 - 1)
    bench = grilly_core.HammingSearchBench(d, cache, dim)
    del cache

    # Warmup
    for _ in range(5):
        bench.search_top1(query)

    # Measure
    wall_times = []
    gpu_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        result = bench.search_top1(query)
        wall_ms = (time.perf_counter() - t0) * 1000
        wall_times.append(wall_ms)
        gpu_times.append(bench.gpu_time_ms)

    w = np.mean(wall_times)
    g = np.mean(gpu_times)
    p99 = np.percentile(wall_times, 99)
    data_mb = n * (dim // 32) * 4 / 1e6  # cache data read by GPU

    status = "PASS <2ms" if w < 2.0 else "FAIL"
    print(f"  N={n:>7,}  wall={w:.3f}ms  gpu={g:.3f}ms  "
          f"p99={p99:.3f}ms  data={data_mb:.0f}MB  [{status}]")

    del bench

print("\nDone.")
