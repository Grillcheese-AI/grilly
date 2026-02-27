"""
Milestone 2: VRAM Hamming Search — THE FUNDING GATE
====================================================
Proves: <2ms lookup at 490K entries on AMD RX 6750 XT
with cache fully resident in GDDR6 VRAM (432 GB/s).

Previous runs were reading 627 MB over PCIe (~16 GB/s).
This run should show a dramatic latency collapse.
"""
import numpy as np
import time
import grilly_core

# ── Init ────────────────────────────────────────────────────────────
d = grilly_core.Device()
d.load_shaders('shaders')
print()

dim = 10240

# ── Step 1: Correctness at small scale ──────────────────────────────
print("=" * 60)
print("STEP 1: Correctness verification (GPU vs CPU)")
print("=" * 60)

for n in [100, 1000]:
    cache = np.random.choice([-1, 1], (n, dim)).astype(np.int8)
    query = np.random.choice([-1, 1], dim).astype(np.int8)

    bench = grilly_core.HammingSearchBench(d, cache, dim)
    gpu_dists = bench.search(query)
    cpu_dists = grilly_core.hamming_search_cpu(query, cache)

    match = np.array_equal(gpu_dists, cpu_dists)
    max_diff = np.max(np.abs(gpu_dists.astype(np.int64) - cpu_dists.astype(np.int64)))
    print(f"  N={n:>6,}  GPU==CPU: {match}  max_diff={max_diff}")
    assert match, f"GPU/CPU MISMATCH at N={n}!"

print("  Correctness: ALL PASS\n")

# ── Step 2: Progressive scale benchmark ─────────────────────────────
print("=" * 60)
print("STEP 2: Latency benchmark (VRAM-resident cache)")
print("=" * 60)

test_sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 250_000, 490_000]

for n in test_sizes:
    try:
        cache = np.random.choice([-1, 1], (n, dim)).astype(np.int8)
        query = np.random.choice([-1, 1], dim).astype(np.int8)

        cache_mb = n * dim / (1024 * 1024)
        packed_mb = n * (dim // 32) * 4 / (1024 * 1024)
        print(f"\n  N={n:>7,}  bipolar={cache_mb:.0f} MB  packed={packed_mb:.1f} MB")

        # Constructor does: acquireDeviceLocal + uploadStaged
        # VRAM diagnostic prints here
        t_init = time.perf_counter()
        bench = grilly_core.HammingSearchBench(d, cache, dim)
        init_ms = (time.perf_counter() - t_init) * 1000
        print(f"    Init (bitpack+upload): {init_ms:.0f} ms")

        # Warm up GPU pipeline
        for _ in range(5):
            bench.search(query)

        # Benchmark: 100 iterations
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            result = bench.search(query)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

        avg = np.mean(times)
        p50 = np.percentile(times, 50)
        p99 = np.percentile(times, 99)
        min_t = np.min(times)

        # Effective bandwidth: GPU reads N * wordsPerVec * 4 bytes of cache
        eff_bw_gbps = (packed_mb / 1024) / (avg / 1000)  # GB/s

        status = "PASS" if avg < 2.0 else ("CLOSE" if avg < 5.0 else "SLOW")
        print(f"    avg={avg:.3f}ms  p50={p50:.3f}ms  p99={p99:.3f}ms  min={min_t:.3f}ms")
        print(f"    bandwidth={eff_bw_gbps:.1f} GB/s  [{status}]")

        # Quick correctness spot-check at scale
        cpu_dists = grilly_core.hamming_search_cpu(query, cache)
        match = np.array_equal(result, cpu_dists)
        if not match:
            max_diff = np.max(np.abs(result.astype(np.int64) - cpu_dists.astype(np.int64)))
            print(f"    [WARN] GPU != CPU, max_diff={max_diff}")
        else:
            print(f"    correctness: EXACT MATCH")

        del bench

    except Exception as e:
        print(f"    [ERROR] {e}")
        break

print("\n" + "=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)
