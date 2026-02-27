"""
Timing breakdown: Where is the 52ms going?

The cache is confirmed in VRAM (DEVICE_LOCAL=1).
At 432 GB/s, reading 598 MB should take ~1.4ms.
But we're seeing 52ms. Something else dominates.

Theory: The latency scales linearly with N, suggesting
either the download (N*4 bytes) or the shader itself is the bottleneck.
Let's test by measuring bandwidth per entry.
"""
import numpy as np
import time
import grilly_core

d = grilly_core.Device()
d.load_shaders('shaders')

dim = 10240

# Test: Does latency change if we use a SMALLER dim?
# If bandwidth-bound on cache read, halving dim should halve time.
# If dispatch-overhead-bound, time should be constant.
print("\n=== Test 1: Dim scaling (same N, different vector sizes) ===")
n = 50000
for test_dim in [1024, 2048, 4096, 8192, 10240]:
    cache = np.random.choice([-1, 1], (n, test_dim)).astype(np.int8)
    query = np.random.choice([-1, 1], test_dim).astype(np.int8)
    bench = grilly_core.HammingSearchBench(d, cache, test_dim)

    # Warm up
    for _ in range(5):
        bench.search(query)

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        bench.search(query)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    avg = np.mean(times)
    words = test_dim // 32
    cache_bytes_read = n * words * 4
    bw = (cache_bytes_read / 1e9) / (avg / 1000)
    print(f"  dim={test_dim:>5}  words={words:>3}  avg={avg:.3f}ms  "
          f"cache_read={cache_bytes_read/(1024*1024):.1f}MB  bw={bw:.1f}GB/s")
    del bench

# Test 2: Does latency scale with N at fixed dim?
print("\n=== Test 2: N scaling at dim=10240 ===")
for n in [1000, 5000, 10000, 50000, 100000]:
    cache = np.random.choice([-1, 1], (n, dim)).astype(np.int8)
    query = np.random.choice([-1, 1], dim).astype(np.int8)
    bench = grilly_core.HammingSearchBench(d, cache, dim)

    for _ in range(5):
        bench.search(query)

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        bench.search(query)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    avg = np.mean(times)
    us_per_entry = avg * 1000 / n  # microseconds per entry
    print(f"  N={n:>7,}  avg={avg:.3f}ms  us/entry={us_per_entry:.4f}")
    del bench

# Test 3: What's the FIXED overhead? (dispatch + fence)
print("\n=== Test 3: Fixed overhead (N=1, trivial dispatch) ===")
for n in [1, 10, 100]:
    cache = np.random.choice([-1, 1], (n, dim)).astype(np.int8)
    query = np.random.choice([-1, 1], dim).astype(np.int8)
    bench = grilly_core.HammingSearchBench(d, cache, dim)

    for _ in range(5):
        bench.search(query)

    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        bench.search(query)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    avg = np.mean(times)
    print(f"  N={n:>5}  avg={avg:.3f}ms")
    del bench

print("\nDone.")
