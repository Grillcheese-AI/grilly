"""490K VRAM benchmark — lean version. Frees numpy cache immediately."""
import numpy as np
import time, gc
import grilly_core

d = grilly_core.Device()
d.load_shaders('shaders')

dim = 10240
n = 490_000

# Quick correctness check
print("\n-- Correctness (N=100) --")
c = (np.random.randint(0, 2, size=(100, dim), dtype=np.int8) * 2 - 1)
q = (np.random.randint(0, 2, size=dim, dtype=np.int8) * 2 - 1)
b = grilly_core.HammingSearchBench(d, c, dim)
assert np.array_equal(b.search(q), grilly_core.hamming_search_cpu(q, c))
print("  GPU == CPU: EXACT MATCH")
del b, c; gc.collect()

# 490K — THE FUNDING GATE
print(f"\n-- N={n:,} (packed={n*320*4/(1024**2):.0f} MB in VRAM) --")
print("  Generating cache...", end=" ", flush=True)
cache = (np.random.randint(0, 2, size=(n, dim), dtype=np.int8) * 2 - 1)
print(f"{cache.nbytes/(1024**3):.1f} GB numpy")

print("  Bitpacking + uploading to VRAM...", end=" ", flush=True)
t0 = time.perf_counter()
bench = grilly_core.HammingSearchBench(d, cache, dim)
print(f"{(time.perf_counter()-t0)*1000:.0f} ms")

# Free the 4.7 GB numpy array — we don't need it anymore
del cache; gc.collect()
print("  Freed numpy cache (4.7 GB system RAM)")

query = (np.random.randint(0, 2, size=dim, dtype=np.int8) * 2 - 1)

# Warm up
for _ in range(10):
    bench.search(query)

# Benchmark
print("  Running 100 iterations...")
times = []
for _ in range(100):
    t0 = time.perf_counter()
    result = bench.search(query)
    times.append((time.perf_counter() - t0) * 1000)

avg = np.mean(times)
p50 = np.percentile(times, 50)
p99 = np.percentile(times, 99)
mn  = np.min(times)
bw  = (n * 320 * 4 / 1e9) / (avg / 1000)

print(f"\n  avg={avg:.3f}ms  p50={p50:.3f}ms  p99={p99:.3f}ms  min={mn:.3f}ms")
print(f"  bandwidth={bw:.1f} GB/s")
print(f"  {'PASS <2ms' if avg < 2 else 'CLOSE <5ms' if avg < 5 else 'SLOW'}")
