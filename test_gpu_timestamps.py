"""GPU timestamp diagnostic: wall-clock vs actual GPU execution time."""
import numpy as np
import time
import grilly_core

d = grilly_core.Device()
d.load_shaders('shaders')

dim = 10240

for n in [1000, 10_000, 50_000, 100_000, 490_000]:
    cache = (np.random.randint(0, 2, size=(n, dim), dtype=np.int8) * 2 - 1)
    query = (np.random.randint(0, 2, size=dim, dtype=np.int8) * 2 - 1)
    bench = grilly_core.HammingSearchBench(d, cache, dim)
    del cache

    # Warmup
    for _ in range(5):
        bench.search(query)

    # Measure
    wall_times = []
    gpu_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        bench.search(query)
        wall_ms = (time.perf_counter() - t0) * 1000
        wall_times.append(wall_ms)
        gpu_times.append(bench.gpu_time_ms)

    w = np.mean(wall_times)
    g = np.mean(gpu_times)
    overhead = w - g
    data_mb = n * 320 * 4 / 1e6
    gpu_bw = data_mb / (g / 1000) / 1000 if g > 0 else 0

    print(f"N={n:>7,}  wall={w:.3f}ms  gpu={g:.3f}ms  "
          f"overhead={overhead:.3f}ms  gpu_bw={gpu_bw:.1f} GB/s")

    del bench
