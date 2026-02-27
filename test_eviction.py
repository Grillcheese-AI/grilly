"""Test if WDDM evicts VRAM after a delay."""
import numpy as np
import time
import grilly_core

d = grilly_core.Device()
d.load_shaders('shaders')

dim = 10240
n = 490_000

cache = (np.random.randint(0, 2, size=(n, dim), dtype=np.int8) * 2 - 1)
query = (np.random.randint(0, 2, size=dim, dtype=np.int8) * 2 - 1)

bench = grilly_core.HammingSearchBench(d, cache, dim)
del cache

# Immediately after upload -- search 5 times
print("== IMMEDIATELY after upload ==")
for i in range(5):
    t0 = time.perf_counter()
    bench.search(query)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  search {i}: {ms:.3f} ms")

# Wait 2 seconds
print("\n== After 2s sleep ==")
time.sleep(2)
for i in range(5):
    t0 = time.perf_counter()
    bench.search(query)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  search {i}: {ms:.3f} ms")

# Wait 5 seconds
print("\n== After 5s sleep ==")
time.sleep(5)
for i in range(5):
    t0 = time.perf_counter()
    bench.search(query)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  search {i}: {ms:.3f} ms")

# Wait 10 seconds
print("\n== After 10s sleep ==")
time.sleep(10)
for i in range(5):
    t0 = time.perf_counter()
    bench.search(query)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  search {i}: {ms:.3f} ms")
