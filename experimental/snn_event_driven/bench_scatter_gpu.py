"""On-GPU A/B: event-driven spike_scatter vs dense _bridge.linear (P3 kill criterion).

Both ops re-upload W (N*N) + download(N) per call, so that cost is shared and the
timing difference isolates the kernel (scatter vs GEMM). Correctness is checked
against numpy first.
"""
import sys, time
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np
import grilly_core as g
from grilly.backend import _bridge as b

dev = b._get_device()   # bridge device: shaders from shaders/spv/ already loaded
print("device:", dev.device_name)
print("has spike_scatter:", hasattr(g, "spike_scatter"))

def make(N, p, seed=0):
    rng = np.random.default_rng(seed)
    spikes = (rng.random(N) < p).astype(np.float32)
    W = rng.standard_normal((N, N)).astype(np.float32)        # [pre, post]
    fired = np.flatnonzero(spikes).astype(np.uint32)
    return spikes, W, fired

def scatter(fired, W, N):
    fi = np.ascontiguousarray(fired).view(np.float32)             # uint32 bytes as f32
    fc = np.array([fired.size], np.uint32).view(np.float32)
    out = g.spike_scatter(dev, fi, fc, np.ascontiguousarray(W), int(N))
    return np.asarray(out, dtype=np.float32)

# ── correctness ──
print("\n[correctness] spike_scatter vs numpy (spikes @ W)")
for N in (1024, 4096):
    spikes, W, fired = make(N, 0.05, seed=1)
    got = scatter(fired, W, N)
    ref = spikes @ W
    print(f"  N={N:5d}  fired={fired.size:4d}  max_abs_diff={np.max(np.abs(got-ref)):.3e}")

def bench(fn, iters=30, warm=5):
    for _ in range(warm): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t0) / iters * 1e3   # ms

# ── timing A/B ──
print(f"\n[A/B timing]  {'N':>5} {'act':>6} {'fired':>6} "
      f"{'scatter_ms':>11} {'dense_ms':>10} {'speedup':>8}")
for N in (1024, 4096):
    spikes_row = (np.random.default_rng(7).random((1, N)) < 0.05).astype(np.float32)
    _, W, _ = make(N, 0.05, seed=2)
    Wlin = np.ascontiguousarray(W)
    dense_ms = bench(lambda: b.linear(spikes_row, Wlin, None))
    for p in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        _, _, fired = make(N, p, seed=3)
        sc_ms = bench(lambda: scatter(fired, W, N))
        spd = dense_ms / sc_ms if sc_ms > 0 else float('nan')
        print(f"       {N:>5} {p:>6.3f} {fired.size:>6} "
              f"{sc_ms:>11.4f} {dense_ms:>10.4f} {spd:>7.2f}x")
