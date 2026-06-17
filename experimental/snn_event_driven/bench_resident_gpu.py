"""Resident-W A/B: does the event-driven advantage appear once W stops
re-uploading every step? resident_bench uploads W once and dispatches `iters`
times; per-step cost = total / iters. mode 0 = scatter (gather over fired),
mode 1 = dense (full GEMV). Both share the identical gather kernel structure;
only the loop length differs (fired vs N), so speedup ~ 1/activity if compute/
read bound.
"""
import sys, time
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np
import grilly_core as g
from grilly.backend import _bridge as b

dev = b._get_device()
print("device:", dev.device_name, "| has resident_bench:", hasattr(g, "resident_bench"))
ITERS = 100

def make(N, p, seed=0):
    rng = np.random.default_rng(seed)
    spikes = (rng.random(N) < p).astype(np.float32)
    W = np.ascontiguousarray(rng.standard_normal((N, N)).astype(np.float32))
    fired = np.flatnonzero(spikes).astype(np.uint32)
    return spikes, W, fired

def run(mode, fired, spikes, W, N, iters):
    fi = np.ascontiguousarray(fired).view(np.float32)
    fc = np.array([fired.size], np.uint32).view(np.float32)
    out = g.resident_bench(dev, mode, fi, fc, spikes, W, int(N), int(iters))
    return np.asarray(out, dtype=np.float32)

# ── correctness (last-iter result vs numpy) ──
print("\n[correctness] resident_bench vs numpy (spikes @ W)")
for N in (1024, 4096):
    spikes, W, fired = make(N, 0.05, seed=1)
    sc = run(0, fired, spikes, W, N, 1)
    dn = run(1, fired, spikes, W, N, 1)
    ref = spikes @ W
    print(f"  N={N:5d}  scatter_diff={np.max(np.abs(sc-ref)):.2e}  "
          f"dense_diff={np.max(np.abs(dn-ref)):.2e}")

def timed(mode, fired, spikes, W, N):
    run(mode, fired, spikes, W, N, 2)            # warmup (pipeline create)
    t0 = time.perf_counter()
    run(mode, fired, spikes, W, N, ITERS)
    return (time.perf_counter() - t0) / ITERS * 1e3   # ms/step

print(f"\n[resident A/B, W uploaded once]  {'N':>5} {'act':>6} {'fired':>6} "
      f"{'scatter_ms':>11} {'dense_ms':>10} {'speedup':>8}")
for N in (1024, 4096):
    for p in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        spikes, W, fired = make(N, p, seed=3)
        sc = timed(0, fired, spikes, W, N)
        dn = timed(1, fired, spikes, W, N)
        spd = dn / sc if sc > 0 else float('nan')
        print(f"          {N:>5} {p:>6.3f} {fired.size:>6} "
              f"{sc:>11.4f} {dn:>10.4f} {spd:>7.2f}x")
