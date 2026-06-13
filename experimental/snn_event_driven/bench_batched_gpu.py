"""Batched-submit A/B: does recording all timesteps into ONE GPU submit remove
the per-step host-sync floor and expose the kernel ceiling?

resident_bench(..., batched): 0 = one submit + host wait per step (floor),
1 = one begin / N dispatch+barrier / one wait. W resident either way.
Per-iter uses few iters; batched uses many so the one-time W upload amortizes out.
"""
import sys, time
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np
import grilly_core as g
from grilly.backend import _bridge as b

dev = b._get_device()
print("device:", dev.device_name)
PERITER, BATCHED = 50, 1000

def make(N, p, seed=0):
    rng = np.random.default_rng(seed)
    spikes = (rng.random(N) < p).astype(np.float32)
    W = np.ascontiguousarray(rng.standard_normal((N, N)).astype(np.float32))
    fired = np.flatnonzero(spikes).astype(np.uint32)
    return spikes, W, fired

def run(mode, fired, spikes, W, N, iters, batched):
    fi = np.ascontiguousarray(fired).view(np.float32)
    fc = np.array([fired.size], np.uint32).view(np.float32)
    return np.asarray(g.resident_bench(dev, mode, fi, fc, spikes, W,
                                       int(N), int(iters), int(batched)),
                      dtype=np.float32)

def timed(mode, fired, spikes, W, N, iters, batched):
    run(mode, fired, spikes, W, N, 2, batched)         # warmup
    t0 = time.perf_counter()
    run(mode, fired, spikes, W, N, iters, batched)
    return (time.perf_counter() - t0) / iters * 1e3    # ms/step

print(f"\n{'N':>5} {'act':>6} {'fired':>6} | {'sc_periter':>10} {'sc_batched':>10} "
      f"{'speedup_b':>9} | {'dn_batched':>10} {'scatter/dense_b':>15}")
for N in (1024, 4096):
    for p in (0.005, 0.01, 0.05, 0.10):
        spikes, W, fired = make(N, p, seed=3)
        sc_pi = timed(0, fired, spikes, W, N, PERITER, 0)
        sc_b  = timed(0, fired, spikes, W, N, BATCHED, 1)
        dn_b  = timed(1, fired, spikes, W, N, BATCHED, 1)
        print(f"{N:>5} {p:>6.3f} {fired.size:>6} | {sc_pi:>10.4f} {sc_b:>10.4f} "
              f"{sc_pi/sc_b:>8.1f}x | {dn_b:>10.4f} {dn_b/sc_b:>14.1f}x")
