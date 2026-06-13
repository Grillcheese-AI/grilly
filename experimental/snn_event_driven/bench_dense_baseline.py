"""Dense GPU baseline for the event-driven SNN A/B (the kill-criterion denominator).

Times the EXISTING registered op path -- grilly.backend._bridge.linear -- which
is exactly how brain/synapsis.py propagates spikes today (dense W @ spikes).
No rebuild needed: this measures the dense baseline the event-driven scatter
op must beat. The scatter side needs a registered _core.spike_scatter op
(mirrors gif_neuron_step) -> separate rebuild step.
"""
import sys, time
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np
from grilly.backend import _bridge as b

print("bridge available:", b.is_available())
dev_ok = b.is_available()

def bench_linear(N, rows, iters=50, warmup=10):
    rng = np.random.default_rng(0)
    spikes = (rng.random((rows, N)) < 0.05).astype(np.float32)   # 5% active
    W = rng.standard_normal((N, N)).astype(np.float32)           # (out, in)
    # correctness/availability probe
    out = b.linear(spikes, W, None)
    if out is None:
        return None
    for _ in range(warmup):
        b.linear(spikes, W, None)
    t0 = time.perf_counter()
    for _ in range(iters):
        b.linear(spikes, W, None)
    dt = (time.perf_counter() - t0) / iters
    return dt * 1e3   # ms/call

print(f"\n{'N':>6} {'rows':>5} {'dense_linear_ms':>16} {'GFLOP/s':>10}")
for N in (512, 1024, 2048, 4096):
    for rows in (1, 64):
        ms = bench_linear(N, rows)
        if ms is None:
            print(f"{N:>6} {rows:>5}   linear() returned None (CPU fallback / no spv)")
            continue
        flops = 2.0 * rows * N * N
        gflops = flops / (ms * 1e-3) / 1e9
        print(f"{N:>6} {rows:>5} {ms:>16.4f} {gflops:>10.1f}")

# dense GIF neuron op probe (the other registered SNN op)
print("\n[gif_neuron_step probe]")
N = 4096
z = np.zeros(N, dtype=np.float32)
res = b.gif_neuron_step(np.ones(N, np.float32), z.copy(), z.copy(), z.copy(),
                        z.copy(), z.copy(), z.copy())
print("gif_neuron_step available:", res is not None,
      "| keys:", list(res.keys()) if isinstance(res, dict) else type(res).__name__)
