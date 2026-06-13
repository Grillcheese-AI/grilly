import sys, time
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np, grilly_core as g
from grilly.backend import _bridge as b
dev = b._get_device()

M, NI, NO, p = 512, 2048, 2048, 0.02
rng = np.random.default_rng(0)
W = np.ascontiguousarray(rng.standard_normal((NI, NO)).astype(np.float32))
K = int(NI * p)

def build(shared):
    if shared:
        f = np.sort(rng.choice(NI, K, replace=False)).astype(np.uint32)
        idx = np.tile(f, M)
        cnt = np.full(M, f.size, np.uint32)
    else:
        parts = []
        cnt = np.empty(M, np.uint32)
        for m in range(M):
            f = np.sort(rng.choice(NI, K, replace=False)).astype(np.uint32)
            parts.append(f); cnt[m] = f.size
        idx = np.concatenate(parts)
    off = np.zeros(M, np.uint32); np.cumsum(cnt[:-1], out=off[1:])
    vals = np.ones(idx.size, np.float32)
    return idx.astype(np.uint32), off, cnt, vals

def tt(args, n=30, w=5):
    idx, off, cnt, vals = args
    fn = lambda: g.spike_propagate_batch(dev, idx.view(np.float32), off.view(np.float32),
                                         cnt.view(np.float32), W, vals, NI, NO, M)
    for _ in range(w): fn()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t0) / n * 1e3

print("K fired/row =", K)
print("shared_fired_ms = %.3f" % tt(build(True)))
print("indep_fired_ms  = %.3f" % tt(build(False)))
