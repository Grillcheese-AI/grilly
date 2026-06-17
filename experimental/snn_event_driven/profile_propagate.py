import sys, time
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np
import grilly_core as g
from grilly.backend import _bridge as b
dev = b._get_device()

M, NI, NO, p = 512, 2048, 2048, 0.02
rng = np.random.default_rng(0)
x = np.zeros((M, NI), np.float32)
mask = rng.random(x.shape) < p
x[mask] = rng.integers(1, 9, size=int(mask.sum())).astype(np.float32)
W = np.ascontiguousarray(rng.standard_normal((NI, NO)).astype(np.float32))  # (in,out)

def compact(xf):
    rows, cols = np.nonzero(xf)
    cnt = np.bincount(rows, minlength=xf.shape[0]).astype(np.uint32)
    off = np.zeros(xf.shape[0], np.uint32)
    if xf.shape[0] > 1:
        np.cumsum(cnt[:-1], out=off[1:])
    return cols.astype(np.uint32), off, cnt, xf[rows, cols].astype(np.float32)

def tt(fn, n=30, w=5):
    for _ in range(w): fn()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter()-t0)/n*1e3

idx, off, cnt, vals = compact(x)
print("total fired:", idx.size, "avg/row:", idx.size/M)

t_compact = tt(lambda: compact(x))

def op_only():
    return g.spike_propagate_batch(dev, idx.view(np.float32), off.view(np.float32),
                                   cnt.view(np.float32), W, vals, NI, NO, M)
t_op = tt(op_only)

out = op_only()
t_asarray = tt(lambda: np.asarray(out, dtype=np.float32))

t_dense = tt(lambda: b.linear(x, np.ascontiguousarray(W.T), None))

print(f"compact_ms = {t_compact:.4f}")
print(f"op_ms      = {t_op:.4f}   (spike_propagate_batch alone)")
print(f"asarray_ms = {t_asarray:.4f}   (Tensor -> numpy)")
print(f"dense_ms   = {t_dense:.4f}   (_bridge.linear)")
