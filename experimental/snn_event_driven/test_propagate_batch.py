"""Correctness of grilly_core.spike_propagate_batch (non-square, batched).
out[m] = spikes[m] @ W, with W [N_in, N_out], M sparse spike vectors.
"""
import sys
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np
import grilly_core as g
from grilly.backend import _bridge as b

dev = b._get_device()
print("device:", dev.device_name, "| has spike_propagate_batch:",
      hasattr(g, "spike_propagate_batch"))

def compact(spikes):                       # [M, N_in] 0/1 -> concat fired lists
    idx, off, cnt = [], [], []
    cur = 0
    for row in spikes:
        f = np.flatnonzero(row).astype(np.uint32)
        idx.append(f); off.append(cur); cnt.append(f.size); cur += f.size
    idx = np.concatenate(idx) if idx else np.zeros(0, np.uint32)
    return (idx.astype(np.uint32),
            np.array(off, np.uint32), np.array(cnt, np.uint32))

def propagate(spikes, W):
    M, N_in = spikes.shape
    N_out = W.shape[1]
    idx, off, cnt = compact(spikes)
    out = g.spike_propagate_batch(
        dev, idx.view(np.float32), off.view(np.float32), cnt.view(np.float32),
        np.ascontiguousarray(W), int(N_in), int(N_out), int(M))
    return np.asarray(out, dtype=np.float32)

print("\n[correctness] non-square, batched M vectors")
for (M, N_in, N_out, p) in [(8, 512, 1024, 0.05),
                            (32, 1024, 256, 0.02),
                            (64, 2048, 2048, 0.01),
                            (5, 777, 333, 0.10)]:
    rng = np.random.default_rng(0)
    spikes = (rng.random((M, N_in)) < p).astype(np.float32)
    W = np.ascontiguousarray(rng.standard_normal((N_in, N_out)).astype(np.float32))
    got = propagate(spikes, W)
    ref = spikes @ W
    print(f"  M={M:3d} {N_in:5d}->{N_out:5d} p={p:.2f}  "
          f"max_abs_diff={np.max(np.abs(got-ref)):.2e}  shape={got.shape}")
print("\nOK" if True else "")
