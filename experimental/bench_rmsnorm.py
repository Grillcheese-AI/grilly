import time, inspect, numpy as np
from grilly.backend import _bridge

print("rmsnorm signature:", end=" ")
try:
    print(inspect.signature(_bridge.rmsnorm))
except Exception as e:
    print("(no sig)", e)

rng = np.random.default_rng(0)
B, S, F = 8, 512, 2048          # 4096 positions x 2048 feat -> 32 MB output
x = rng.standard_normal((B, S, F)).astype(np.float32)
w = rng.standard_normal((F,)).astype(np.float32)
eps = 1e-6

# reference
ms = np.mean(x.astype(np.float64)**2, axis=-1, keepdims=True)
ref = (x / np.sqrt(ms + eps) * w).astype(np.float32)

def call():
    # try a few common conventions
    for args in ((x, w, eps), (x, w), (x.reshape(-1, F), w, eps)):
        try:
            return _bridge.rmsnorm(*args)
        except Exception:
            continue
    raise RuntimeError("no rmsnorm call convention worked")

out = np.asarray(call()).reshape(ref.shape)
mad = float(np.max(np.abs(out - ref)))
print(f"correctness max_abs_diff = {mad:.3e}  shape={out.shape}")

# timing
N = 7
ts = []
for _ in range(N):
    t0 = time.perf_counter()
    _bridge.rmsnorm(x, w, eps)
    ts.append((time.perf_counter() - t0) * 1e3)
ts.sort()
print(f"rmsnorm {B}x{S}x{F} (32MB out): median {ts[N//2]:.1f} ms  min {ts[0]:.1f}  max {ts[-1]:.1f}")
