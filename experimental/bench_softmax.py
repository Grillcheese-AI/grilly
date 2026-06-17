import time, inspect, numpy as np
from grilly.backend import _bridge

cands = [a for a in dir(_bridge) if 'softmax' in a.lower()]
print("softmax bridge attrs:", cands)
fn = _bridge.softmax if hasattr(_bridge, 'softmax') else getattr(_bridge, cands[0])
try: print("sig:", inspect.signature(fn))
except Exception as e: print("(no sig)", e)

rng = np.random.default_rng(0)
B, S, F = 8, 512, 2048           # 32 MB output
x = rng.standard_normal((B, S, F)).astype(np.float32)

xm = x - x.max(axis=-1, keepdims=True)
e = np.exp(xm.astype(np.float64))
ref = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)

def call():
    for args in ((x,), (x, B, S, F), (x.reshape(-1, F),)):
        try: return _bridge.softmax(*args)
        except Exception: continue
    raise RuntimeError("no softmax call convention worked")

out = np.asarray(call()).reshape(ref.shape)
print(f"correctness max_abs_diff = {float(np.max(np.abs(out-ref))):.3e}  rowsum~{float(out[0,0].sum()):.5f}")

ts = []
for _ in range(7):
    t0 = time.perf_counter(); _bridge.softmax(x); ts.append((time.perf_counter()-t0)*1e3)
ts.sort()
print(f"softmax {B}x{S}x{F} (32MB out): median {ts[3]:.1f} ms  min {ts[0]:.1f}")
