"""Unit gate for step 3: resident AdamW (adamw-update.glsl via TapeContext) vs a
numpy AdamW reference, over N steps with PERSISTENT resident W/m/v buffers.

Validates two new things together:
  - BufferRegistry persistent entries survive begin()/clear() with a stable id
    (W, m, v registered once; the per-step grad is step-scoped and re-registered).
  - adamw_update matches numpy AdamW (decoupled weight decay + bias correction)
    bit-closely, step after step, with the moments carried resident across steps.
"""
import sys
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np
import grilly_core as gc

np.random.seed(3)
N_W, STEPS = 1000, 25
lr, b1, b2, eps, wd = 0.01, 0.9, 0.999, 1e-8, 0.01

W0 = (0.1 * np.random.randn(N_W)).astype(np.float32)
grads = [(0.5 * np.random.randn(N_W)).astype(np.float32) for _ in range(STEPS)]

dev = gc.Device(); dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")

t = gc.TapeContext(dev)
w_id = t.register_weight(W0.copy())                  # persistent resident
m_id = t.register_weight(np.zeros(N_W, np.float32))  # persistent Adam m
v_id = t.register_weight(np.zeros(N_W, np.float32))  # persistent Adam v

# numpy reference state
W = W0.copy().astype(np.float64); m = np.zeros(N_W); v = np.zeros(N_W)

print("=== resident AdamW vs numpy AdamW (persistent resident W/m/v) ===")
worst = 0.0
for step in range(1, STEPS + 1):
    g = grads[step - 1]
    t.begin()                                        # clears step buffers; W/m/v persist
    g_id = t.register_input(g, False)                # step-scoped grad
    b1t, b2t = b1 ** step, b2 ** step
    t.forward_begin()                                # generic batch open
    t.adamw_update(w_id, g_id, m_id, v_id, N_W, lr, b1, b2, eps, wd, b1t, b2t, False)
    t.forward_submit()
    W_gpu = t.read_buffer(w_id, [N_W])

    # numpy AdamW (mirror adamw-update.glsl exactly)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g * g)
    mh = m / (1 - b1t); vh = v / (1 - b2t)
    W = W * (1 - lr * wd) - lr * mh / (np.sqrt(np.maximum(vh, 0.0)) + eps)

    err = float(np.abs(W_gpu - W).max())
    worst = max(worst, err)
    if step == 1 or step % 5 == 0:
        print("  step %2d  max_abs_diff=%.3e" % (step, err))

print("\nworst max_abs_diff over %d steps = %.3e" % (STEPS, worst))
ok = worst < 1e-5
print("RESIDENT-ADAMW:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
