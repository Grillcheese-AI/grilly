"""backward_rmsnorm test. RMSNorm: y_i = w_f * x_i * r, r = 1/sqrt(mean(x^2)+eps).
Per row (n=features):
  c = sum_j(g_j * w_j * x_j)
  dL/dx_i = r*w_i*g_i - (r^3/n)*x_i*c
  dL/dw_f = sum_positions(g * x * r)
"""
import numpy as np
import grilly_core as gc

np.random.seed(11)
B, F = 4, 8            # positions, features
eps = 1e-6
x = np.random.randn(B, F).astype(np.float32)
w = np.random.randn(F).astype(np.float32)
grad_y = np.random.randn(B, F).astype(np.float32)

# numpy reference
mean_sq = (x**2).mean(axis=1, keepdims=True)         # (B,1)
r = 1.0 / np.sqrt(mean_sq + eps)                      # (B,1)
# grad_x
c = (grad_y * w[None,:] * x).sum(axis=1, keepdims=True)   # (B,1)
ref_gx = r * w[None,:] * grad_y - (r**3 / F) * x * c
# grad_w
ref_gw = (grad_y * x * r).sum(axis=0)                # (F,)

dev = gc.Device()
dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")
tape = gc.TapeContext(dev)
tape.begin()

def R(b, s, rg=True):
    rr = gc.TensorRef(); rr.buffer_id = b; rr.set_shape(s); rr.requires_grad = rg; return rr

x_id  = tape.register_input(x, True)
w_id  = tape.register_input(w, True)
y_id  = tape.register_input(np.zeros((B,F),np.float32), True)
go_id = tape.register_input(grad_y, False)

n = tape.record_op(gc.OpType.RMSNorm, [R(x_id,[B,F]), R(w_id,[F])], [R(y_id,[B,F])])
tape.save_for_backward(n, [x_id, w_id])

tape.backward(n, go_id)
print("stats:", tape.last_backward_stats())

gx = tape.read_buffer(tape.get_grad_buffer(x_id), [B, F])
gw = tape.read_buffer(tape.get_grad_buffer(w_id), [F])

def rep(name, got, ref):
    e = np.abs(got - ref).max()
    print("[%s] %s max_abs_err=%.3e" % ("OK" if e < 1e-3 else "FAIL", name, e))
    return e < 1e-3

ok = rep("grad_x", gx, ref_gx) & rep("grad_w", gw, ref_gw)
print("RESULT:", "PASS" if ok else "FAIL")
