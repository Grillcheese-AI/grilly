"""backward_mingru test. Forward (per the shader/header):
  x_scan = sigmoid(g) * tanh(v)
  a = 0.05 + 0.9*sigmoid(d)
  h_t = a_t * h_{t-1} + x_scan_t   (h_{-1} = 0)
Layout: (batch, seq, hidden), row-major.

Backward: given dL/dh (gradH), produce dL/dg, dL/dv, dL/dd.
"""
import numpy as np
import grilly_core as gc

np.random.seed(21)
Bt, S, Hd = 2, 6, 4
g = np.random.randn(Bt, S, Hd).astype(np.float32)
v = np.random.randn(Bt, S, Hd).astype(np.float32)
d = np.random.randn(Bt, S, Hd).astype(np.float32)
gradH = np.random.randn(Bt, S, Hd).astype(np.float32)

def sigmoid(z): return 1.0/(1.0+np.exp(-z))

sg = sigmoid(g); tv = np.tanh(v); sd = sigmoid(d)
x_scan = sg * tv
a = 0.001 + 0.998*sd          # matches mingru shaders (header comment is stale)

# forward scan -> H
H = np.zeros_like(g)
for t in range(S):
    prev = H[:, t-1, :] if t > 0 else 0.0
    H[:, t, :] = a[:, t, :]*prev + x_scan[:, t, :]

# backward scan: dh_t = gradH_t + a_{t+1}*dh_{t+1}
dh = np.zeros_like(g)
for t in reversed(range(S)):
    nxt = a[:, t+1, :]*dh[:, t+1, :] if t < S-1 else 0.0
    dh[:, t, :] = gradH[:, t, :] + nxt

# x_scan_t enters h_t directly:   dL/dx_scan_t = dh_t
dx = dh
# a_t multiplies h_{t-1}:         dL/da_t = dh_t * h_{t-1}
da = np.zeros_like(g)
for t in range(S):
    prev = H[:, t-1, :] if t > 0 else 0.0
    da[:, t, :] = dh[:, t, :]*prev

# chain to g,v,d
# x_scan = sigmoid(g)*tanh(v)
dg = dx * tv * (sg*(1-sg))
dv = dx * sg * (1 - tv*tv)
# a = 0.001+0.998*sigmoid(d)
dd = da * 0.998 * (sd*(1-sd))

dev = gc.Device()
dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")
tape = gc.TapeContext(dev)
tape.begin()

def R(b, s, rg=True):
    r = gc.TensorRef(); r.buffer_id=b; r.set_shape(s); r.requires_grad=rg; return r

g_id = tape.register_input(g, True)
v_id = tape.register_input(v, True)
d_id = tape.register_input(d, True)
h_id = tape.register_input(H, True)       # saved forward output
go_id = tape.register_input(gradH, False)

n = tape.record_op(gc.OpType.MinGRU,
                   [R(g_id,[Bt,S,Hd]), R(v_id,[Bt,S,Hd]), R(d_id,[Bt,S,Hd])],
                   [R(h_id,[Bt,S,Hd])])
tape.save_for_backward(n, [g_id, v_id, d_id, h_id])

tape.backward(n, go_id)
print("stats:", tape.last_backward_stats())

gG = tape.read_buffer(tape.get_grad_buffer(g_id), [Bt,S,Hd])
gV = tape.read_buffer(tape.get_grad_buffer(v_id), [Bt,S,Hd])
gD = tape.read_buffer(tape.get_grad_buffer(d_id), [Bt,S,Hd])

def rep(name, got, ref):
    e = np.abs(got-ref).max()
    print("[%s] %s max_abs_err=%.3e" % ("OK" if e < 1e-3 else "FAIL", name, e))
    return e < 1e-3

ok = all([rep("grad_g", gG, dg), rep("grad_v", gV, dv), rep("grad_d", gD, dd)])
print("RESULT:", "PASS" if ok else "FAIL")
