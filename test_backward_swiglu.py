"""backward_swiglu test. SwiGLU: out = x1*silu(x2), input=[x1|x2] concat (2*hidden).
  dL/dx1 = grad_out * silu(x2)
  dL/dx2 = grad_out * x1 * silu'(x2),  silu'(z)=sigmoid(z)*(1+z*(1-sigmoid(z)))
grad_input is 2*hidden wide: [dL/dx1 | dL/dx2].
"""
import sys
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np
import grilly_core as gc

np.random.seed(5)
B, Hd = 4, 6
x1 = np.random.randn(B, Hd).astype(np.float32)
x2 = np.random.randn(B, Hd).astype(np.float32)
inp = np.concatenate([x1, x2], axis=1).astype(np.float32)
grad_out = np.random.randn(B, Hd).astype(np.float32)

def sig(z): return 1.0/(1.0+np.exp(-z))
silu = x2*sig(x2)
dsilu = sig(x2)*(1.0 + x2*(1.0-sig(x2)))
ref = np.concatenate([grad_out*silu, grad_out*x1*dsilu], axis=1)

dev = gc.Device(); dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")
t = gc.TapeContext(dev); t.begin()
def R(b,s,rg=True):
    r=gc.TensorRef(); r.buffer_id=b; r.set_shape(s); r.requires_grad=rg; return r

in_id = t.register_input(inp, True)
o_id  = t.register_input(np.zeros((B,Hd),np.float32), True)
go_id = t.register_input(grad_out, False)
n = t.record_op(gc.OpType.SwiGLU, [R(in_id,[B,2*Hd])], [R(o_id,[B,Hd])])
t.save_for_backward(n, [in_id])
t.backward(n, go_id)
print("stats:", t.last_backward_stats())

g = t.read_buffer(t.get_grad_buffer(in_id), [B, 2*Hd])
e = np.abs(g-ref).max()
print("[%s] swiglu grad max_abs_err=%.3e" % ("OK" if e<1e-3 else "FAIL", e))
print("RESULT:", "PASS" if e<1e-3 else "FAIL")
