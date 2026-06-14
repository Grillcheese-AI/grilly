"""backward_silu test. SiLU: y = x*sigmoid(x).
dL/dx = dL/dy * (sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x)))
The handler takes grad_output (incoming) and saved input x.
"""
import numpy as np
import grilly_core as gc

np.random.seed(4)
N = 32
x = np.random.randn(N).astype(np.float32)
grad_out = np.random.randn(N).astype(np.float32)

sig = 1.0 / (1.0 + np.exp(-x))
dsilu = sig + x * sig * (1.0 - sig)          # d/dx [x*sigmoid(x)]
ref_grad = grad_out * dsilu

dev = gc.Device()
dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")
tape = gc.TapeContext(dev)
tape.begin()

def ref(buf_id, shape, rg=True):
    r = gc.TensorRef(); r.buffer_id = buf_id; r.set_shape(shape); r.requires_grad = rg
    return r

x_id  = tape.register_input(x, True)
go_id = tape.register_input(grad_out, False)
y_id  = tape.register_input(np.zeros_like(x), True)  # forward output buffer

n = tape.record_op(gc.OpType.SiLU, [ref(x_id,[N])], [ref(y_id,[N])])
tape.save_for_backward(n, [x_id])     # saved input = x

tape.backward(n, go_id)               # seed dL/dy = grad_out
print("stats:", tape.last_backward_stats())

grad = tape.read_buffer(tape.get_grad_buffer(x_id), [N])
e = np.abs(grad - ref_grad).max()
print("[%s] silu grad max_abs_err=%.3e" % ("OK" if e < 1e-3 else "FAIL", e))
print("RESULT:", "PASS" if e < 1e-3 else "FAIL")
