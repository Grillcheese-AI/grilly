"""Fan-out accumulation test.

x feeds TWO linear layers; their outputs are added:
    a = x @ W1^T
    b = x @ W2^T
    y = a + b
dL/dx must accumulate BOTH paths:  dL/dx = grad_y @ W1 + grad_y @ W2
This is the only test that exercises the find_or_insert_grad second-contribution
batchedAdd (elementwise-add) fan-out path.
"""
import numpy as np
import grilly_core as gc

np.random.seed(7)
B, IN, OUT = 4, 5, 3
x  = np.random.randn(B, IN).astype(np.float32)
W1 = np.random.randn(OUT, IN).astype(np.float32)
W2 = np.random.randn(OUT, IN).astype(np.float32)
grad_y = np.random.randn(B, OUT).astype(np.float32)

# numpy reference
ref_gx = grad_y @ W1 + grad_y @ W2     # (B,IN) — fan-out sum
ref_gW1 = grad_y.T @ x
ref_gW2 = grad_y.T @ x

dev = gc.Device()
dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")
tape = gc.TapeContext(dev)
tape.begin()

def R(b, s, rg=True):
    r = gc.TensorRef(); r.buffer_id = b; r.set_shape(s); r.requires_grad = rg; return r

x_id  = tape.register_input(x, True)
W1_id = tape.register_input(W1, True)
W2_id = tape.register_input(W2, True)
a_id  = tape.register_input(np.zeros((B,OUT),np.float32), True)
b_id  = tape.register_input(np.zeros((B,OUT),np.float32), True)
go_id = tape.register_input(grad_y, False)

# a = x @ W1^T
n_a = tape.record_op(gc.OpType.Linear, [R(x_id,[B,IN]), R(W1_id,[OUT,IN])], [R(a_id,[B,OUT])])
tape.save_for_backward(n_a, [x_id, W1_id])
# b = x @ W2^T
n_b = tape.record_op(gc.OpType.Linear, [R(x_id,[B,IN]), R(W2_id,[OUT,IN])], [R(b_id,[B,OUT])])
tape.save_for_backward(n_b, [x_id, W2_id])
# y = a + b   (output buffer id unused; we seed its grad)
n_y = tape.record_op(gc.OpType.Add, [R(a_id,[B,OUT]), R(b_id,[B,OUT])], [R(0,[B,OUT],False)])

tape.backward(n_y, go_id)
print("stats:", tape.last_backward_stats())

gx  = tape.read_buffer(tape.get_grad_buffer(x_id),  [B, IN])
gW1 = tape.read_buffer(tape.get_grad_buffer(W1_id), [OUT, IN])
gW2 = tape.read_buffer(tape.get_grad_buffer(W2_id), [OUT, IN])

def rep(name, got, r):
    e = np.abs(got - r).max()
    print("[%s] %s max_abs_err=%.3e" % ("OK" if e < 1e-3 else "FAIL", name, e))
    return e < 1e-3

ok = all([rep("grad_x (fan-out)", gx, ref_gx),
          rep("grad_W1", gW1, ref_gW1),
          rep("grad_W2", gW2, ref_gW2)])
print("RESULT:", "PASS" if ok else "FAIL")
