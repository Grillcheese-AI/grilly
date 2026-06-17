"""Multi-op propagation test: a 2-layer Linear chain.

  y1 = x  @ W1^T     (BATCH,H)
  y2 = y1 @ W2^T     (BATCH,OUT)

Seed grad_y2, run backward over the whole tape, and verify grad_W2, grad_W1,
and grad_x against numpy. This exercises:
  - backward_linear at two nodes
  - the pull-based propagation (y1's producer node activates from the grad
    deposited under y1's buffer_id by the second Linear's input-grad)
"""
import numpy as np
import grilly_core as gc

np.random.seed(1)
BATCH, IN, H, OUT = 4, 5, 6, 3
x  = np.random.randn(BATCH, IN).astype(np.float32)
W1 = np.random.randn(H, IN).astype(np.float32)
W2 = np.random.randn(OUT, H).astype(np.float32)
grad_y2 = np.random.randn(BATCH, OUT).astype(np.float32)

# numpy reference (forward then backward)
y1 = x @ W1.T                       # (BATCH,H)
# y2 = y1 @ W2.T
grad_y1 = grad_y2 @ W2              # (BATCH,H)
ref_grad_W2 = grad_y2.T @ y1       # (OUT,H)
ref_grad_W1 = grad_y1.T @ x        # (H,IN)
ref_grad_x  = grad_y1 @ W1         # (BATCH,IN)

dev = gc.Device()
dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")
tape = gc.TapeContext(dev)
tape.begin()

x_id  = tape.register_input(x, True)
W1_id = tape.register_input(W1, True)
W2_id = tape.register_input(W2, True)
y1_id = tape.register_input(y1, True)   # intermediate buffer (forward value)
go_id = tape.register_input(grad_y2, False)

def ref(buf_id, shape, rg=True):
    r = gc.TensorRef(); r.buffer_id = buf_id; r.set_shape(shape); r.requires_grad = rg
    return r

# Node 1: y1 = x @ W1^T
n1 = tape.record_op(gc.OpType.Linear,
                    [ref(x_id,[BATCH,IN]), ref(W1_id,[H,IN])],
                    [ref(y1_id,[BATCH,H])])
tape.save_for_backward(n1, [x_id, W1_id])

# Node 2: y2 = y1 @ W2^T   (output buffer id can be 0/unused; we seed its grad)
y2_ref = ref(0,[BATCH,OUT], False)
n2 = tape.record_op(gc.OpType.Linear,
                    [ref(y1_id,[BATCH,H]), ref(W2_id,[OUT,H])],
                    [y2_ref])
tape.save_for_backward(n2, [y1_id, W2_id])

tape.backward(n2, go_id)
print("stats:", tape.last_backward_stats())

gW2 = tape.read_buffer(tape.get_grad_buffer(W2_id), [OUT, H])
gW1 = tape.read_buffer(tape.get_grad_buffer(W1_id), [H, IN])
gx  = tape.read_buffer(tape.get_grad_buffer(x_id),  [BATCH, IN])

def rep(name, got, r):
    e = np.abs(got - r).max()
    ok = "OK" if e < 1e-3 else "FAIL"
    print("[%s] %s max_abs_err=%.3e" % (ok, name, e))
    return e < 1e-3

ok = all([
    rep("grad_W2", gW2, ref_grad_W2),
    rep("grad_W1", gW1, ref_grad_W1),
    rep("grad_x",  gx,  ref_grad_x),
])
print("RESULT:", "PASS" if ok else "FAIL")
