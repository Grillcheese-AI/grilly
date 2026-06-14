"""Isolated correctness test for the resident backward_linear handler.

Builds one Linear node on the tape, runs backward, reads the three gradient
buffers back, and compares against numpy references.

Linear forward:  y = x @ W^T      (W is (out, in))
Backward:
  dL/dx = dL/dy @ W
  dL/dW = dL/dy^T @ x
  dL/db = sum(dL/dy, axis=0)
"""
import numpy as np
import grilly_core as gc

np.random.seed(0)

BATCH, IN, OUT = 4, 5, 3
x = np.random.randn(BATCH, IN).astype(np.float32)
W = np.random.randn(OUT, IN).astype(np.float32)
grad_out = np.random.randn(BATCH, OUT).astype(np.float32)

# numpy reference grads
ref_grad_x = grad_out @ W                 # (BATCH, IN)
ref_grad_W = grad_out.T @ x               # (OUT, IN)
ref_grad_b = grad_out.sum(axis=0)         # (OUT,)

dev = gc.Device()
shader_dir = r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv"
dev.load_shaders(shader_dir)

tape = gc.TapeContext(dev)
tape.begin()

# Register resident buffers (returns registry buffer_ids)
x_id = tape.register_input(x, True)
W_id = tape.register_input(W, True)
go_id = tape.register_input(grad_out, False)

# Build TensorRefs for the Linear node
x_ref = gc.TensorRef()
x_ref.buffer_id = x_id
x_ref.set_shape([BATCH, IN])
x_ref.requires_grad = True

W_ref = gc.TensorRef()
W_ref.buffer_id = W_id
W_ref.set_shape([OUT, IN])
W_ref.requires_grad = True

y_ref = gc.TensorRef()
y_ref.set_shape([BATCH, OUT])

node = tape.record_op(gc.OpType.Linear, [x_ref, W_ref], [y_ref])
tape.save_for_backward(node, [x_id, W_id])

# Run backward, seeding dL/dy = grad_out
tape.backward(node, go_id)
print("backward stats:", tape.last_backward_stats())

gx_id = tape.get_grad_buffer(x_id)
gW_id = tape.get_grad_buffer(W_id)
print("grad ids: x=%d W=%d" % (gx_id, gW_id))

grad_x = tape.read_buffer(gx_id, [BATCH, IN])
grad_W = tape.read_buffer(gW_id, [OUT, IN])

def report(name, got, ref):
    err = np.abs(got - ref).max()
    rel = err / (np.abs(ref).max() + 1e-8)
    ok = "OK" if err < 1e-3 else "FAIL"
    print("[%s] %s  max_abs_err=%.3e  rel=%.3e" % (ok, name, err, rel))
    return err < 1e-3

ok_x = report("grad_x", grad_x, ref_grad_x)
ok_W = report("grad_W", grad_W, ref_grad_W)

print("RESULT:", "PASS" if (ok_x and ok_W) else "FAIL")
