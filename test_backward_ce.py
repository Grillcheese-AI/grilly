"""Cross-entropy backward test, plus CE->Linear chain (the training entry point).

CE backward:  dL/dlogits = softmax(logits) - one_hot(targets)   (mean? no — per-row)
Then chain:   logits = h @ W^T ; loss = CE(logits, targets)
              dL/dW = dL/dlogits^T @ h ; dL/dh = dL/dlogits @ W
"""
import numpy as np
import grilly_core as gc

np.random.seed(2)
B, C = 4, 5           # batch, num classes
logits = np.random.randn(B, C).astype(np.float32)
targets = np.array([2, 0, 4, 1], dtype=np.uint32)
targets_f = targets.astype(np.float32)   # shader reads targets as float

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

sm = softmax(logits)
onehot = np.zeros_like(sm); onehot[np.arange(B), targets] = 1.0
ref_grad = sm - onehot          # (B,C)

dev = gc.Device()
dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")
tape = gc.TapeContext(dev)
tape.begin()

def ref(buf_id, shape, rg=True):
    r = gc.TensorRef(); r.buffer_id = buf_id; r.set_shape(shape); r.requires_grad = rg
    return r

# ---- Test 1: standalone CE backward ----
log_id = tape.register_input(logits, True)
tgt_id = tape.register_input(targets_f, False)

ce = tape.record_op(gc.OpType.CrossEntropy,
                    [ref(log_id,[B,C])],
                    [ref(0,[1], False)])
tape.save_for_backward(ce, [log_id, tgt_id])
tape.backward(ce, 0)   # loss node: no incoming grad buffer needed
print("CE stats:", tape.last_backward_stats())

grad = tape.read_buffer(tape.get_grad_buffer(log_id), [B, C])
e1 = np.abs(grad - ref_grad).max()
print("[%s] CE grad_logits max_abs_err=%.3e" % ("OK" if e1 < 1e-3 else "FAIL", e1))

# ---- Test 2: CE -> Linear chain (training entry point) ----
np.random.seed(3)
H = 6
h  = np.random.randn(B, H).astype(np.float32)
W  = np.random.randn(C, H).astype(np.float32)
logits2 = h @ W.T
sm2 = softmax(logits2)
oh2 = np.zeros_like(sm2); oh2[np.arange(B), targets] = 1.0
g_logits = sm2 - oh2
ref_gW = g_logits.T @ h          # (C,H)
ref_gh = g_logits @ W            # (B,H)

tape.begin()
h_id  = tape.register_input(h, True)
W_id  = tape.register_input(W, True)
l2_id = tape.register_input(logits2, True)
tgt2  = tape.register_input(targets_f, False)

n_lin = tape.record_op(gc.OpType.Linear,
                       [ref(h_id,[B,H]), ref(W_id,[C,H])],
                       [ref(l2_id,[B,C])])
tape.save_for_backward(n_lin, [h_id, W_id])

n_ce = tape.record_op(gc.OpType.CrossEntropy,
                      [ref(l2_id,[B,C])],
                      [ref(0,[1], False)])
tape.save_for_backward(n_ce, [l2_id, tgt2])

tape.backward(n_ce, 0)
print("chain stats:", tape.last_backward_stats())

gW = tape.read_buffer(tape.get_grad_buffer(W_id), [C, H])
gh = tape.read_buffer(tape.get_grad_buffer(h_id), [B, H])
e2 = np.abs(gW - ref_gW).max()
e3 = np.abs(gh - ref_gh).max()
print("[%s] chain grad_W max_abs_err=%.3e" % ("OK" if e2 < 1e-3 else "FAIL", e2))
print("[%s] chain grad_h max_abs_err=%.3e" % ("OK" if e3 < 1e-3 else "FAIL", e3))

ok = e1 < 1e-3 and e2 < 1e-3 and e3 < 1e-3
print("RESULT:", "PASS" if ok else "FAIL")
