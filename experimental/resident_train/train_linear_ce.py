"""Integration test 1: end-to-end training loop using the RESIDENT backward engine.

Task: memorize a tiny classification dataset (B samples, IN features -> C classes).
  forward:  logits = x @ W^T        (numpy)
            loss   = CE(logits, targets)
  backward: resident grilly autograd  (backward_cross_entropy + backward_linear)
  optim:    numpy AdamW on W

Proves the full loop: forward -> resident grad read-back -> optimizer step ->
re-register -> repeat, and that grad sign/scale actually descend the loss.
"""
import sys
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np
import grilly_core as gc

np.random.seed(0)
B, IN, C = 16, 8, 4
x = np.random.randn(B, IN).astype(np.float32)
targets = np.random.randint(0, C, size=B).astype(np.uint32)
targets_f = targets.astype(np.float32)        # CE shader reads targets as float
W = (0.1 * np.random.randn(C, IN)).astype(np.float32)

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def ce_loss(logits, tgt):
    sm = softmax(logits)
    return -np.log(sm[np.arange(len(tgt)), tgt] + 1e-12).mean()

dev = gc.Device()
dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")

def R(b, s, rg=True):
    r = gc.TensorRef(); r.buffer_id = b; r.set_shape(s); r.requires_grad = rg; return r

def resident_grad_W(W, x, targets_f):
    """One forward+backward through the resident engine; return dL/dW (mean over batch)."""
    logits = x @ W.T                      # (B,C) forward in numpy
    tape = gc.TapeContext(dev); tape.begin()
    x_id  = tape.register_input(x, True)
    W_id  = tape.register_input(W, True)
    l_id  = tape.register_input(logits, True)
    t_id  = tape.register_input(targets_f, False)
    n_lin = tape.record_op(gc.OpType.Linear, [R(x_id,[B,IN]), R(W_id,[C,IN])], [R(l_id,[B,C])])
    tape.save_for_backward(n_lin, [x_id, W_id])
    n_ce  = tape.record_op(gc.OpType.CrossEntropy, [R(l_id,[B,C])], [R(0,[1],False)])
    tape.save_for_backward(n_ce, [l_id, t_id])
    tape.backward(n_ce, 0)
    gW = tape.read_buffer(tape.get_grad_buffer(W_id), [C, IN])
    # CE backward returns per-sample (softmax-onehot); linear grad_W sums over batch.
    # Divide by B to match the mean-CE-loss convention used in ce_loss().
    return gW / B

# numpy AdamW state
mW = np.zeros_like(W); vW = np.zeros_like(W)
lr, b1, b2, eps, wd = 0.05, 0.9, 0.999, 1e-8, 0.0

print("step   loss")
for step in range(1, 201):
    logits = x @ W.T
    loss = ce_loss(logits, targets)
    if step == 1 or step % 20 == 0:
        acc = (softmax(logits).argmax(1) == targets).mean()
        print("%4d   %.4f   acc=%.2f" % (step, loss, acc))
    gW = resident_grad_W(W, x, targets_f)
    # AdamW
    mW = b1*mW + (1-b1)*gW
    vW = b2*vW + (1-b2)*(gW*gW)
    mhat = mW / (1 - b1**step); vhat = vW / (1 - b2**step)
    W = W - lr * (mhat / (np.sqrt(vhat) + eps) + wd*W)

final_loss = ce_loss(x @ W.T, targets)
final_acc = (softmax(x @ W.T).argmax(1) == targets).mean()
print("\nfinal loss=%.4f acc=%.2f" % (final_loss, final_acc))
print("RESULT:", "PASS" if final_loss < 0.1 and final_acc == 1.0 else "FAIL")
