"""Integration test 2: train through the MinGRU recurrence (resident backward).

Setup: G, V, D (B, S, Hd) are the learnable gate projections (optimized directly).
  H        = MinGRU(G, V, D)            (forward in numpy, shader convention)
  pooled   = mean_t H[:, t, :]          (B, Hd)
  logits   = pooled @ Whead^T           (B, C)
  loss     = CE(logits, targets)

Backward path: CE -> Linear(head) -> mean-pool (manual) -> MinGRU.
The MinGRU grad (dL/dH) is seeded from the pooled head grad broadcast over time.
Proves the recurrence backward produces USABLE grads under iteration.

NOTE forward: x_scan=sigmoid(g)*tanh(v); a=0.001+0.998*sigmoid(d);
              h_t = a_t*h_{t-1}+x_scan_t   (matches mingru shaders).
"""
import sys
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np
import grilly_core as gc

np.random.seed(3)
B, S, Hd, C = 8, 5, 6, 3
G = (0.3*np.random.randn(B, S, Hd)).astype(np.float32)
V = (0.3*np.random.randn(B, S, Hd)).astype(np.float32)
D = (0.3*np.random.randn(B, S, Hd)).astype(np.float32)
Whead = (0.2*np.random.randn(C, Hd)).astype(np.float32)
targets = np.random.randint(0, C, size=B).astype(np.uint32)
targets_f = targets.astype(np.float32)

def sigmoid(z): return 1.0/(1.0+np.exp(-z))
def softmax(z):
    z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def ce_loss(lg, t): return -np.log(softmax(lg)[np.arange(len(t)),t]+1e-12).mean()

def mingru_forward(G,V,D):
    sg=sigmoid(G); tv=np.tanh(V); sd=sigmoid(D)
    x_scan=sg*tv; a=0.001+0.998*sd
    H=np.zeros_like(G)
    for t in range(S):
        prev=H[:,t-1,:] if t>0 else 0.0
        H[:,t,:]=a[:,t,:]*prev+x_scan[:,t,:]
    return H

dev = gc.Device(); dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")
def R(b,s,rg=True):
    r=gc.TensorRef(); r.buffer_id=b; r.set_shape(s); r.requires_grad=rg; return r

def grads(G,V,D,Whead):
    """Return dL/dG, dL/dV, dL/dD, dL/dWhead via resident backward."""
    H = mingru_forward(G,V,D)
    pooled = H.mean(axis=1)               # (B,Hd)
    logits = pooled @ Whead.T             # (B,C)

    tape = gc.TapeContext(dev); tape.begin()
    # --- head: Linear + CE (resident) ---
    p_id = tape.register_input(pooled, True)
    wh_id= tape.register_input(Whead, True)
    l_id = tape.register_input(logits, True)
    t_id = tape.register_input(targets_f, False)
    n_lin= tape.record_op(gc.OpType.Linear, [R(p_id,[B,Hd]), R(wh_id,[C,Hd])], [R(l_id,[B,C])])
    tape.save_for_backward(n_lin, [p_id, wh_id])
    n_ce = tape.record_op(gc.OpType.CrossEntropy, [R(l_id,[B,C])], [R(0,[1],False)])
    tape.save_for_backward(n_ce, [l_id, t_id])
    tape.backward(n_ce, 0)
    gWhead = tape.read_buffer(tape.get_grad_buffer(wh_id), [C,Hd]) / B
    g_pooled = tape.read_buffer(tape.get_grad_buffer(p_id), [B,Hd]) / B  # dL/dpooled (mean-CE)

    # mean-pool backward: dL/dH[:,t,:] = g_pooled / S  for all t
    gradH = np.repeat((g_pooled/S)[:,None,:], S, axis=1).astype(np.float32)

    # --- MinGRU backward (resident) ---
    tape2 = gc.TapeContext(dev); tape2.begin()
    g_id=tape2.register_input(G,True); v_id=tape2.register_input(V,True); d_id=tape2.register_input(D,True)
    h_id=tape2.register_input(H,True); gh_id=tape2.register_input(gradH,False)
    n=tape2.record_op(gc.OpType.MinGRU,
                      [R(g_id,[B,S,Hd]),R(v_id,[B,S,Hd]),R(d_id,[B,S,Hd])],[R(h_id,[B,S,Hd])])
    tape2.save_for_backward(n,[g_id,v_id,d_id,h_id])
    tape2.backward(n, gh_id)
    gG=tape2.read_buffer(tape2.get_grad_buffer(g_id),[B,S,Hd])
    gV=tape2.read_buffer(tape2.get_grad_buffer(v_id),[B,S,Hd])
    gD=tape2.read_buffer(tape2.get_grad_buffer(d_id),[B,S,Hd])
    return gG,gV,gD,gWhead

# AdamW on all params
def adamw_init(p): return [np.zeros_like(p), np.zeros_like(p)]
st = {k:adamw_init(v) for k,v in dict(G=G,V=V,D=D,Whead=Whead).items()}
lr,b1,b2,eps = 0.05,0.9,0.999,1e-8
def step_param(p, g, s, t):
    s[0]=b1*s[0]+(1-b1)*g; s[1]=b2*s[1]+(1-b2)*(g*g)
    mh=s[0]/(1-b1**t); vh=s[1]/(1-b2**t)
    return p - lr*(mh/(np.sqrt(vh)+eps))

print("step   loss   acc")
for step in range(1, 151):
    H=mingru_forward(G,V,D); pooled=H.mean(1); logits=pooled@Whead.T
    loss=ce_loss(logits,targets)
    if step==1 or step%15==0:
        acc=(softmax(logits).argmax(1)==targets).mean()
        print("%4d   %.4f   %.2f"%(step,loss,acc))
    gG,gV,gD,gWh = grads(G,V,D,Whead)
    G=step_param(G,gG,st['G'],step); V=step_param(V,gV,st['V'],step)
    D=step_param(D,gD,st['D'],step); Whead=step_param(Whead,gWh,st['Whead'],step)

H=mingru_forward(G,V,D); logits=H.mean(1)@Whead.T
fl=ce_loss(logits,targets); fa=(softmax(logits).argmax(1)==targets).mean()
print("\nfinal loss=%.4f acc=%.2f"%(fl,fa))
print("RESULT:", "PASS" if fl < 0.2 and fa==1.0 else "FAIL")
