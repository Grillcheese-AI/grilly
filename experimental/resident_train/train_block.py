"""Integration test 3: FULL Cubby-style block, resident backward, gradient-checked.

Single connected tape from CE all the way back to X, exercising EVERY handler:
  RMSNorm, Linear (x4: WG/WV/WD/Wg/Whead), MinGRU, Add (x2 residuals), SwiGLU, CE.

Block (S=1 so pooling is identity -> one connected graph; MinGRU still scans):
  n1     = RMSNorm(X, w1)
  Gp,Vp,Dp = n1 @ {WG,WV,WD}^T
  H      = MinGRU(Gp,Vp,Dp)
  r1     = X + H                         (residual 1, fan-out on X and H)
  n2     = RMSNorm(r1, w2)
  gate   = n2 @ Wg^T                      (Wg: 2Hd x Hd)
  ff     = SwiGLU(gate)
  r2     = r1 + ff                        (residual 2, fan-out on r1)
  logits = r2 @ Whead^T                   (pooled == r2 since S=1)
  loss   = CE(logits, targets)

Validate dL/dX (resident) against numpy finite differences, then train.
"""
import sys
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np
import grilly_core as gc

np.random.seed(7)
B, Hd, C = 4, 8, 4
S = 1
BS = B*S
EPS = 1e-6

def randn(*s, sc=0.2): return (sc*np.random.randn(*s)).astype(np.float32)
P = dict(
    w1=np.ones(Hd, np.float32), w2=np.ones(Hd, np.float32),
    WG=randn(Hd,Hd), WV=randn(Hd,Hd), WD=randn(Hd,Hd),
    Wg=randn(2*Hd,Hd), Whead=randn(C,Hd),
)
X = randn(BS,Hd, sc=0.5)               # (BS, Hd) flat; (B,1,Hd) for MinGRU
targets = np.random.randint(0,C,size=B).astype(np.uint32)
targets_f = targets.astype(np.float32)

def sig(z): return 1.0/(1.0+np.exp(-z))
def softmax(z):
    z=z-z.max(-1,keepdims=True); e=np.exp(z); return e/e.sum(-1,keepdims=True)
def rmsnorm(x,w):
    ms=(x*x).mean(-1,keepdims=True); r=1.0/np.sqrt(ms+EPS); return x*r*w
def mingru_flat(G,V,D):                 # (BS,Hd) with S=1 -> h = x_scan
    sg=sig(G); tv=np.tanh(V); sd=sig(D); xs=sg*tv
    return xs                           # a*h_{-1}=0, so H = x_scan
def swiglu(gate):
    x1=gate[...,:Hd]; x2=gate[...,Hd:]; return x1*(x2*sig(x2))

def forward(P, X, full=False):
    n1 = rmsnorm(X, P['w1'])
    Gp = n1 @ P['WG'].T; Vp = n1 @ P['WV'].T; Dp = n1 @ P['WD'].T
    H  = mingru_flat(Gp,Vp,Dp)
    r1 = X + H
    n2 = rmsnorm(r1, P['w2'])
    gate = n2 @ P['Wg'].T
    ff = swiglu(gate)
    r2 = r1 + ff
    logits = r2 @ P['Whead'].T
    loss = -np.log(softmax(logits)[np.arange(B),targets]+1e-12).mean()
    if not full: return loss
    return dict(n1=n1,Gp=Gp,Vp=Vp,Dp=Dp,H=H,r1=r1,n2=n2,gate=gate,ff=ff,
                r2=r2,logits=logits,loss=loss)

dev = gc.Device(); dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")
def R(b,s,rg=True):
    r=gc.TensorRef(); r.buffer_id=b; r.set_shape(s); r.requires_grad=rg; return r

def resident_grads(P, X):
    f = forward(P, X, full=True)
    t = gc.TapeContext(dev); t.begin()
    X_id=t.register_input(X,True)
    w1_id=t.register_input(P['w1'],True); w2_id=t.register_input(P['w2'],True)
    WG_id=t.register_input(P['WG'],True); WV_id=t.register_input(P['WV'],True); WD_id=t.register_input(P['WD'],True)
    Wg_id=t.register_input(P['Wg'],True); Wh_id=t.register_input(P['Whead'],True)
    tgt_id=t.register_input(targets_f,False)
    n1_id=t.register_input(f['n1'],True)
    Gp_id=t.register_input(f['Gp'],True); Vp_id=t.register_input(f['Vp'],True); Dp_id=t.register_input(f['Dp'],True)
    H_id=t.register_input(f['H'],True); r1_id=t.register_input(f['r1'],True)
    n2_id=t.register_input(f['n2'],True); gate_id=t.register_input(f['gate'],True)
    ff_id=t.register_input(f['ff'],True); r2_id=t.register_input(f['r2'],True)
    logits_id=t.register_input(f['logits'],True)

    nn1=t.record_op(gc.OpType.RMSNorm,[R(X_id,[BS,Hd]),R(w1_id,[Hd])],[R(n1_id,[BS,Hd])])
    t.save_for_backward(nn1,[X_id,w1_id])
    nG=t.record_op(gc.OpType.Linear,[R(n1_id,[BS,Hd]),R(WG_id,[Hd,Hd])],[R(Gp_id,[BS,Hd])]); t.save_for_backward(nG,[n1_id,WG_id])
    nV=t.record_op(gc.OpType.Linear,[R(n1_id,[BS,Hd]),R(WV_id,[Hd,Hd])],[R(Vp_id,[BS,Hd])]); t.save_for_backward(nV,[n1_id,WV_id])
    nD=t.record_op(gc.OpType.Linear,[R(n1_id,[BS,Hd]),R(WD_id,[Hd,Hd])],[R(Dp_id,[BS,Hd])]); t.save_for_backward(nD,[n1_id,WD_id])
    nM=t.record_op(gc.OpType.MinGRU,[R(Gp_id,[B,S,Hd]),R(Vp_id,[B,S,Hd]),R(Dp_id,[B,S,Hd])],[R(H_id,[B,S,Hd])])
    t.save_for_backward(nM,[Gp_id,Vp_id,Dp_id,H_id])
    nA1=t.record_op(gc.OpType.Add,[R(X_id,[BS,Hd]),R(H_id,[BS,Hd])],[R(r1_id,[BS,Hd])])
    nn2=t.record_op(gc.OpType.RMSNorm,[R(r1_id,[BS,Hd]),R(w2_id,[Hd])],[R(n2_id,[BS,Hd])]); t.save_for_backward(nn2,[r1_id,w2_id])
    nGate=t.record_op(gc.OpType.Linear,[R(n2_id,[BS,Hd]),R(Wg_id,[2*Hd,Hd])],[R(gate_id,[BS,2*Hd])]); t.save_for_backward(nGate,[n2_id,Wg_id])
    nSw=t.record_op(gc.OpType.SwiGLU,[R(gate_id,[BS,2*Hd])],[R(ff_id,[BS,Hd])]); t.save_for_backward(nSw,[gate_id])
    nA2=t.record_op(gc.OpType.Add,[R(r1_id,[BS,Hd]),R(ff_id,[BS,Hd])],[R(r2_id,[BS,Hd])])
    nHead=t.record_op(gc.OpType.Linear,[R(r2_id,[B,Hd]),R(Wh_id,[C,Hd])],[R(logits_id,[B,C])]); t.save_for_backward(nHead,[r2_id,Wh_id])
    nCE=t.record_op(gc.OpType.CrossEntropy,[R(logits_id,[B,C])],[R(0,[1],False)]); t.save_for_backward(nCE,[logits_id,tgt_id])

    t.backward(nCE,0)
    def gr(bid,sh): return t.read_buffer(t.get_grad_buffer(bid),sh)/B   # mean-CE
    return dict(
        X=gr(X_id,[BS,Hd]), w1=gr(w1_id,[Hd]), w2=gr(w2_id,[Hd]),
        WG=gr(WG_id,[Hd,Hd]), WV=gr(WV_id,[Hd,Hd]), WD=gr(WD_id,[Hd,Hd]),
        Wg=gr(Wg_id,[2*Hd,Hd]), Whead=gr(Wh_id,[C,Hd]),
    )

# ---------- gradient check dL/dX via finite differences ----------
print("forward loss:", forward(P, X))
g = resident_grads(P, X)
gX = g['X']
h = 1e-3
fd = np.zeros_like(X)
for i in range(BS):
    for j in range(Hd):
        Xp = X.copy(); Xp[i,j]+=h
        Xm = X.copy(); Xm[i,j]-=h
        fd[i,j] = (forward(P,Xp)-forward(P,Xm))/(2*h)
err = np.abs(gX - fd).max()
rel = err / (np.abs(fd).max()+1e-9)
print("dL/dX  resident-vs-finite-diff  max_abs_err=%.3e  rel=%.3e" % (err, rel))
print("GRADCHECK:", "PASS" if rel < 1e-2 else "FAIL")

# ---------- training loop: confirm loss descends with trustworthy grads ----------
print("\n--- training (AdamW on all params) ---")
trainable = ['w1','w2','WG','WV','WD','Wg','Whead']
st = {k: [np.zeros_like(P[k]), np.zeros_like(P[k])] for k in trainable}
lr, b1, b2, eps = 0.03, 0.9, 0.999, 1e-8
def softmax2(z):
    z=z-z.max(-1,keepdims=True); e=np.exp(z); return e/e.sum(-1,keepdims=True)
print("step   loss   acc")
for step in range(1, 121):
    f = forward(P, X, full=True)
    loss = f['loss']
    if step==1 or step%15==0:
        acc = (softmax2(f['logits']).argmax(1)==targets).mean()
        print("%4d   %.4f   %.2f" % (step, loss, acc))
    g = resident_grads(P, X)
    for k in trainable:
        gk = g[k]
        st[k][0] = b1*st[k][0] + (1-b1)*gk
        st[k][1] = b2*st[k][1] + (1-b2)*(gk*gk)
        mh = st[k][0]/(1-b1**step); vh = st[k][1]/(1-b2**step)
        P[k] = P[k] - lr*(mh/(np.sqrt(vh)+eps))
fl = forward(P, X)
fa = (softmax2(forward(P,X,full=True)['logits']).argmax(1)==targets).mean()
print("\nfinal loss=%.4f acc=%.2f" % (fl, fa))
print("TRAIN:", "PASS" if fl < 0.3 and fa==1.0 else "FAIL")
