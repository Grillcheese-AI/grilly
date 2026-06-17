"""Bisect the full-block gradient-check failure by testing prefixes."""
import sys
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np
import grilly_core as gc

np.random.seed(7)
B, Hd, C = 4, 8, 4
BS = B*1; S = 1
EPS = 1e-6
def randn(*s, sc=0.2): return (sc*np.random.randn(*s)).astype(np.float32)
P = dict(w1=np.ones(Hd,np.float32), WG=randn(Hd,Hd), WV=randn(Hd,Hd), WD=randn(Hd,Hd),
         Whead=randn(C,Hd))
X = randn(BS,Hd, sc=0.5)
targets = np.random.randint(0,C,size=B).astype(np.uint32)
targets_f = targets.astype(np.float32)
def sig(z): return 1.0/(1.0+np.exp(-z))
def softmax(z):
    z=z-z.max(-1,keepdims=True); e=np.exp(z); return e/e.sum(-1,keepdims=True)
def rmsnorm(x,w):
    ms=(x*x).mean(-1,keepdims=True); r=1.0/np.sqrt(ms+EPS); return x*r*w
def ce(logits): return -np.log(softmax(logits)[np.arange(B),targets]+1e-12).mean()

dev=gc.Device(); dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")
def R(b,s,rg=True):
    r=gc.TensorRef(); r.buffer_id=b; r.set_shape(s); r.requires_grad=rg; return r

# ---- TEST A: RMSNorm -> Linear(head) -> CE ----
def fwdA(P,X,full=False):
    n1=rmsnorm(X,P['w1']); logits=n1@P['Whead'].T; loss=ce(logits)
    return (dict(n1=n1,logits=logits),loss) if full else loss
def gradA_X(P,X):
    f,_=fwdA(P,X,True)
    t=gc.TapeContext(dev); t.begin()
    X_id=t.register_input(X,True); w1_id=t.register_input(P['w1'],True)
    Wh_id=t.register_input(P['Whead'],True); tgt=t.register_input(targets_f,False)
    n1_id=t.register_input(f['n1'],True); lg_id=t.register_input(f['logits'],True)
    nn1=t.record_op(gc.OpType.RMSNorm,[R(X_id,[BS,Hd]),R(w1_id,[Hd])],[R(n1_id,[BS,Hd])]); t.save_for_backward(nn1,[X_id,w1_id])
    nh=t.record_op(gc.OpType.Linear,[R(n1_id,[BS,Hd]),R(Wh_id,[C,Hd])],[R(lg_id,[B,C])]); t.save_for_backward(nh,[n1_id,Wh_id])
    nce=t.record_op(gc.OpType.CrossEntropy,[R(lg_id,[B,C])],[R(0,[1],False)]); t.save_for_backward(nce,[lg_id,tgt])
    t.backward(nce,0)
    return t.read_buffer(t.get_grad_buffer(X_id),[BS,Hd])/B

def fdcheck(name, fwd, grad_fn, P, X):
    gX=grad_fn(P,X); h=1e-3; fd=np.zeros_like(X)
    for i in range(BS):
        for j in range(Hd):
            Xp=X.copy(); Xp[i,j]+=h; Xm=X.copy(); Xm[i,j]-=h
            fd[i,j]=(fwd(P,Xp)-fwd(P,Xm))/(2*h)
    e=np.abs(gX-fd).max(); rel=e/(np.abs(fd).max()+1e-9)
    print("%-28s max_abs_err=%.3e rel=%.3e  %s" % (name, e, rel, "PASS" if rel<1e-2 else "FAIL"))
    return rel<1e-2

fdcheck("A: RMSNorm->Lin->CE", fwdA, gradA_X, P, X)

# ---- TEST B: add the X fan-out residual: r1 = X + Linear(RMSNorm(X)) ----
# n1=RMSNorm(X); h=n1@WG^T; r1=X+h; logits=r1@Whead^T; CE
def fwdB(P,X,full=False):
    n1=rmsnorm(X,P['w1']); h=n1@P['WG'].T; r1=X+h; logits=r1@P['Whead'].T; loss=ce(logits)
    return (dict(n1=n1,h=h,r1=r1,logits=logits),loss) if full else loss
def gradB_X(P,X):
    f,_=fwdB(P,X,True)
    t=gc.TapeContext(dev); t.begin()
    X_id=t.register_input(X,True); w1_id=t.register_input(P['w1'],True)
    WG_id=t.register_input(P['WG'],True); Wh_id=t.register_input(P['Whead'],True); tgt=t.register_input(targets_f,False)
    n1_id=t.register_input(f['n1'],True); h_id=t.register_input(f['h'],True); r1_id=t.register_input(f['r1'],True); lg_id=t.register_input(f['logits'],True)
    nn1=t.record_op(gc.OpType.RMSNorm,[R(X_id,[BS,Hd]),R(w1_id,[Hd])],[R(n1_id,[BS,Hd])]); t.save_for_backward(nn1,[X_id,w1_id])
    ng=t.record_op(gc.OpType.Linear,[R(n1_id,[BS,Hd]),R(WG_id,[Hd,Hd])],[R(h_id,[BS,Hd])]); t.save_for_backward(ng,[n1_id,WG_id])
    na=t.record_op(gc.OpType.Add,[R(X_id,[BS,Hd]),R(h_id,[BS,Hd])],[R(r1_id,[BS,Hd])])
    nh=t.record_op(gc.OpType.Linear,[R(r1_id,[B,Hd]),R(Wh_id,[C,Hd])],[R(lg_id,[B,C])]); t.save_for_backward(nh,[r1_id,Wh_id])
    nce=t.record_op(gc.OpType.CrossEntropy,[R(lg_id,[B,C])],[R(0,[1],False)]); t.save_for_backward(nce,[lg_id,tgt])
    t.backward(nce,0)
    return t.read_buffer(t.get_grad_buffer(X_id),[BS,Hd])/B

fdcheck("B: X-fanout residual", fwdB, gradB_X, P, X)

# ---- TEST C: MinGRU in the middle (S=1) ----
def mingru_flat(G,V,D):
    sg=sig(G); tv=np.tanh(V); sd=sig(D); return sg*tv
def fwdC(P,X,full=False):
    n1=rmsnorm(X,P['w1']); Gp=n1@P['WG'].T; Vp=n1@P['WV'].T; Dp=n1@P['WD'].T
    H=mingru_flat(Gp,Vp,Dp); logits=H@P['Whead'].T; loss=ce(logits)
    return (dict(n1=n1,Gp=Gp,Vp=Vp,Dp=Dp,H=H,logits=logits),loss) if full else loss
def gradC_X(P,X):
    f,_=fwdC(P,X,True)
    t=gc.TapeContext(dev); t.begin()
    X_id=t.register_input(X,True); w1_id=t.register_input(P['w1'],True)
    WG_id=t.register_input(P['WG'],True); WV_id=t.register_input(P['WV'],True); WD_id=t.register_input(P['WD'],True)
    Wh_id=t.register_input(P['Whead'],True); tgt=t.register_input(targets_f,False)
    n1_id=t.register_input(f['n1'],True); Gp_id=t.register_input(f['Gp'],True); Vp_id=t.register_input(f['Vp'],True); Dp_id=t.register_input(f['Dp'],True)
    H_id=t.register_input(f['H'],True); lg_id=t.register_input(f['logits'],True)
    nn1=t.record_op(gc.OpType.RMSNorm,[R(X_id,[BS,Hd]),R(w1_id,[Hd])],[R(n1_id,[BS,Hd])]); t.save_for_backward(nn1,[X_id,w1_id])
    nG=t.record_op(gc.OpType.Linear,[R(n1_id,[BS,Hd]),R(WG_id,[Hd,Hd])],[R(Gp_id,[BS,Hd])]); t.save_for_backward(nG,[n1_id,WG_id])
    nV=t.record_op(gc.OpType.Linear,[R(n1_id,[BS,Hd]),R(WV_id,[Hd,Hd])],[R(Vp_id,[BS,Hd])]); t.save_for_backward(nV,[n1_id,WV_id])
    nD=t.record_op(gc.OpType.Linear,[R(n1_id,[BS,Hd]),R(WD_id,[Hd,Hd])],[R(Dp_id,[BS,Hd])]); t.save_for_backward(nD,[n1_id,WD_id])
    nM=t.record_op(gc.OpType.MinGRU,[R(Gp_id,[B,S,Hd]),R(Vp_id,[B,S,Hd]),R(Dp_id,[B,S,Hd])],[R(H_id,[B,S,Hd])]); t.save_for_backward(nM,[Gp_id,Vp_id,Dp_id,H_id])
    nh=t.record_op(gc.OpType.Linear,[R(H_id,[B,Hd]),R(Wh_id,[C,Hd])],[R(lg_id,[B,C])]); t.save_for_backward(nh,[H_id,Wh_id])
    nce=t.record_op(gc.OpType.CrossEntropy,[R(lg_id,[B,C])],[R(0,[1],False)]); t.save_for_backward(nce,[lg_id,tgt])
    t.backward(nce,0)
    return t.read_buffer(t.get_grad_buffer(X_id),[BS,Hd])/B

fdcheck("C: +MinGRU (n1->3Lin->MinGRU)", fwdC, gradC_X, P, X)

# ---- TEST D: 3-way fan-out WITHOUT MinGRU: n1->3 Linears->Add(Add)->head ----
def fwdD(P,X,full=False):
    n1=rmsnorm(X,P['w1']); Gp=n1@P['WG'].T; Vp=n1@P['WV'].T; Dp=n1@P['WD'].T
    s=Gp+Vp+Dp; logits=s@P['Whead'].T; loss=ce(logits)
    return (dict(n1=n1,Gp=Gp,Vp=Vp,Dp=Dp,s1=Gp+Vp,s=s,logits=logits),loss) if full else loss
def gradD_X(P,X):
    f,_=fwdD(P,X,True)
    t=gc.TapeContext(dev); t.begin()
    X_id=t.register_input(X,True); w1_id=t.register_input(P['w1'],True)
    WG_id=t.register_input(P['WG'],True); WV_id=t.register_input(P['WV'],True); WD_id=t.register_input(P['WD'],True)
    Wh_id=t.register_input(P['Whead'],True); tgt=t.register_input(targets_f,False)
    n1_id=t.register_input(f['n1'],True); Gp_id=t.register_input(f['Gp'],True); Vp_id=t.register_input(f['Vp'],True); Dp_id=t.register_input(f['Dp'],True)
    s1_id=t.register_input(f['s1'],True); s_id=t.register_input(f['s'],True); lg_id=t.register_input(f['logits'],True)
    nn1=t.record_op(gc.OpType.RMSNorm,[R(X_id,[BS,Hd]),R(w1_id,[Hd])],[R(n1_id,[BS,Hd])]); t.save_for_backward(nn1,[X_id,w1_id])
    nG=t.record_op(gc.OpType.Linear,[R(n1_id,[BS,Hd]),R(WG_id,[Hd,Hd])],[R(Gp_id,[BS,Hd])]); t.save_for_backward(nG,[n1_id,WG_id])
    nV=t.record_op(gc.OpType.Linear,[R(n1_id,[BS,Hd]),R(WV_id,[Hd,Hd])],[R(Vp_id,[BS,Hd])]); t.save_for_backward(nV,[n1_id,WV_id])
    nD=t.record_op(gc.OpType.Linear,[R(n1_id,[BS,Hd]),R(WD_id,[Hd,Hd])],[R(Dp_id,[BS,Hd])]); t.save_for_backward(nD,[n1_id,WD_id])
    na1=t.record_op(gc.OpType.Add,[R(Gp_id,[BS,Hd]),R(Vp_id,[BS,Hd])],[R(s1_id,[BS,Hd])])
    na2=t.record_op(gc.OpType.Add,[R(s1_id,[BS,Hd]),R(Dp_id,[BS,Hd])],[R(s_id,[BS,Hd])])
    nh=t.record_op(gc.OpType.Linear,[R(s_id,[B,Hd]),R(Wh_id,[C,Hd])],[R(lg_id,[B,C])]); t.save_for_backward(nh,[s_id,Wh_id])
    nce=t.record_op(gc.OpType.CrossEntropy,[R(lg_id,[B,C])],[R(0,[1],False)]); t.save_for_backward(nce,[lg_id,tgt])
    t.backward(nce,0)
    return t.read_buffer(t.get_grad_buffer(X_id),[BS,Hd])/B

fdcheck("D: 3-way fanout no MinGRU", fwdD, gradD_X, P, X)

# ---- TEST E: MinGRU alone (no upstream fan-out): X->MinGRU->head ----
def fwdE(P,X,full=False):
    H=mingru_flat(X,X,X)   # reuse X as all three projections (isolates MinGRU grad merge)
    logits=H@P['Whead'].T; loss=ce(logits)
    return (dict(H=H,logits=logits),loss) if full else loss
def gradE_X(P,X):
    f,_=fwdE(P,X,True)
    t=gc.TapeContext(dev); t.begin()
    X_id=t.register_input(X,True); Wh_id=t.register_input(P['Whead'],True); tgt=t.register_input(targets_f,False)
    H_id=t.register_input(f['H'],True); lg_id=t.register_input(f['logits'],True)
    # G,V,D all = X  -> X fans out into all 3 MinGRU inputs
    nM=t.record_op(gc.OpType.MinGRU,[R(X_id,[B,S,Hd]),R(X_id,[B,S,Hd]),R(X_id,[B,S,Hd])],[R(H_id,[B,S,Hd])]); t.save_for_backward(nM,[X_id,X_id,X_id,H_id])
    nh=t.record_op(gc.OpType.Linear,[R(H_id,[B,Hd]),R(Wh_id,[C,Hd])],[R(lg_id,[B,C])]); t.save_for_backward(nh,[H_id,Wh_id])
    nce=t.record_op(gc.OpType.CrossEntropy,[R(lg_id,[B,C])],[R(0,[1],False)]); t.save_for_backward(nce,[lg_id,tgt])
    t.backward(nce,0)
    return t.read_buffer(t.get_grad_buffer(X_id),[BS,Hd])/B

fdcheck("E: MinGRU X->all3 (self-fanout)", fwdE, gradE_X, P, X)

