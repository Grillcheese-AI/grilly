"""FULL TRAINING TEST [RESIDENT FORWARD variant] — resident autograd, multi-step, train + held-out eval.

Task: sum-quartile classification. Each sample is a length-L sequence of tokens
in [0,V); the label is the quartile bucket of the token sum. A real learnable,
generalizable aggregation: the model must sum across the sequence and threshold.
Trained ONLINE (fresh batch each step) so falling loss == generalization, not
memorization. train_x/test_x are held-out eval sets the model never trains on.

Model (one Cubby-style block, per-position hidden Hd, pooled over sequence):
  emb     = E[tokens]                         (B, L, Hd)   embedding lookup (numpy)
  for each position independently then pooled:
  n1      = RMSNorm(emb, w1)                   RESIDENT
  Gp,Vp,Dp= n1 @ {WG,WV,WD}^T                  RESIDENT (linear)
  H        = MinGRU(Gp,Vp,Dp) over L           (numpy forward; resident backward)
  r1      = emb + H                            residual
  n2      = RMSNorm(r1, w2)                     RESIDENT
  gate    = n2 @ Wg^T   (2Hd)                  RESIDENT
  ff      = SwiGLU(gate)                        RESIDENT
  r2      = r1 + ff                            residual
  pooled  = mean_L r2                           (B, Hd)
  logits  = pooled @ Whead^T                    (B, C)
  loss    = CE(logits, labels)

Backward is fully resident through every op. Forward is resident for
Linear/RMSNorm/SwiGLU; MinGRU forward + embedding + pooling are numpy (their
backward is resident / handled). Trains with AdamW; reports train + test acc.
"""
import sys
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np
import grilly_core as gc

np.random.seed(0)
V, L, C, Hd = 12, 6, 4, 24        # vocab, seq len, classes, hidden
N_TRAIN, N_TEST, B = 256, 128, 32
EPS = 1e-6

# Monotonic aggregation label: bucket the token-sum into C balanced classes.
# (sum-mod-C was adversarial to smooth approximators; a threshold on the sum is
# genuinely learnable by a mean-pooling sequence model and generalizes.)
_big = np.random.randint(0, V, size=(100000, L)).sum(1)
_bounds = np.quantile(_big, [(i+1)/C for i in range(C-1)])
def sum_to_label(toks):
    return np.digitize(toks.sum(1), _bounds).astype(np.uint32)
def make_data(n):
    toks = np.random.randint(0, V, size=(n, L)).astype(np.int64)
    return toks, sum_to_label(toks)
train_x, train_y = make_data(N_TRAIN)   # fixed sets used only for eval reporting
test_x,  test_y  = make_data(N_TEST)

def randn(*s, sc): return (sc*np.random.randn(*s)).astype(np.float32)
P = dict(
    E=randn(V, Hd, sc=0.1),
    w1=np.ones(Hd, np.float32), w2=np.ones(Hd, np.float32),
    WG=randn(Hd, Hd, sc=0.2), WV=randn(Hd, Hd, sc=0.2), WD=randn(Hd, Hd, sc=0.2),
    Wg=randn(2*Hd, Hd, sc=0.2), Whead=randn(C, Hd, sc=0.2),
)

def sig(z): return 1.0/(1.0+np.exp(-z))
def softmax(z):
    z=z-z.max(-1,keepdims=True); e=np.exp(z); return e/e.sum(-1,keepdims=True)

def mingru_fwd(G, V_, D):   # (B,L,Hd)
    sg=sig(G); tv=np.tanh(V_); sd=sig(D); xs=sg*tv; a=0.001+0.998*sd
    H=np.zeros_like(G)
    for t in range(L):
        prev=H[:,t-1,:] if t>0 else 0.0
        H[:,t,:]=a[:,t,:]*prev+xs[:,t,:]
    return H

dev = gc.Device(); dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")
def R(b,s,rg=True):
    r=gc.TensorRef(); r.buffer_id=b; r.set_shape(s); r.requires_grad=rg; return r

def forward_np(P, toks):
    """Pure-numpy forward for eval/loss (mirrors the resident graph exactly)."""
    b=len(toks); emb=P['E'][toks]                  # (b,L,Hd)
    flat=emb.reshape(b*L,Hd)
    ms=(flat*flat).mean(1,keepdims=True); n1=flat*P['w1']/np.sqrt(ms+EPS)
    Gp=n1@P['WG'].T; Vp=n1@P['WV'].T; Dp=n1@P['WD'].T
    H=mingru_fwd(Gp.reshape(b,L,Hd),Vp.reshape(b,L,Hd),Dp.reshape(b,L,Hd))
    r1=emb+H
    r1f=r1.reshape(b*L,Hd)
    ms2=(r1f*r1f).mean(1,keepdims=True); n2=r1f*P['w2']/np.sqrt(ms2+EPS)
    gate=n2@P['Wg'].T; x1=gate[:,:Hd]; x2=gate[:,Hd:]; ff=(x1*(x2*sig(x2))).reshape(b,L,Hd)
    r2=r1+ff
    pooled=r2.mean(1)
    logits=pooled@P['Whead'].T
    return logits

def loss_acc(P, toks, labels):
    lg=forward_np(P,toks); sm=softmax(lg)
    loss=-np.log(sm[np.arange(len(labels)),labels]+1e-12).mean()
    acc=(sm.argmax(1)==labels).mean()
    return loss, acc

def grads(P, toks, labels):
    """Resident-forward (where available) + resident-backward over a batch.
    Returns grads for all params."""
    b=len(toks); BL=b*L
    emb=P['E'][toks]                                # (b,L,Hd) numpy lookup
    embf=emb.reshape(BL,Hd).astype(np.float32)

    # RESIDENT FORWARD (validated behind VMA_DEBUG_MARGIN). Forward is computed
    # on-GPU via forward_rmsnorm/linear/swiglu, read back, and re-registered to
    # seed the backward tape. Same t/t2/t3 backward structure as the numpy path.
    t=gc.TapeContext(dev); t.begin()
    emb_id=t.register_input(embf,True)
    w1=t.register_input(P['w1'],True); w2=t.register_input(P['w2'],True)
    WG=t.register_input(P['WG'],True); WV=t.register_input(P['WV'],True); WD=t.register_input(P['WD'],True)
    Wg=t.register_input(P['Wg'],True); Wh=t.register_input(P['Whead'],True)
    t.forward_begin()
    n1_id=t.forward_rmsnorm(emb_id,w1,BL,Hd)
    Gp_id=t.forward_linear(n1_id,WG,0,BL,Hd,Hd)
    Vp_id=t.forward_linear(n1_id,WV,0,BL,Hd,Hd)
    Dp_id=t.forward_linear(n1_id,WD,0,BL,Hd,Hd)
    t.forward_submit()
    Gp=t.read_buffer(Gp_id,[BL,Hd]); Vp=t.read_buffer(Vp_id,[BL,Hd]); Dp=t.read_buffer(Dp_id,[BL,Hd])
    H=mingru_fwd(Gp.reshape(b,L,Hd),Vp.reshape(b,L,Hd),Dp.reshape(b,L,Hd))
    Hf=H.reshape(BL,Hd).astype(np.float32)
    r1=(emb+H); r1f=r1.reshape(BL,Hd).astype(np.float32)
    r1_id=t.register_input(r1f,True)
    t.forward_begin()
    n2_id=t.forward_rmsnorm(r1_id,w2,BL,Hd)
    gate_id=t.forward_linear(n2_id,Wg,0,BL,Hd,2*Hd)
    ff_id=t.forward_swiglu(gate_id,BL,Hd)
    t.forward_submit()
    gate=t.read_buffer(gate_id,[BL,2*Hd]); ff=t.read_buffer(ff_id,[BL,Hd])
    r2=(r1+ff.reshape(b,L,Hd)); pooled=r2.mean(1).astype(np.float32)
    logits=(pooled@P['Whead'].T).astype(np.float32)
    n1r=t.read_buffer(n1_id,[BL,Hd]); n2r=t.read_buffer(n2_id,[BL,Hd])
    pooled_id=t.register_input(pooled,True); logits_id=t.register_input(logits,True)
    labf=labels.astype(np.float32); lab_id=t.register_input(labf,False)
    nHead=t.record_op(gc.OpType.Linear,[R(pooled_id,[b,Hd]),R(Wh,[C,Hd])],[R(logits_id,[b,C])]); t.save_for_backward(nHead,[pooled_id,Wh])
    nCE=t.record_op(gc.OpType.CrossEntropy,[R(logits_id,[b,C])],[R(0,[1],False)]); t.save_for_backward(nCE,[logits_id,lab_id])
    t.backward(nCE,0)
    gWhead=t.read_buffer(t.get_grad_buffer(Wh),[C,Hd])/b
    g_pooled=t.read_buffer(t.get_grad_buffer(pooled_id),[b,Hd])/b   # dL/dpooled

    # mean-pool backward: dL/dr2[:,l,:] = g_pooled / L
    gradR2=np.repeat((g_pooled/L)[:,None,:],L,axis=1).astype(np.float32)  # (b,L,Hd)
    gR2f=gradR2.reshape(BL,Hd)

    # r2 = r1 + ff  -> grad to r1 (path A) and ff (path B)
    # ff branch: SwiGLU(gate) <- gate=Linear(n2,Wg) <- n2=RMSNorm(r1,w2)
    t2=gc.TapeContext(dev); t2.begin()
    gr2_id=t2.register_input(gR2f,False)
    r1b=t2.register_input(r1f,True); w2b=t2.register_input(P['w2'],True); Wgb=t2.register_input(P['Wg'],True)
    n2b2=t2.register_input(n2r,True); gateb2=t2.register_input(gate.astype(np.float32),True); ffb2=t2.register_input(ff.astype(np.float32),True)
    nrms2=t2.record_op(gc.OpType.RMSNorm,[R(r1b,[BL,Hd]),R(w2b,[Hd])],[R(n2b2,[BL,Hd])]); t2.save_for_backward(nrms2,[r1b,w2b])
    ngate=t2.record_op(gc.OpType.Linear,[R(n2b2,[BL,Hd]),R(Wgb,[2*Hd,Hd])],[R(gateb2,[BL,2*Hd])]); t2.save_for_backward(ngate,[n2b2,Wgb])
    nsw=t2.record_op(gc.OpType.SwiGLU,[R(gateb2,[BL,2*Hd])],[R(ffb2,[BL,Hd])]); t2.save_for_backward(nsw,[gateb2])
    t2.backward(nsw, gr2_id)   # seed dL/dff = dL/dr2 (since r2=r1+ff)
    gw2=t2.read_buffer(t2.get_grad_buffer(w2b),[Hd])
    gWg=t2.read_buffer(t2.get_grad_buffer(Wgb),[2*Hd,Hd])
    gR1_ff=t2.read_buffer(t2.get_grad_buffer(r1b),[BL,Hd])  # grad into r1 via ff branch

    # total grad into r1 = gR2 (direct) + gR1_ff
    gR1f=gR2f + gR1_ff
    # r1 = emb + H  -> grad to emb (direct) and H
    # H branch: MinGRU(Gp,Vp,Dp) <- Gp/Vp/Dp = Linear(n1,W) <- n1=RMSNorm(emb,w1)
    t3=gc.TapeContext(dev); t3.begin()
    gH_id=t3.register_input(gR1f,False)   # dL/dH = grad into r1 (r1=emb+H)
    embb=t3.register_input(embf,True); w1b=t3.register_input(P['w1'],True)
    WGb=t3.register_input(P['WG'],True); WVb=t3.register_input(P['WV'],True); WDb=t3.register_input(P['WD'],True)
    n1b3=t3.register_input(n1r,True)
    Gpb=t3.register_input(Gp.astype(np.float32),True); Vpb=t3.register_input(Vp.astype(np.float32),True); Dpb=t3.register_input(Dp.astype(np.float32),True)
    Hb=t3.register_input(Hf,True)
    nrms1=t3.record_op(gc.OpType.RMSNorm,[R(embb,[BL,Hd]),R(w1b,[Hd])],[R(n1b3,[BL,Hd])]); t3.save_for_backward(nrms1,[embb,w1b])
    nG=t3.record_op(gc.OpType.Linear,[R(n1b3,[BL,Hd]),R(WGb,[Hd,Hd])],[R(Gpb,[BL,Hd])]); t3.save_for_backward(nG,[n1b3,WGb])
    nV=t3.record_op(gc.OpType.Linear,[R(n1b3,[BL,Hd]),R(WVb,[Hd,Hd])],[R(Vpb,[BL,Hd])]); t3.save_for_backward(nV,[n1b3,WVb])
    nD=t3.record_op(gc.OpType.Linear,[R(n1b3,[BL,Hd]),R(WDb,[Hd,Hd])],[R(Dpb,[BL,Hd])]); t3.save_for_backward(nD,[n1b3,WDb])
    nM=t3.record_op(gc.OpType.MinGRU,[R(Gpb,[b,L,Hd]),R(Vpb,[b,L,Hd]),R(Dpb,[b,L,Hd])],[R(Hb,[b,L,Hd])]); t3.save_for_backward(nM,[Gpb,Vpb,Dpb,Hb])
    t3.backward(nM, gH_id)
    gw1=t3.read_buffer(t3.get_grad_buffer(w1b),[Hd])
    gWG=t3.read_buffer(t3.get_grad_buffer(WGb),[Hd,Hd])
    gWV=t3.read_buffer(t3.get_grad_buffer(WVb),[Hd,Hd])
    gWD=t3.read_buffer(t3.get_grad_buffer(WDb),[Hd,Hd])
    gEmb_n1=t3.read_buffer(t3.get_grad_buffer(embb),[BL,Hd])   # grad into emb via n1 branch

    # total grad into emb = gR1 (direct, r1=emb+H) + gEmb_n1
    gEmbf=(gR1f + gEmb_n1).reshape(b,L,Hd)
    # embedding backward: scatter-add grad onto E rows
    gE=np.zeros_like(P['E'])
    np.add.at(gE, toks.reshape(-1), gEmbf.reshape(b*L,Hd))

    return dict(E=gE, w1=gw1, w2=gw2, WG=gWG, WV=gWV, WD=gWD, Wg=gWg, Whead=gWhead)

# ---------- AdamW training ----------
trainable=['E','w1','w2','WG','WV','WD','Wg','Whead']
st={k:[np.zeros_like(P[k]),np.zeros_like(P[k])] for k in trainable}
lr,b1,b2,eps,wd=0.01,0.9,0.999,1e-8,0.001
def adamw(k,g,step):
    st[k][0]=b1*st[k][0]+(1-b1)*g; st[k][1]=b2*st[k][1]+(1-b2)*(g*g)
    mh=st[k][0]/(1-b1**step); vh=st[k][1]/(1-b2**step)
    P[k]=P[k]-lr*(mh/(np.sqrt(vh)+eps)+wd*P[k])

# ---------- gradient check (validate stitched backward before trusting training) ----------
if 'gradcheck' in sys.argv:
    print("=== GRADCHECK: stitched resident backward vs finite differences ===")
    bx, by = train_x[:8], train_y[:8]
    g = grads(P, bx, by)
    def fd_param(name, h=1e-3):
        base = P[name].copy(); fd = np.zeros_like(base)
        it = np.nditer(base, flags=['multi_index'])
        while not it.finished:
            i = it.multi_index
            P[name][i] = base[i] + h; lp, _ = loss_acc(P, bx, by)
            P[name][i] = base[i] - h; lm, _ = loss_acc(P, bx, by)
            P[name][i] = base[i]
            fd[i] = (lp - lm) / (2*h); it.iternext()
        return fd
    all_ok = True
    for nm in ['Whead', 'Wg', 'WG', 'WV', 'WD', 'w1', 'w2', 'E']:
        fd = fd_param(nm)
        e = np.abs(g[nm] - fd).max(); rel = e / (np.abs(fd).max() + 1e-9)
        ok = rel < 3e-2
        all_ok = all_ok and ok
        print("  %-7s rel_err=%.3e  %s" % (nm, rel, "PASS" if ok else "FAIL"))
    print("GRADCHECK:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)

print("=== FULL TRAINING TEST: sum-quartile-class (online), V=%d L=%d C=%d Hd=%d ===" % (V,L,C,Hd))
print("step  train_loss  train_acc  test_acc")
n_steps=400
gstep=0
for step in range(1, n_steps+1):
    bx,by=make_data(B)          # FRESH data each step: never seen twice
    gstep+=1
    g=grads(P,bx,by)
    for k in trainable: adamw(k,g[k],gstep)
    if step==1 or step%40==0:
        trl,tra=loss_acc(P,train_x,train_y)
        _,tea=loss_acc(P,test_x,test_y)
        print("%4d   %.4f      %.3f      %.3f" % (step,trl,tra,tea))

trl,tra=loss_acc(P,train_x,train_y); _,tea=loss_acc(P,test_x,test_y)
print("\nFINAL  train_loss=%.4f train_acc=%.3f test_acc=%.3f" % (trl,tra,tea))
# chance = 1/C = 0.25; success = clearly above chance and generalizing
print("RESULT:", "PASS" if tea > 0.5 and trl < 1.0 else "FAIL")
