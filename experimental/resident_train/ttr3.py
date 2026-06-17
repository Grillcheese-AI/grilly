"""Minimal repro of the resident-FORWARD 0xC0000005 crash (see AUTOGRAD_STATE.md
"RESIDENT-FORWARD CRASH"): resident forward (rmsnorm+linear, NO read-back) then a
backward -> access violation. Adding a read_buffer of the forward output makes it
vanish (and exposes the grads-come-back-0.0 correctness bug). Exit -1073741819."""
import sys; sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np, grilly_core as gc
np.random.seed(0)
dev=gc.Device(); dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")
def R(b,s,rg=True):
    r=gc.TensorRef(); r.buffer_id=b; r.set_shape(s); r.requires_grad=rg; return r
BL,Hd,C=192,24,4
def lin_ce_backward(tape):
    X=np.random.randn(BL,Hd).astype(np.float32); W=(0.2*np.random.randn(C,Hd)).astype(np.float32)
    tgt=np.random.randint(0,C,BL).astype(np.float32)
    tape.begin()
    x=tape.register_input(X,True); w=tape.register_input(W,True)
    lg=tape.register_input(np.zeros((BL,C),np.float32),True); tg=tape.register_input(tgt,False)
    nl=tape.record_op(gc.OpType.Linear,[R(x,[BL,Hd]),R(w,[C,Hd])],[R(lg,[BL,C])]); tape.save_for_backward(nl,[x,w])
    nce=tape.record_op(gc.OpType.CrossEntropy,[R(lg,[BL,C])],[R(0,[1],False)]); tape.save_for_backward(nce,[lg,tg])
    tape.backward(nce,0)
    return float(np.abs(tape.read_buffer(tape.get_grad_buffer(w),[C,Hd])).max())

print("PATTERN: t does FORWARD cycles + backward, then t2 does backward", flush=True)
t = gc.TapeContext(dev); t.begin()
emb=np.random.randn(BL,Hd).astype(np.float32); w1=np.ones(Hd,np.float32); WG=(0.2*np.random.randn(Hd,Hd)).astype(np.float32)
e=t.register_input(emb,True); w1i=t.register_input(w1,True); WGi=t.register_input(WG,True)
t.forward_begin()
n1=t.forward_rmsnorm(e,w1i,BL,Hd)
gp=t.forward_linear(n1,WGi,0,BL,Hd,Hd)
t.forward_submit()
print("  t forward ok", flush=True)
# now t does a head backward like grads
Wh=t.register_input((0.2*np.random.randn(C,Hd)).astype(np.float32),True)
pl=t.register_input(np.random.randn(BL,Hd).astype(np.float32),True)
lg=t.register_input(np.zeros((BL,C),np.float32),True); tg=t.register_input(np.random.randint(0,C,BL).astype(np.float32),False)
nH=t.record_op(gc.OpType.Linear,[R(pl,[BL,Hd]),R(Wh,[C,Hd])],[R(lg,[BL,C])]); t.save_for_backward(nH,[pl,Wh])
nce=t.record_op(gc.OpType.CrossEntropy,[R(lg,[BL,C])],[R(0,[1],False)]); t.save_for_backward(nce,[lg,tg])
t.backward(nce,0)
print("  t head backward ok", flush=True)
print("  now t2 backward while t alive...", flush=True)
t2 = gc.TapeContext(dev)
print("  t2 bw ->", lin_ce_backward(t2), flush=True)
print("DONE", flush=True)
