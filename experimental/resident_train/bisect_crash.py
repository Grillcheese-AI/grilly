import sys; sys.path.insert(0, r'C:\Users\grill\Documents\GitHub\grilly')
import numpy as np, grilly_core as gc
np.random.seed(0)
dev=gc.Device(); dev.load_shaders(r'C:\Users\grill\Documents\GitHub\grilly\shaders\spv')
def R(b,s,rg=True):
    r=gc.TensorRef(); r.buffer_id=b; r.set_shape(s); r.requires_grad=rg; return r
BL,Hd,C=72,24,4   # 12 batch * 6 seq = 72 rows like train_full (B=12 here for test)
b=12; L=6

# mimic grads(): forward in 2 cycles on tape t, then 3 separate backward tapes
def step():
    emb=np.random.randn(BL,Hd).astype(np.float32)
    w1=np.ones(Hd,np.float32); WG=np.random.randn(Hd,Hd).astype(np.float32)
    t=gc.TapeContext(dev); t.begin()
    e=t.register_input(emb,True); w1i=t.register_input(w1,True)
    WGi=t.register_input(WG,True); WVi=t.register_input(WG,True); WDi=t.register_input(WG,True)
    print("  fwd cycle1...", flush=True)
    t.forward_begin()
    n1=t.forward_rmsnorm(e,w1i,BL,Hd)
    gp=t.forward_linear(n1,WGi,0,BL,Hd,Hd)
    vp=t.forward_linear(n1,WVi,0,BL,Hd,Hd)
    dp=t.forward_linear(n1,WDi,0,BL,Hd,Hd)
    t.forward_submit()
    Gp=t.read_buffer(gp,[BL,Hd]); Vp=t.read_buffer(vp,[BL,Hd]); Dp=t.read_buffer(dp,[BL,Hd])
    print("  fwd cycle1 ok", flush=True)
    r1=t.register_input(emb,True); w2=t.register_input(np.ones(Hd,np.float32),True)
    Wg=t.register_input(np.random.randn(2*Hd,Hd).astype(np.float32),True)
    print("  fwd cycle2...", flush=True)
    t.forward_begin()
    n2=t.forward_rmsnorm(r1,w2,BL,Hd)
    gate=t.forward_linear(n2,Wg,0,BL,Hd,2*Hd)
    ff=t.forward_swiglu(gate,BL,Hd)
    t.forward_submit()
    gatev=t.read_buffer(gate,[BL,2*Hd]); ffv=t.read_buffer(ff,[BL,Hd])
    print("  fwd cycle2 ok", flush=True)
    # head backward on tape t
    Wh=t.register_input(np.random.randn(C,Hd).astype(np.float32),True)
    pooled=t.register_input(np.random.randn(b,Hd).astype(np.float32),True)
    logits=t.register_input(np.random.randn(b,C).astype(np.float32),True)
    lab=t.register_input(np.random.randint(0,C,b).astype(np.float32),False)
    nH=t.record_op(gc.OpType.Linear,[R(pooled,[b,Hd]),R(Wh,[C,Hd])],[R(logits,[b,C])]); t.save_for_backward(nH,[pooled,Wh])
    nCE=t.record_op(gc.OpType.CrossEntropy,[R(logits,[b,C])],[R(0,[1],False)]); t.save_for_backward(nCE,[logits,lab])
    print("  head backward...", flush=True)
    t.backward(nCE,0)
    print("  head backward ok", flush=True)

for i in range(3):
    print("STEP", i, flush=True)
    step()
print("ALL DONE", flush=True)
