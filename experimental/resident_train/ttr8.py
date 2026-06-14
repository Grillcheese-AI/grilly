"""Proof the multi-tape t/t2/t3 BACKWARD structure is clean with numpy-forward seeds:
3 full iterations, exit 0. Isolates the crash to the resident FORWARD ops, not the
backward engine. This is the structure train_full.py grads() uses. See AUTOGRAD_STATE.md."""
import sys; sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np, grilly_core as gc
np.random.seed(0)
dev=gc.Device(); dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")
def R(b,s,rg=True):
    r=gc.TensorRef(); r.buffer_id=b; r.set_shape(s); r.requires_grad=rg; return r
b,L,Hd,C=32,6,24,4; BL=b*L
def rnd(*s): return (0.2*np.random.randn(*s)).astype(np.float32)
# 3 separate backward tapes, NO resident forward (pure numpy seeds) -- mimics grads() if forward were numpy
for it in range(3):
    print("ITER", it, flush=True)
    # tape t: head Linear+CE
    t=gc.TapeContext(dev); t.begin()
    Wh=t.register_input(rnd(C,Hd),True); pl=t.register_input(rnd(b,Hd),True)
    lg=t.register_input(np.zeros((b,C),np.float32),True); tg=t.register_input(np.random.randint(0,C,b).astype(np.float32),False)
    nH=t.record_op(gc.OpType.Linear,[R(pl,[b,Hd]),R(Wh,[C,Hd])],[R(lg,[b,C])]); t.save_for_backward(nH,[pl,Wh])
    nce=t.record_op(gc.OpType.CrossEntropy,[R(lg,[b,C])],[R(0,[1],False)]); t.save_for_backward(nce,[lg,tg])
    t.backward(nce,0); _=t.read_buffer(t.get_grad_buffer(Wh),[C,Hd])
    print("  t ok", flush=True)
    # tape t2: RMSNorm->Linear->SwiGLU
    t2=gc.TapeContext(dev); t2.begin()
    gr2=t2.register_input(rnd(BL,Hd),False)
    r1b=t2.register_input(rnd(BL,Hd),True); w2b=t2.register_input(np.ones(Hd,np.float32),True); Wgb=t2.register_input(rnd(2*Hd,Hd),True)
    n2b=t2.register_input(rnd(BL,Hd),True); gateb=t2.register_input(rnd(BL,2*Hd),True); ffb=t2.register_input(rnd(BL,Hd),True)
    nr=t2.record_op(gc.OpType.RMSNorm,[R(r1b,[BL,Hd]),R(w2b,[Hd])],[R(n2b,[BL,Hd])]); t2.save_for_backward(nr,[r1b,w2b])
    ng=t2.record_op(gc.OpType.Linear,[R(n2b,[BL,Hd]),R(Wgb,[2*Hd,Hd])],[R(gateb,[BL,2*Hd])]); t2.save_for_backward(ng,[n2b,Wgb])
    ns=t2.record_op(gc.OpType.SwiGLU,[R(gateb,[BL,2*Hd])],[R(ffb,[BL,Hd])]); t2.save_for_backward(ns,[gateb])
    t2.backward(ns,gr2); _=t2.read_buffer(t2.get_grad_buffer(r1b),[BL,Hd])
    print("  t2 ok", flush=True)
    # tape t3: RMSNorm->3Linear->MinGRU
    t3=gc.TapeContext(dev); t3.begin()
    gH=t3.register_input(rnd(BL,Hd),False)
    embb=t3.register_input(rnd(BL,Hd),True); w1b=t3.register_input(np.ones(Hd,np.float32),True)
    WGb=t3.register_input(rnd(Hd,Hd),True); WVb=t3.register_input(rnd(Hd,Hd),True); WDb=t3.register_input(rnd(Hd,Hd),True)
    n1b=t3.register_input(rnd(BL,Hd),True)
    Gpb=t3.register_input(rnd(BL,Hd),True); Vpb=t3.register_input(rnd(BL,Hd),True); Dpb=t3.register_input(rnd(BL,Hd),True); Hb=t3.register_input(rnd(BL,Hd),True)
    nr=t3.record_op(gc.OpType.RMSNorm,[R(embb,[BL,Hd]),R(w1b,[Hd])],[R(n1b,[BL,Hd])]); t3.save_for_backward(nr,[embb,w1b])
    nG=t3.record_op(gc.OpType.Linear,[R(n1b,[BL,Hd]),R(WGb,[Hd,Hd])],[R(Gpb,[BL,Hd])]); t3.save_for_backward(nG,[n1b,WGb])
    nV=t3.record_op(gc.OpType.Linear,[R(n1b,[BL,Hd]),R(WVb,[Hd,Hd])],[R(Vpb,[BL,Hd])]); t3.save_for_backward(nV,[n1b,WVb])
    nD=t3.record_op(gc.OpType.Linear,[R(n1b,[BL,Hd]),R(WDb,[Hd,Hd])],[R(Dpb,[BL,Hd])]); t3.save_for_backward(nD,[n1b,WDb])
    nM=t3.record_op(gc.OpType.MinGRU,[R(Gpb,[b,L,Hd]),R(Vpb,[b,L,Hd]),R(Dpb,[b,L,Hd])],[R(Hb,[b,L,Hd])]); t3.save_for_backward(nM,[Gpb,Vpb,Dpb,Hb])
    t3.backward(nM,gH); _=t3.read_buffer(t3.get_grad_buffer(embb),[BL,Hd])
    print("  t3 ok", flush=True)
print("ALL DONE (numpy-forward style, 3 iters)", flush=True)
