"""Parity test: resident forward_mingru (GPU) vs numpy MinGRU reference.
Validates the newly-wired TapeContext.forward_mingru against the same
sequential-scan reference used by train_full_resident.py."""
import sys
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np
import grilly_core as gc

np.random.seed(0)
B, L, Hd = 4, 6, 24

def sig(z): return 1.0/(1.0+np.exp(-z))
def mingru_fwd(G, V_, D):
    sg=sig(G); tv=np.tanh(V_); sd=sig(D); xs=sg*tv; a=0.001+0.998*sd
    H=np.zeros_like(G)
    for t in range(L):
        prev=H[:,t-1,:] if t>0 else 0.0
        H[:,t,:]=a[:,t,:]*prev+xs[:,t,:]
    return H

G=(0.7*np.random.randn(B,L,Hd)).astype(np.float32)
V=(0.7*np.random.randn(B,L,Hd)).astype(np.float32)
D=(0.7*np.random.randn(B,L,Hd)).astype(np.float32)
H_ref=mingru_fwd(G,V,D)

dev=gc.Device(); dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")
t=gc.TapeContext(dev); t.begin()
g_id=t.register_input(G.reshape(-1),True)
v_id=t.register_input(V.reshape(-1),True)
d_id=t.register_input(D.reshape(-1),True)
t.forward_begin()
h_id=t.forward_mingru(g_id,v_id,d_id,B,L,Hd)
t.forward_submit()
H_gpu=t.read_buffer(h_id,[B,L,Hd])

err=float(np.abs(H_gpu-H_ref).max())
rel=err/(float(np.abs(H_ref).max())+1e-9)
print("forward_mingru parity: max_abs_diff=%.3e  rel=%.3e" % (err, rel))
print("RESULT:", "PASS" if rel < 1e-4 else "FAIL")
sys.exit(0 if rel < 1e-4 else 1)
