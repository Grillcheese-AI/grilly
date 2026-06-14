"""Parity test: resident forward_embedding (GPU) vs numpy E[ids] (exact gather)."""
import sys
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\grilly")
import numpy as np
import grilly_core as gc

np.random.seed(0)
Vv, d, B, S = 50, 16, 3, 5
E = (0.1*np.random.randn(Vv, d)).astype(np.float32)
ids = np.random.randint(0, Vv, size=(B, S)).astype(np.uint32)
ref = E[ids]  # (B,S,d)

dev=gc.Device(); dev.load_shaders(r"C:\Users\grill\Documents\GitHub\grilly\shaders\spv")
t=gc.TapeContext(dev); t.begin()
ids_id=t.register_input_u32(ids.reshape(-1))
tbl_id=t.register_input(E.reshape(-1),True)
t.forward_begin()
o_id=t.forward_embedding(ids_id, tbl_id, B, S, Vv, d)
t.forward_submit()
out=t.read_buffer(o_id,[B,S,d])
err=float(np.abs(out-ref).max())
print("forward_embedding parity: max_abs_diff=%.3e" % err)
print("RESULT:", "PASS" if err == 0.0 else "FAIL")
sys.exit(0 if err == 0.0 else 1)
