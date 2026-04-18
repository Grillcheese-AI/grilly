import torch
import numpy as np
from grilly.nn.autograd import Variable, sigmoid, tanh
from grilly.nn.prefix_scan import prefix_scan_causal, min_gru

def test_mingru_parity():
    B, S, D = 2, 16, 8
    
    # Random inputs
    g = np.random.randn(B, S, D).astype(np.float32)
    v = np.random.randn(B, S, D).astype(np.float32)
    d = np.random.randn(B, S, D).astype(np.float32)
    
    gv = Variable(g, requires_grad=True)
    vv = Variable(v, requires_grad=True)
    dv = Variable(d, requires_grad=True)
    
    # 1. Reference Implementation (Python)
    x_scan = sigmoid(gv) * tanh(vv)
    a = 0.001 + sigmoid(dv) * 0.998
    h_ref = prefix_scan_causal(x_scan, a)    
    # 2. Fused Implementation
    h_fused = min_gru(gv, vv, dv)
    
    # Forward check
    np.testing.assert_allclose(h_fused.data, h_ref.data, atol=1e-5, rtol=1e-5)
    print("Forward Parity: OK")
    
    # Backward check
    loss_ref = (h_ref * h_ref).sum()
    loss_ref.backward()
    
    grad_g_ref = np.array(gv.grad.data).copy()
    grad_v_ref = np.array(vv.grad.data).copy()
    grad_d_ref = np.array(dv.grad.data).copy()
    
    gv.zero_grad()
    vv.zero_grad()
    dv.zero_grad()
    
    loss_fused = (h_fused * h_fused).sum()
    loss_fused.backward()
    
    grad_g_fused = np.array(gv.grad.data)
    grad_v_fused = np.array(vv.grad.data)
    grad_d_fused = np.array(dv.grad.data)
    
    np.testing.assert_allclose(grad_g_fused, grad_g_ref, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(grad_v_fused, grad_v_ref, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(grad_d_fused, grad_d_ref, atol=1e-4, rtol=1e-4)
    print("Backward Parity: OK")

if __name__ == "__main__":
    # Ensure bridge is loaded (loads shaders/spv)
    import grilly
    test_mingru_parity()
