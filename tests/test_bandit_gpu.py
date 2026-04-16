
import numpy as np
import grilly_core
from cubemind.experimental.bandits import OnlineBanditSolver as RefSolver

def test_bandit_gpu_parity():
    # Setup
    n_instances = 3
    mu_hat = np.array([
        [0.8, 0.1, 0.99],
        [0.5, 0.9, 0.01], 
        [0.2, 0.3, 0.01],
        [0.1, 0.2, 0.01]
    ], dtype=np.float32)
    
    n_samples = np.array([
        [100, 10, 5000],
        [50, 200, 10],
        [30, 10, 10],
        [10, 5, 10]
    ], dtype=np.float32)
    
    iters = 200
    delta = 0.1
    
    # Python Reference
    ref_solver = RefSolver(mu_hat.shape[0], dist="gaussian")
    # Reference compute_optimal_proportions handles 2D input (K, n)
    w_ref = ref_solver.compute_optimal_proportions(mu_hat, iters=iters)
    
    # GPU Solver
    device = grilly_core.Device()
    # Need to load shaders if they are in a specific directory
    import os
    shader_dir = os.path.join(os.getcwd(), "shaders", "spv")
    device.load_shaders(shader_dir)
    
    res = grilly_core.bandit_solve(device, mu_hat, n_samples, iters, delta)
    w_gpu = res["target_w"]
    stop_gpu = res["stop_flags"]
    
    print("\nTarget W (Ref):\n", w_ref)
    print("Target W (GPU):\n", w_gpu)
    print("Stop Flags (GPU):", stop_gpu)
    
    # Parity check for proportions
    np.testing.assert_allclose(w_gpu, w_ref, atol=1e-2)
    
    # Check stopping logic
    # Instance 0: mu=[0.8, 0.5, 0.2, 0.1], N=[100, 50, 30, 10]. Likely should stop if total N is high enough.
    # Instance 1: mu=[0.1, 0.9, 0.3, 0.2], N=[10, 200, 10, 5]. 
    from cubemind.experimental.bandits import stop_criterion
    stop_ref0 = stop_criterion(mu_hat[:, 0], n_samples[:, 0], delta)
    stop_ref1 = stop_criterion(mu_hat[:, 1], n_samples[:, 1], delta)
    stop_ref2 = stop_criterion(mu_hat[:, 2], n_samples[:, 2], delta)
    
    assert stop_gpu[0] == (1 if stop_ref0 else 0)
    assert stop_gpu[1] == (1 if stop_ref1 else 0)
    assert stop_gpu[2] == (1 if stop_ref2 else 0)
    print("Instance 2 Stop (Ref/GPU):", stop_ref2, "/", stop_gpu[2])
    print("Parity OK")

if __name__ == "__main__":
    test_bandit_gpu_parity()
