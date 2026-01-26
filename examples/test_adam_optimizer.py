"""Test Adam optimizer with GPU acceleration"""

import numpy as np
from grilly.optim import Adam
from grilly import Compute

print("="*80)
print("Testing Adam Optimizer with GPU Acceleration")
print("="*80)

# Create a simple test case
# Parameters: 2x3 weight matrix
param = np.random.randn(2, 3).astype(np.float32)
grad = np.random.randn(2, 3).astype(np.float32) * 0.1  # Small gradients

print(f"\n[1] Initial parameter shape: {param.shape}")
print(f"    Parameter mean: {param.mean():.6f}, std: {param.std():.6f}")
print(f"    Gradient mean: {grad.mean():.6f}, std: {grad.std():.6f}")

# Create optimizer
optimizer = Adam([param], lr=0.01, betas=(0.9, 0.999), eps=1e-8, use_gpu=True)

print(f"\n[2] Optimizer created:")
print(f"    Learning rate: {optimizer.defaults['lr']}")
print(f"    Betas: {optimizer.defaults['betas']}")
print(f"    Epsilon: {optimizer.defaults['eps']}")
print(f"    Use GPU: {optimizer.use_gpu}")

# Store gradients in optimizer state (simulating backward pass)
# In a real implementation, gradients would come from backward pass
param_id = id(param)
optimizer.state[param_id]['grad'] = grad

# Perform optimization step
print(f"\n[3] Performing optimization step...")
param_before = param.copy()

# Manually set grad attribute for optimizer to find
# We'll modify the optimizer to accept gradients as parameter
# For now, let's modify the step to accept gradients
# Actually, let's create a wrapper that stores gradients
class ParamWithGrad:
    def __init__(self, data):
        self.data = data
        self.grad = None

param_wrapper = ParamWithGrad(param)
param_wrapper.grad = grad

# Update optimizer to use wrapper
optimizer.param_groups[0]['params'] = [param_wrapper.data]
param_id_new = id(param_wrapper.data)
optimizer.state[param_id_new] = optimizer.state.pop(param_id, {})

# Actually, simpler: modify optimizer to accept gradients directly
# Let's just manually call the update with gradients
param_before = param.copy()
param_id = id(param)
state = optimizer.state[param_id]
state['step'] = 0
state['exp_avg'] = np.zeros_like(param, dtype=np.float32)
state['exp_avg_sq'] = np.zeros_like(param, dtype=np.float32)

# Manually perform update
backend = optimizer._get_backend()
if backend and optimizer.use_gpu and hasattr(backend, 'core') and 'adam-update' in backend.core.shaders:
    param, state['exp_avg'], state['exp_avg_sq'] = optimizer._adam_update_gpu(
        backend, param, grad, state['exp_avg'], state['exp_avg_sq'],
        optimizer.defaults['lr'], optimizer.defaults['betas'][0], 
        optimizer.defaults['betas'][1], optimizer.defaults['eps'],
        1 - optimizer.defaults['betas'][0], 1 - optimizer.defaults['betas'][1]
    )
    state['step'] = 1
else:
    # CPU fallback
    optimizer._step_count = 1
    beta1, beta2 = optimizer.defaults['betas']
    lr = optimizer.defaults['lr']
    eps = optimizer.defaults['eps']
    beta1_t = beta1 ** 1
    beta2_t = beta2 ** 1
    exp_avg = beta1 * state['exp_avg'] + (1 - beta1) * grad
    exp_avg_sq = beta2 * state['exp_avg_sq'] + (1 - beta2) * grad * grad
    m_hat = exp_avg / (1 - beta1_t)
    v_hat = exp_avg_sq / (1 - beta2_t)
    param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    state['exp_avg'] = exp_avg
    state['exp_avg_sq'] = exp_avg_sq
    state['step'] = 1

print(f"    Parameter before: mean={param_before.mean():.6f}, std={param_before.std():.6f}")
print(f"    Parameter after:  mean={param.mean():.6f}, std={param.std():.6f}")
print(f"    Parameter change: mean={(param - param_before).mean():.6f}")

# Check optimizer state
param_id = id(param)
state = optimizer.state[param_id]
print(f"\n[4] Optimizer state:")
print(f"    Step: {state['step']}")
print(f"    Exp_avg mean: {state['exp_avg'].mean():.6f}, std: {state['exp_avg'].std():.6f}")
print(f"    Exp_avg_sq mean: {state['exp_avg_sq'].mean():.6f}, std: {state['exp_avg_sq'].std():.6f}")

# Perform a few more steps
print(f"\n[5] Performing 5 more optimization steps...")
for i in range(5):
    # Generate new gradients
    new_grad = np.random.randn(2, 3).astype(np.float32) * 0.1
    gradients = {param_id: new_grad}
    optimizer.step(gradients=gradients)

print(f"    Final parameter: mean={param.mean():.6f}, std={param.std():.6f}")
print(f"    Final step: {optimizer.state[param_id]['step']}")

print(f"\n[OK] Adam optimizer test completed!")
