"""Test backward pass integration with optimizer"""

import numpy as np
from grilly.nn import Linear, Sequential
from grilly.optim import Adam
from grilly import functional

print("="*80)
print("Testing Backward Pass Integration with Optimizer")
print("="*80)

# Create a simple model
model = Sequential(
    Linear(10, 5),
    Linear(5, 1)
)

print(f"\n[1] Model created:")
print(f"    Parameters: {sum(p.size for p in model.parameters())}")

# Create optimizer
optimizer = Adam(model.parameters(), lr=0.01, use_gpu=True)

print(f"\n[2] Optimizer created:")
print(f"    Learning rate: {optimizer.defaults['lr']}")

# Create dummy data
batch_size = 4
x = np.random.randn(batch_size, 10).astype(np.float32)
target = np.random.randn(batch_size, 1).astype(np.float32)

print(f"\n[3] Input data:")
print(f"    Input shape: {x.shape}")
print(f"    Target shape: {target.shape}")

# Forward pass
print(f"\n[4] Forward pass...")
output = model(x)
print(f"    Output shape: {output.shape}")
print(f"    Output mean: {output.mean():.6f}, std: {output.std():.6f}")

# Compute loss (MSE)
loss = np.mean((output - target) ** 2)
print(f"\n[5] Loss: {loss:.6f}")

# Backward pass (manual for now - compute gradients)
print(f"\n[6] Backward pass...")

# Loss gradient w.r.t. output
grad_output = 2.0 * (output - target) / batch_size

# Backward through layers (in reverse order)
# For Sequential, we need to manually backprop
# In a full implementation, this would be automatic

# Get layers
layers = [model._modules[str(i)] for i in range(len(model._modules))]

# Backward through last layer
grad = grad_output
for i in range(len(layers) - 1, -1, -1):
    layer = layers[i]
    if hasattr(layer, 'backward'):
        # Get input from forward pass (would be stored in real implementation)
        # For now, we'll do a forward pass to get intermediate activations
        # In practice, these would be cached during forward pass
        if i == 0:
            layer_input = x
        else:
            # Recompute intermediate activation
            layer_input = layers[0](x)
        grad = layer.backward(grad, layer_input)
    else:
        # Fallback: compute gradient manually
        if isinstance(layer, Linear):
            # Simple gradient computation
            if i == len(layers) - 1:
                # Last layer
                layer_input = layers[0](x) if i > 0 else x
                grad_weight = grad.T @ layer_input
                grad_bias = np.sum(grad, axis=0)
                
                if layer.weight is not None:
                    layer.weight.grad = grad_weight
                if layer.bias is not None:
                    layer.bias.grad = grad_bias
                
                grad = grad @ layer.weight
            else:
                # Intermediate layer
                layer_input = x if i == 0 else layers[0](x)
                grad_weight = grad.T @ layer_input
                grad_bias = np.sum(grad, axis=0)
                
                if layer.weight is not None:
                    layer.weight.grad = grad_weight
                if layer.bias is not None:
                    layer.bias.grad = grad_bias
                
                grad = grad @ layer.weight

print(f"    Gradients computed for all parameters")

# Check gradients
print(f"\n[7] Checking gradients...")
for name, param in model.named_parameters():
    if hasattr(param, 'grad') and param.grad is not None:
        print(f"    {name}: grad shape={param.grad.shape}, mean={param.grad.mean():.6f}")
    else:
        print(f"    {name}: No gradient")

# Optimizer step
print(f"\n[8] Optimizer step...")
param_before = {}
for name, param in model.named_parameters():
    param_before[name] = param.copy()

optimizer.step()

print(f"    Parameters updated")
for name, param in model.named_parameters():
    if name in param_before:
        change = np.abs(param - param_before[name]).mean()
        print(f"    {name}: mean change={change:.6f}")

print(f"\n[OK] Backward pass and optimizer integration test completed!")
