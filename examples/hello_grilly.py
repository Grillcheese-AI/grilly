"""Minimal forward + backward pass with Grilly autograd."""

import grilly.nn as nn

# Create a linear layer and random input with gradient tracking
layer = nn.Linear(128, 10)
x = nn.randn(32, 128, requires_grad=True)

# Forward pass
logits = x @ nn.Variable(layer.weight.T) + nn.Variable(layer.bias)
loss = logits.sum()

print(f"Input:  {x.data.shape}")
print(f"Output: {logits.data.shape}")

# Backward pass
loss.backward()

print(f"Grad:   {x.grad.shape}")
print("Done.")
