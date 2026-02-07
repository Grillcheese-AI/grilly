Training and Optimization
=========================

Training model in Grilly
------------------------

Grilly training uses explicit gradient flow:

1. Forward pass through modules.
2. Compute loss value.
3. Build gradient with respect to model output.
4. Call backward on modules/containers.
5. Call optimizer step.

Unlike frameworks with global autograd by default, many Grilly paths expose
direct backward methods per module.

Canonical loop
--------------

.. code-block:: python

   import numpy as np
   import grilly.nn as nn
   import grilly.optim as optim

   model = nn.Sequential(
       nn.Linear(128, 256),
       nn.GELU(),
       nn.Linear(256, 10),
   )
   optimizer = optim.Adam(model.parameters(), lr=1e-3)

   x = np.random.randn(64, 128).astype(np.float32)
   y = np.random.randn(64, 10).astype(np.float32)

   pred = model(x)
   loss = np.mean((pred - y) ** 2)
   grad_out = (2.0 / y.size) * (pred - y)

   model.zero_grad()
   model.backward(grad_out)
   optimizer.step()

Optimizers
----------

`grilly.optim` includes:

- `Adam`, `AdamW`
- `SGD`
- `NLMS`
- `NaturalGradient`
- scheduler utilities (`StepLR`, `CosineAnnealingLR`, etc.)

The optimizer layer can use GPU-backed update paths when available, with CPU
fallback behavior when needed.

Gradient lifecycle
------------------

Typical parameter lifecycle:

- `param.grad` is written during backward.
- `optimizer.step()` consumes gradient and updates parameter.
- `zero_grad()` clears or resets gradients for the next step.

Numerical and runtime tips
--------------------------

1. Keep all training arrays in `float32`.
2. Verify gradients are finite before optimizer step.
3. Start with small learning rates for custom module stacks.
4. For long runs, checkpoint model/optimizer state periodically.

Related advanced learning ops
-----------------------------

Beyond standard optimizers, Grilly includes GPU-accelerated learning kernels:

- Fisher information updates and EWC penalties.
- NLMS prediction and updates.
- Whitening transforms.
- Contrastive and specialized loss helpers.
