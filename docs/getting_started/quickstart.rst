Quickstart
==========

This quickstart walks through your first Grilly workload.

Step 1: Import and initialize
-----------------------------

.. code-block:: python

   import numpy as np
   import grilly

   backend = grilly.Compute()

Step 2: Run a feedforward operation
-----------------------------------

.. code-block:: python

   x = np.random.randn(32, 384).astype(np.float32)
   w = np.random.randn(384, 128).astype(np.float32)
   b = np.zeros(128, dtype=np.float32)

   y = backend.fnn.linear(x, w, b)
   y = backend.fnn.activation_gelu(y)

   print("Output shape:", y.shape)

Step 3: Run a neural module (PyTorch-style)
-------------------------------------------

.. code-block:: python

   import grilly.nn as nn

   model = nn.Sequential(
       nn.Linear(384, 256),
       nn.GELU(),
       nn.Linear(256, 10),
   )

   logits = model(x)
   print("Logits shape:", logits.shape)

Step 4: Use a simple training step
----------------------------------

.. code-block:: python

   target = np.random.randn(32, 10).astype(np.float32)
   grad_out = (2.0 / target.size) * (logits - target)

   model.zero_grad()
   model.backward(grad_out)

Step 5: Update weights
----------------------

.. code-block:: python

   import grilly.optim as optim

   optimizer = optim.Adam(model.parameters(), lr=1e-3)
   optimizer.step()

At this point you have run forward, backward, and optimizer update in the same style as early PyTorch workflows.
