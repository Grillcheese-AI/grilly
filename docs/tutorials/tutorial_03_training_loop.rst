Tutorial 03: End-to-End Training Loop
=====================================

Goal: run a full loop with forward, backward, and optimizer steps.

Step 1: Setup model and optimizer
---------------------------------

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

Step 2: Create training batch
-----------------------------

.. code-block:: python

   x = np.random.randn(64, 128).astype(np.float32)
   y = np.random.randn(64, 10).astype(np.float32)

Step 3: Forward
---------------

.. code-block:: python

   pred = model(x)
   loss = np.mean((pred - y) ** 2)

Step 4: Backward
----------------

.. code-block:: python

   grad_out = (2.0 / y.size) * (pred - y)
   model.zero_grad()
   model.backward(grad_out)

Step 5: Parameter update
------------------------

.. code-block:: python

   optimizer.step()

Step 6: Repeat over epochs
--------------------------

.. code-block:: python

   for epoch in range(20):
       pred = model(x)
       loss = np.mean((pred - y) ** 2)
       grad_out = (2.0 / y.size) * (pred - y)
       model.zero_grad()
       model.backward(grad_out)
       optimizer.step()
       print(f"epoch={epoch} loss={float(loss):.6f}")

You now have the standard framework loop pattern you can adapt to real datasets.
