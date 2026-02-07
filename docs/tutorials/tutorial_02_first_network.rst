Tutorial 02: Build Your First Network
=====================================

Goal: build a small MLP with the module API.

Step 1: Define the model
------------------------

.. code-block:: python

   import grilly.nn as nn

   model = nn.Sequential(
       nn.Linear(128, 256),
       nn.GELU(),
       nn.Linear(256, 64),
       nn.ReLU(),
       nn.Linear(64, 10),
   )

Step 2: Create synthetic data
-----------------------------

.. code-block:: python

   import numpy as np

   x = np.random.randn(32, 128).astype(np.float32)
   y_true = np.random.randn(32, 10).astype(np.float32)

Step 3: Forward pass
--------------------

.. code-block:: python

   y_pred = model(x)
   print("Prediction shape:", y_pred.shape)

Step 4: Compute a basic loss
----------------------------

.. code-block:: python

   loss = np.mean((y_pred - y_true) ** 2)
   print("MSE loss:", float(loss))

Step 5: Build output gradient
-----------------------------

For MSE, dL/dy is:

.. code-block:: python

   grad_out = (2.0 / y_true.size) * (y_pred - y_true)

This gradient is the input for backward propagation in the next tutorial.
