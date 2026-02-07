Tutorial 01: Compute API Basics
===============================

Goal: get comfortable with low-level backend namespaces.

Step 1: Initialize compute
--------------------------

.. code-block:: python

   import numpy as np
   import grilly

   backend = grilly.Compute()

Step 2: Run FNN ops
-------------------

.. code-block:: python

   x = np.random.randn(16, 64).astype(np.float32)
   w = np.random.randn(64, 32).astype(np.float32)
   b = np.zeros(32, dtype=np.float32)

   y = backend.fnn.linear(x, w, b)
   y = backend.fnn.activation_relu(y)

Step 3: Run attention ops
-------------------------

.. code-block:: python

   q = np.random.randn(2, 4, 16, 32).astype(np.float32)
   k = np.random.randn(2, 4, 16, 32).astype(np.float32)
   v = np.random.randn(2, 4, 16, 32).astype(np.float32)

   out = backend.attention.flash_attention2(q, k, v)
   print(out.shape)

Step 4: Run memory search
-------------------------

.. code-block:: python

   query = np.random.randn(1, 128).astype(np.float32)
   database = np.random.randn(2000, 128).astype(np.float32)

   distances = backend.faiss.compute_distances(query, database)
   topk_values, topk_indices = backend.faiss.topk(distances, k=5)

Step 5: Cleanup
---------------

.. code-block:: python

   backend.cleanup()

You now know the main backend layout and how to call operation families directly.
