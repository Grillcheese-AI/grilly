Tutorial 04: Attention and Memory Workflow
==========================================

Goal: combine attention output with memory retrieval.

Step 1: Prepare attention inputs
--------------------------------

.. code-block:: python

   import numpy as np
   import grilly

   backend = grilly.Compute()

   batch = 4
   heads = 8
   seq = 32
   head_dim = 64

   q = np.random.randn(batch, heads, seq, head_dim).astype(np.float32)
   k = np.random.randn(batch, heads, seq, head_dim).astype(np.float32)
   v = np.random.randn(batch, heads, seq, head_dim).astype(np.float32)

Step 2: Compute attention output
--------------------------------

.. code-block:: python

   attn_out = backend.attention.flash_attention2(q, k, v)
   print("attention output:", attn_out.shape)

Step 3: Build memory database
-----------------------------

.. code-block:: python

   query = np.random.randn(1, 256).astype(np.float32)
   database = np.random.randn(5000, 256).astype(np.float32)

Step 4: Retrieve nearest vectors
--------------------------------

.. code-block:: python

   distances = backend.faiss.compute_distances(query, database)
   topk_values, topk_indices = backend.faiss.topk(distances, k=8)
   retrieved = database[topk_indices[0]]

Step 5: Use retrieved context
-----------------------------

At this point you can:

1. Concatenate retrieved vectors with model state.
2. Inject retrieved context before the next decoder/FFN block.
3. Re-rank or route candidates with additional similarity passes.

Step 6: Cleanup
---------------

.. code-block:: python

   backend.cleanup()
