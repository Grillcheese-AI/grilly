Tutorial 08: Vector Search and Routing
======================================

Goal: build a semantic retrieval step and use top-k results for routing.

Step 1: Initialize backend and vectors
--------------------------------------

.. code-block:: python

   import numpy as np
   import grilly

   backend = grilly.Compute()

   queries = np.random.randn(3, 384).astype(np.float32)
   database = np.random.randn(20000, 384).astype(np.float32)

Step 2: Compute distances
-------------------------

.. code-block:: python

   distances = backend.faiss.compute_distances(queries, database)
   print("distance shape:", distances.shape)

Step 3: Get top-k neighbors
---------------------------

.. code-block:: python

   topk_values, topk_indices = backend.faiss.topk(distances, k=5)
   print("indices:", topk_indices.shape)

Step 4: Build retrieval context
-------------------------------

.. code-block:: python

   # Retrieve top candidates for query 0
   selected = database[topk_indices[0]]
   print("retrieved context:", selected.shape)

Step 5: Optional symbolic routing with bipolar keys
---------------------------------------------------

.. code-block:: python

   from grilly.experimental.vsa.ops import BinaryOps

   realm_names = ["health", "science", "finance"]
   codebook = np.stack([BinaryOps.hash_to_bipolar(r, 384) for r in realm_names], axis=0)
   query = BinaryOps.hash_to_bipolar("science", 384)

   sims = np.dot(codebook, query) / 384.0
   routed_realm = realm_names[int(np.argmax(sims))]
   print("routed realm:", routed_realm)
