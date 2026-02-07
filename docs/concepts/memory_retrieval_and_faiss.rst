Memory, Retrieval, and FAISS-style Search
=========================================

Retrieval primitives
--------------------

Grilly provides retrieval operations through `backend.memory` and `backend.faiss`.
Typical retrieval flow:

1. Encode query and database vectors.
2. Compute distances or similarity scores.
3. Select top-k candidates.
4. Fuse retrieved context into downstream model state.

FAISS-style APIs
----------------

Main entry points:

- `backend.faiss.compute_distances(queries, database, distance_type=...)`
- `backend.faiss.topk(distances, k)`

This supports semantic retrieval, routing, and nearest-neighbor tasks.

Memory APIs
-----------

The memory namespace includes operations for:

- memory read/write
- context aggregation
- memory injection patterns (concat, gate, residual)

These are useful for retrieval-augmented architectures where external context
must be merged with attention outputs.

Example search flow
-------------------

.. code-block:: python

   import numpy as np
   import grilly

   backend = grilly.Compute()

   queries = np.random.randn(4, 384).astype(np.float32)
   database = np.random.randn(10000, 384).astype(np.float32)

   distances = backend.faiss.compute_distances(queries, database)
   topk_values, topk_indices = backend.faiss.topk(distances, k=10)

   print(topk_values.shape, topk_indices.shape)

Use cases
---------

- semantic search
- memory-augmented generation
- expert routing
- nearest-neighbor classification
