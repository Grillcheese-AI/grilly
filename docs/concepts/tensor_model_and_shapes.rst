Tensor Model and Shapes
=======================

Primary tensor type
-------------------

Grilly APIs are built around NumPy arrays.

- Most compute paths expect `np.float32`.
- Some indexing paths use integer dtypes (`np.int32`, `np.int64`).
- Several modules can accept tensor-like objects and convert internally.

Shape conventions
-----------------

Common conventions used by Grilly modules:

- Dense feedforward: `(batch, features)` or `(batch, seq, features)`
- Conv2d family: `(batch, channels, height, width)`
- Attention (module-level): `(batch, seq, embed_dim)`
- Flash attention backend path: often `(batch, heads, seq, head_dim)`
- Memory search: queries `(Q, D)`, database/codebook `(N, D)`

Why shape discipline matters
----------------------------

Many kernels dispatch with explicit shape-derived workgroups. Wrong layout can
silently hurt performance or break correctness.

Best practices:

1. Normalize dtype and layout before call boundaries.
2. Keep tensors contiguous when possible.
3. Explicitly print shapes in early pipeline debugging.

Example input guards
--------------------

.. code-block:: python

   import numpy as np

   def ensure_f32(x):
       x = np.asarray(x)
       if x.dtype != np.float32:
           x = x.astype(np.float32)
       return x

   def expect_2d(x):
       if x.ndim != 2:
           raise ValueError(f"expected 2D tensor, got shape {x.shape}")
       return x

Parameter and gradient storage
------------------------------

`grilly.nn` parameters are stored as parameter-like arrays with optional `.grad`.
Backward calls populate `.grad`, and optimizers consume those gradients.

For stable training loops:

1. Forward pass.
2. Build output gradient for your loss.
3. `model.zero_grad()`
4. `model.backward(grad_output)`
5. `optimizer.step()`
