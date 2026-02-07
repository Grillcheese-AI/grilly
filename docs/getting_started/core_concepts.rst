Core Concepts
=============

This page explains how to think about Grilly as a framework.

1. Compute backend
------------------

`grilly.Compute()` is the Vulkan runtime entry point. It exposes operation groups:

- `backend.fnn` for feedforward ops
- `backend.attention` for attention kernels
- `backend.snn` for spiking ops
- `backend.memory` and `backend.faiss` for memory/retrieval

2. Module API
-------------

`grilly.nn` is designed like `torch.nn`:

- Subclass/compose `nn.Module`
- Call modules directly (`out = module(x)`)
- Use `model.parameters()` for optimizer input

3. Functional API
-----------------

`grilly.functional` mirrors `torch.nn.functional` style for stateless ops.

Example:

.. code-block:: python

   import grilly.functional as F
   y = F.relu(x)

4. Backward and gradients
-------------------------

Most core layers expose explicit `backward(...)` methods. Typical flow:

1. Forward pass through module(s)
2. Build a gradient tensor for model output
3. Call `model.backward(grad_output)`
4. Call `optimizer.step()`

5. CPU fallback behavior
------------------------

When a shader path is unavailable, many operations fall back to NumPy/CPU implementations.
This makes development easier on machines that do not fully support every kernel.
