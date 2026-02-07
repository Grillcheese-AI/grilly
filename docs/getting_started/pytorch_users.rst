For PyTorch Users
=================

If you already know PyTorch, use this map to get productive quickly.

Concept mapping
---------------

- `torch.nn.Module` -> `grilly.nn.Module`
- `torch.nn.Sequential` -> `grilly.nn.Sequential`
- `torch.nn.functional` -> `grilly.functional`
- `torch.optim.Adam` -> `grilly.optim.Adam`
- `torch.device("cuda")` -> `grilly.Compute()` (Vulkan backend)

Workflow mapping
----------------

PyTorch style:

.. code-block:: python

   out = model(x)
   loss.backward()
   optimizer.step()

Grilly style:

.. code-block:: python

   out = model(x)
   grad_out = ...  # gradient of loss w.r.t. out
   model.backward(grad_out)
   optimizer.step()

Parameter handling
------------------

Use `model.parameters()` with optimizers exactly like PyTorch:

.. code-block:: python

   optimizer = grilly.optim.Adam(model.parameters(), lr=1e-3)

Tensor type expectations
------------------------

Grilly primarily operates on NumPy arrays (`np.float32`).
If you pass PyTorch-like tensors into module calls, many module paths convert them internally.
